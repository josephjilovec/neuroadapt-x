import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Assume EEGNet is importable from the same directory for a clean structure
# For testing purposes below, we'll redefine the relevant class
try:
    from .eegnet import EEGNet
except ImportError:
    # This block is for direct execution testing only
    class EEGNet(nn.Module):
        def __init__(self, n_channels, n_times, n_classes, F1=8, D=2, kernel_T=64, P1=8, P2=4, dropout_rate=0.25):
            super().__init__()
            F2 = F1 * D
            self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_T), padding=(0, kernel_T // 2), bias=False)
            self.depthwise_conv = nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False)
            self.bn1 = nn.BatchNorm2d(F2) # Placeholder BN layer
            self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, P1), stride=(1, P1))
            self.drop1 = nn.Dropout(dropout_rate)

            kernel_S = 16
            self.separable_conv_depth = nn.Conv2d(F2, F2, kernel_size=(1, kernel_S), padding=(0, kernel_S // 2), groups=F2, bias=False)
            self.separable_conv_point = nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)
            self.bn2 = nn.BatchNorm2d(F2) # Placeholder BN layer
            self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, P2), stride=(1, P2))
            self.drop2 = nn.Dropout(dropout_rate)

            final_time_dim = n_times // P1 // P2
            self.n_features_before_fc = F2 * final_time_dim
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.n_features_before_fc, n_classes))

        def forward(self, x):
            # Simplified forward for placeholder, actual implementation is in eegnet.py
            x = x.unsqueeze(1) 
            x = F.relu(self.depthwise_conv(self.conv1(x)))
            x = self.bn1(x)
            x = self.avg_pool1(x)
            x = self.drop1(x)
            x = F.relu(self.separable_conv_point(self.separable_conv_depth(x)))
            x = self.bn2(x)
            x = self.avg_pool2(x)
            features = self.drop2(x) # Features used for CORAL
            
            logits = self.classifier(features)
            return logits, features.flatten(start_dim=1)

# --- 1. Adaptive Batch Normalization (AdaBN) ---

class AdaBN2d(nn.Module):
    """
    Adaptive Batch Normalization (AdaBN) layer for 2D convolutions.
    
    This layer allows the mean and variance statistics to be adapted 
    to the target domain (stressed data) while keeping the source domain's 
    running statistics frozen when operating in adaptation mode.
    
    Crucially, during adaptation, this layer computes and uses the statistics
    of the *current batch* for the target data.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaBN2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Standard BN parameters (for source domain, trained offline)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        # Running statistics (Source domain stats)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Adaptation status flag
        self.is_adapt = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If is_adapt=True, calculates and uses batch statistics (target domain).
        If is_adapt=False, uses running statistics (source domain, standard BN).
        """
        # Determine the mean and variance to use
        if self.training and not self.is_adapt:
            # Source Domain Training (Standard BN)
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            
            # Update running stats for source domain
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
        elif self.training and self.is_adapt:
            # Target Domain Adaptation (AdaBN - Use Batch Stats)
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            
            # Note: Running stats are NOT updated during adaptation
            
        else:
            # Evaluation (Inference) Mode - Use frozen running source stats
            mean = self.running_mean
            var = self.running_var

        # Normalize and scale/shift (same for all modes)
        x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x

    def adapt(self):
        """Sets the layer to adaptation mode (use batch stats for target)."""
        self.is_adapt = True

    def train(self, mode=True):
        """Sets the layer to training mode (use standard source BN rules)."""
        super(AdaBN2d, self).train(mode)
        self.is_adapt = False
        return self

# --- 2. CORAL Loss ---

class CORALLoss(nn.Module):
    """
    Correlation Alignment (CORAL) loss for domain adaptation.
    Minimizes the distance between the covariance matrices of source and target features.
    
    L_CORAL = (1 / 4 * d^2) * ||C_S - C_T||_F^2
    Where C_S and C_T are covariance matrices, and d is feature dimension.
    """
    def __init__(self):
        super(CORALLoss, self).__init__()

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        Calculates the CORAL loss.

        Args:
            source_features: Tensor of features from the source domain (N_s, D).
            target_features: Tensor of features from the target domain (N_t, D).

        Returns:
            The CORAL loss value (scalar).
        """
        def compute_covariance(features):
            n = features.size(0)
            d = features.size(1)
            
            # Center the features: F - mean(F)
            mean_f = torch.mean(features, dim=0, keepdim=True)
            centered_f = features - mean_f
            
            # Compute covariance matrix: (F^T * F) / (N-1)
            # We use N instead of N-1 as N is often large in ML contexts
            cov_f = torch.mm(centered_f.t(), centered_f) / n
            return cov_f

        # Calculate covariance matrices
        cov_s = compute_covariance(source_features)
        cov_t = compute_covariance(target_features)
        
        # Calculate the Frobenius norm of the difference
        loss = torch.sum(torch.pow(cov_s - cov_t, 2))
        
        # Scale by 1/(4*d^2) as per original CORAL paper (optional, often absorbed in lambda)
        # However, for simplicity and stability, we just return the sum of squared differences
        
        return loss

# --- 3. Adaptive EEGNet Wrapper ---

class AdaptiveEEGNet(nn.Module):
    """
    EEGNet model extended for domain adaptation using AdaBN and feature exposure 
    for CORAL loss calculation.
    """
    def __init__(self, base_eegnet: EEGNet):
        super(AdaptiveEEGNet, self).__init__()
        
        # --- Copy Base EEGNet Architecture ---
        
        # Use the base model structure
        self.conv1 = base_eegnet.conv1
        self.depthwise_conv = base_eegnet.depthwise_conv
        self.elu1 = base_eegnet.elu1
        self.avg_pool1 = base_eegnet.avg_pool1
        self.drop1 = base_eegnet.drop1
        
        self.separable_conv_depth = base_eegnet.separable_conv_depth
        self.separable_conv_point = base_eegnet.separable_conv_point
        self.elu2 = base_eegnet.elu2
        self.avg_pool2 = base_eegnet.avg_pool2
        self.drop2 = base_eegnet.drop2
        
        self.classifier = base_eegnet.classifier
        
        # --- Replace Standard BN layers with AdaBN layers ---
        
        F2 = base_eegnet.bn1.num_features # Get F2 value from base BN layer
        
        # Block 1 BN replacement
        self.bn1 = AdaBN2d(F2)
        # Copy weights/bias if they exist (assuming the base model was trained)
        self.bn1.weight.data.copy_(base_eegnet.bn1.weight.data)
        self.bn1.bias.data.copy_(base_eegnet.bn1.bias.data)
        self.bn1.running_mean.copy_(base_eegnet.bn1.running_mean)
        self.bn1.running_var.copy_(base_eegnet.bn1.running_var)

        # Block 2 BN replacement
        self.bn2 = AdaBN2d(F2)
        self.bn2.weight.data.copy_(base_eegnet.bn2.weight.data)
        self.bn2.bias.data.copy_(base_eegnet.bn2.bias.data)
        self.bn2.running_mean.copy_(base_eegnet.bn2.running_mean)
        self.bn2.running_var.copy_(base_eegnet.bn2.running_var)
        
        # This flag controls the behavior of the entire network
        self._is_adapt = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass, returning both logits and the features
        needed for CORAL loss calculation.
        
        Returns: (logits, features_for_alignment)
        """
        x = x.unsqueeze(1) # (N, C, T) -> (N, 1, C, T)
        
        # --- Block 1 ---
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.bn1(x) # Adaptive BN 1
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.drop1(x)
        
        # --- Block 2 ---
        x = self.separable_conv_depth(x)
        x = self.separable_conv_point(x)
        x = self.bn2(x) # Adaptive BN 2
        x = self.elu2(x)
        x = self.avg_pool2(x)
        features = self.drop2(x)
        
        # Features for CORAL alignment (flattened but before FC layer)
        features_flat = features.flatten(start_dim=1)
        
        # --- Classification Head ---
        logits = self.classifier(features_flat)
        
        return logits, features_flat
    
    def set_adaptation_mode(self, is_adapt: bool):
        """
        Sets the AdaBN layers to use batch statistics (is_adapt=True) 
        or use running source statistics (is_adapt=False).
        """
        self._is_adapt = is_adapt
        for module in self.modules():
            if isinstance(module, AdaBN2d):
                if is_adapt:
                    module.adapt()
                else:
                    module.train() # Resets is_adapt flag within AdaBN2d

# --- Main Execution for Testing ---

if __name__ == '__main__':
    # Test Parameters (Must match eegnet.py)
    N_CHANNELS = 3
    N_TIMES = 400 
    N_CLASSES = 2
    BATCH_SIZE = 16
    
    print("--- Testing Adaptive Model Components ---")

    # 1. Test CORAL Loss
    print("\n[1] Testing CORAL Loss...")
    D_features = 100 # Example feature dimension
    S_features = torch.randn(BATCH_SIZE, D_features) # Source Batch
    T_features = torch.randn(BATCH_SIZE, D_features) * 5 + 10 # Target Batch (simulated shift)
    
    coral_criterion = CORALLoss()
    loss = coral_criterion(S_features, T_features)
    print(f"CORAL Loss (Source vs Stressed): {loss.item():.4f}")
    
    # Loss should be low if domains are similar
    S_features_2 = S_features * 1.0001
    loss_similar = coral_criterion(S_features, S_features_2)
    print(f"CORAL Loss (Source vs Similar): {loss_similar.item():.4f}")
    
    assert loss_similar < loss, "CORAL loss should be higher for stressed data."
    print("CORAL Loss function passed basic check.")

    # 2. Test AdaptiveEEGNet
    print("\n[2] Testing AdaptiveEEGNet...")
    
    # Create and pre-train (simulate) a baseline EEGNet
    base_model = EEGNet(N_CHANNELS, N_TIMES, N_CLASSES)
    # Simulate loading pre-trained weights/running stats
    print("Simulating pre-trained base model.")

    # Wrap the base model with the Adaptive layer
    adaptive_model = AdaptiveEEGNet(base_model)
    print(f"Adaptive model created. Base BN layers replaced by AdaBN2d.")
    
    dummy_input = torch.randn(BATCH_SIZE, N_CHANNELS, N_TIMES)
    
    # Test 2a: Source/Evaluation Mode (Uses frozen running stats)
    adaptive_model.set_adaptation_mode(False)
    adaptive_model.eval()
    logits_eval, features_eval = adaptive_model(dummy_input)
    print(f"Eval Mode Output Logits shape: {tuple(logits_eval.shape)}")

    # Test 2b: Adaptation Mode (Uses batch stats)
    adaptive_model.set_adaptation_mode(True)
    adaptive_model.train() # Must be in train mode for AdaBN to use batch stats
    logits_adapt, features_adapt = adaptive_model(dummy_input)
    print(f"Adapt Mode Output Logits shape: {tuple(logits_adapt.shape)}")
    
    # A simple check: the outputs *should* be slightly different because the
    # normalization statistics are computed from the small batch.
    diff = torch.sum(torch.abs(logits_eval - logits_adapt)).item()
    print(f"Sum of absolute logit difference (Eval vs Adapt): {diff:.4f}")
    
    assert diff > 1e-4, "Outputs should differ when switching from running stats (Eval) to batch stats (Adapt)."
    print("AdaptiveEEGNet forward pass and mode switching successful.")
