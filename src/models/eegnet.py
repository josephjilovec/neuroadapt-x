import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# --- EEGNet Architecture (Original and popular structure) ---

class EEGNet(nn.Module):
    """
    A compact convolutional neural network for EEG-based brain-computer interfaces (BCIs).
    
    This implementation follows the structure proposed by V. J. Lawhern et al. (2018).
    
    The input data shape is assumed to be (batch_size, n_channels, n_times).
    The model internally expects the input to be (batch_size, 1, n_channels, n_times).
    """

    def __init__(
        self, 
        n_channels: int, 
        n_times: int, 
        n_classes: int, 
        F1: int = 8,           # Number of features maps in temporal conv
        D: int = 2,            # Depth multiplier for Depthwise Conv
        F2: int = 16,          # Number of feature maps in Separable Conv (F2 = F1 * D)
        kernel_T: int = 64,    # Kernel size for temporal convolution
        P1: int = 8,           # Pooling factor for Block 1
        P2: int = 4,           # Pooling factor for Block 2
        dropout_rate: float = 0.25
    ):
        """
        Initializes the EEGNet model layers.

        Args:
            n_channels (int): Number of EEG channels (e.g., 3 for C3/Cz/C4).
            n_times (int): Number of time points per epoch (e.g., 400).
            n_classes (int): Number of classification targets (e.g., 2 for Left/Right MI).
            F1 (int): Number of temporal filters.
            D (int): Depth multiplier (number of spatial filters per temporal filter).
            F2 (int): Number of separable conv filters (must be F1*D).
            kernel_T (int): Kernel size for the initial temporal convolution.
            P1 (int): Pooling window for Block 1.
            P2 (int): Pooling window for Block 2.
            dropout_rate (float): Dropout probability.
        """
        super(EEGNet, self).__init__()
        
        # Ensure F2 = F1 * D
        F2 = F1 * D
        
        # --- Block 1: Temporal and Spatial Feature Extraction ---
        
        # Layer 1: Temporal Convolution
        # Input shape: (N, 1, n_channels, n_times)
        # Output shape: (N, F1, n_channels, n_times - kernel_T + 1)
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=F1, 
            kernel_size=(1, kernel_T), 
            padding=(0, kernel_T // 2), # Maintain time dimension size
            bias=False
        )

        # Layer 2: Depthwise Convolution (Spatial Filtering)
        # Groups = F1 ensures the filter only operates on its own set of feature maps
        # Input shape: (N, F1, n_channels, time_dim)
        # Output shape: (N, F1 * D, 1, time_dim) -> (N, F2, 1, time_dim)
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1, 
            out_channels=F1 * D, 
            kernel_size=(n_channels, 1), 
            groups=F1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        
        # Layer 3: Average Pooling 1
        # Pools along the time dimension
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, P1), stride=(1, P1))
        self.drop1 = nn.Dropout(dropout_rate)
        
        # --- Block 2: Separable Convolution (Further Temporal Feature Learning) ---
        
        # Layer 4: Separable Conv (Depthwise Part)
        # Input shape: (N, F2, 1, time_dim / P1)
        # Output shape: (N, F2, 1, time_dim / P1 - kernel_S + 1)
        kernel_S = 16 # Kernel size for separable convolution
        self.separable_conv_depth = nn.Conv2d(
            in_channels=F2, 
            out_channels=F2, 
            kernel_size=(1, kernel_S), 
            padding=(0, kernel_S // 2), 
            groups=F2, # Depthwise: groups=channels
            bias=False
        )
        
        # Layer 5: Separable Conv (Pointwise Part)
        # Input shape: (N, F2, 1, time_dim_after_depth)
        # Output shape: (N, F2, 1, time_dim_after_depth)
        self.separable_conv_point = nn.Conv2d(
            in_channels=F2, 
            out_channels=F2, 
            kernel_size=(1, 1), 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        
        # Layer 6: Average Pooling 2
        # Pools along the time dimension
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, P2), stride=(1, P2))
        self.drop2 = nn.Dropout(dropout_rate)
        
        # --- Classification Head ---
        
        # Calculate the size of the features after the final pooling layer
        # Time dimension is reduced by P1 then P2
        final_time_dim = n_times // P1 // P2
        
        # Calculate the number of features before the fully connected layer
        self.n_features_before_fc = F2 * final_time_dim
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_features_before_fc, n_classes),
            # nn.LogSoftmax(dim=1) # LogSoftmax is often used if CrossEntropyLoss is used later
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, T). N=Batch, C=Channels, T=Time.

        Returns:
            torch.Tensor: Output logits of shape (N, n_classes).
        """
        # Data reshaping: (N, C, T) -> (N, 1, C, T) to match 2D Conv input format
        x = x.unsqueeze(1) 
        
        # --- Block 1 ---
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.drop1(x)
        
        # --- Block 2 ---
        x = self.separable_conv_depth(x)
        x = self.separable_conv_point(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.drop2(x)
        
        # --- Classification Head ---
        logits = self.classifier(x)
        
        return logits

# --- Main Execution for Testing ---

if __name__ == '__main__':
    # Based on configuration in preprocess.py: 3 channels, 4 seconds at 100Hz
    N_CHANNELS = 3
    N_TIMES = 400 
    N_CLASSES = 2
    BATCH_SIZE = 16
    
    print("--- Testing EEGNet Model ---")
    
    # Instantiate the model
    model = EEGNet(
        n_channels=N_CHANNELS, 
        n_times=N_TIMES, 
        n_classes=N_CLASSES
    )
    
    # Create a dummy batch of data: (Batch, Channels, Time)
    dummy_input = torch.randn(BATCH_SIZE, N_CHANNELS, N_TIMES)
    print(f"Input shape: {tuple(dummy_input.shape)}")
    
    # Perform the forward pass
    try:
        output = model(dummy_input)
        
        print(f"Output shape (logits): {tuple(output.shape)}")
        
        assert output.shape == (BATCH_SIZE, N_CLASSES)
        print("Model test successful: Output shape matches expected (Batch, Classes).")
        
        # Calculate total parameters (optional)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

    except Exception as e:
        print(f"Model forward pass failed: {e}")

    # Optional visualization of the network structure:
    # print("\nModel Structure:")
    # print(model)

# The EEGNet architecture is key to the project's success due to its small footprint and good performance on MI tasks.
#
