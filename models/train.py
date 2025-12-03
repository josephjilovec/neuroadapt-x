import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# Import the models and loss functions
try:
    from .eegnet import EEGNet
    from .adaptive import AdaptiveEEGNet, CORALLoss
except ImportError:
    # Fallback for running the script directly in development
    print("Warning: Running train.py standalone. Ensure eegnet.py and adaptive.py are in path.")
    # Placeholders/Mocks for Models (if running outside the package structure)
    class MockEEGNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.classifier = nn.Linear(100, 2)
        def forward(self, x):
            return self.classifier(torch.randn(x.size(0), 100))
    EEGNet = MockEEGNet
    AdaptiveEEGNet = MockEEGNet
    class MockCORALLoss(nn.Module):
        def forward(self, *args, **kwargs): return torch.tensor(0.0)
    CORALLoss = MockCORALLoss

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CHANNELS = 3    # C3, Cz, C4 (or similar subset)
N_TIMES = 400     # 4 seconds at 100Hz
N_CLASSES = 2     # Left vs Right MI
BATCH_SIZE = 64
EPOCHS_SOURCE = 50
EPOCHS_ADAPT = 5
CORAL_LAMBDA = 0.5 # Weight for the CORAL loss

# --- Placeholder Data Loading Functions ---
def create_mock_dataloader(num_samples, channels, times, classes, is_stressed=False):
    """
    Creates mock data loaders for demonstration purposes.
    In a real scenario, this would load data using src/data/load_datasets.py.
    """
    X = torch.randn(num_samples, channels, times, dtype=torch.float32)
    y = torch.randint(0, classes, (num_samples,), dtype=torch.long)
    
    if is_stressed:
        # Simulate domain shift by adding mean/variance shift to the data
        X = X + 0.5 * torch.randn_like(X) + 0.1 # Simulate baseline drift/noise
        
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Functions ---

def train_source_domain(model: EEGNet, dataloader: DataLoader, epochs: int, device: torch.device, model_path: str):
    """
    Trains the baseline EEGNet model using standard supervised learning.
    """
    print("\n--- Phase 1: Training Source Domain Baseline (EEGNet) ---")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # EEGNet returns logits only
            logits = model(X) 
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
        
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {time.time() - start_time:.2f}s")
    
    print(f"\nSource training complete. Best model saved to {model_path}")
    
def set_model_trainable(model: AdaptiveEEGNet, classifier_only: bool = False):
    """
    Controls which parameters are trainable during adaptation.
    Freezes feature extractor layers but allows AdaBN and Classifier to train.
    """
    if classifier_only:
        print("-> Freezing all layers except the Classifier head.")
    else:
        print("-> Freezing Feature Extractor (Convs), enabling AdaBN and Classifier.")

    for name, param in model.named_parameters():
        # Default: Freeze everything
        param.requires_grad = False
        
        # Unfreeze the AdaBN layers (weight and bias)
        if 'bn' in name and any(isinstance(m, CORALLoss) for m in model.modules()): # Check for AdaBN within the wrapper
            if 'weight' in name or 'bias' in name:
                param.requires_grad = True

        # Unfreeze the final classification head
        if 'classifier' in name:
            param.requires_grad = True
    
    # Verify the count of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Total trainable parameters for adaptation: {trainable_params:,}")


def train_adaptation_domain(
    adaptive_model: AdaptiveEEGNet, 
    source_dataloader: DataLoader, 
    target_dataloader: DataLoader, 
    epochs: int, 
    device: torch.device,
    adaptation_lambda: float,
):
    """
    Performs unsupervised domain adaptation using CORAL loss + AdaBN.
    
    The AdaptiveEEGNet is set to adaptation mode (AdaBN uses batch stats 
    for target data). We optimize the CORAL loss between source and target 
    features, plus the Cross-Entropy loss (if labels are available, but 
    here we focus on the unsupervised alignment).
    
    NOTE: For pure unsupervised DA, we only rely on the CORAL loss 
    to align features, and AdaBN to normalize statistics.
    """
    print(f"\n--- Phase 2: Unsupervised Domain Adaptation (AdaBN + CORAL, Lambda={adaptation_lambda}) ---")
    adaptive_model.to(device)
    
    # Set trainable parameters (AdaBN and Classifier)
    set_model_trainable(adaptive_model, classifier_only=False)
    
    # Use Cross-Entropy Loss on labeled target data if available (Semi-Supervised). 
    # For this demo, we assume we have *some* target labels to keep the classifier aligned.
    criterion_ce = nn.CrossEntropyLoss()
    criterion_coral = CORALLoss()
    
    # Optimizer only updates parameters where requires_grad=True
    optimizer = optim.Adam(adaptive_model.parameters(), lr=1e-4)
    
    # We must iterate over both source and target data simultaneously for CORAL
    source_iter = iter(source_dataloader)

    for epoch in range(1, epochs + 1):
        adaptive_model.train()
        adaptive_model.set_adaptation_mode(True) # Crucial: Tell AdaBN to use batch stats
        
        total_loss = 0.0
        start_time = time.time()

        for step, (X_T, y_T) in enumerate(target_dataloader):
            try:
                X_S, y_S = next(source_iter)
            except StopIteration:
                # Reset source iterator when it runs out
                source_iter = iter(source_dataloader)
                X_S, y_S = next(source_iter)
            
            X_S, y_S = X_S.to(device), y_S.to(device)
            X_T, y_T = X_T.to(device), y_T.to(device)
            
            optimizer.zero_grad()
            
            # 1. Forward pass on Source (for features C_S)
            _, features_S = adaptive_model(X_S)
            
            # 2. Forward pass on Target (for features C_T and Classification L_CE)
            logits_T, features_T = adaptive_model(X_T)
            
            # 3. Calculate Losses
            loss_ce = criterion_ce(logits_T, y_T) # Use target labels if available (semi-supervised demo)
            loss_coral = criterion_coral(features_S, features_T)
            
            # Total Loss combines classification and feature alignment
            total_loss_batch = loss_ce + adaptation_lambda * loss_coral
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            
        avg_loss = total_loss / len(target_dataloader)
        
        print(f"Adapt Epoch {epoch}/{epochs} | Total Loss: {avg_loss:.4f} (CE: {loss_ce.item():.4f}, CORAL: {loss_coral.item():.4f}) | Time: {time.time() - start_time:.2f}s")
        
    print("\nAdaptation training complete. Model is ready for stress-resilient inference.")
    
    # Set model back to evaluation mode for inference/demo
    adaptive_model.eval()
    adaptive_model.set_adaptation_mode(False)
    return adaptive_model

# --- Main Execution ---

def main():
    # --- 1. Load Data ---
    print(f"Using device: {DEVICE}")
    print("Creating mock datasets (Source: Clean, Target: Stressed)...")
    
    # Note: Target data is typically much smaller for adaptation, 
    # but here we use a similar size for the mock training loop.
    SOURCE_SAMPLES = 8000
    TARGET_SAMPLES = 2000 
    
    source_loader = create_mock_dataloader(SOURCE_SAMPLES, N_CHANNELS, N_TIMES, N_CLASSES, is_stressed=False)
    target_loader = create_mock_dataloader(TARGET_SAMPLES, N_CHANNELS, N_TIMES, N_CLASSES, is_stressed=True)
    
    # --- 2. Train Base EEGNet ---
    
    base_model = EEGNet(N_CHANNELS, N_TIMES, N_CLASSES)
    SOURCE_MODEL_PATH = "eegnet_source_weights.pth"
    
    # train_source_domain(base_model, source_loader, EPOCHS_SOURCE, DEVICE, SOURCE_MODEL_PATH)

    # --- SIMULATION: Load the "trained" weights ---
    print("\n[SIMULATION] Skipping full source training. Loading mock pre-trained model state.")
    try:
        # If we successfully save the model once, we can uncomment this
        # base_model.load_state_dict(torch.load(SOURCE_MODEL_PATH))
        pass 
    except:
        # If not, we just use the randomly initialized model
        pass

    # --- 3. Initialize Adaptive Model ---
    
    adaptive_model = AdaptiveEEGNet(base_model)
    
    # --- 4. Adapt Model to Target Domain (Simulated Stress) ---
    
    final_adaptive_model = train_adaptation_domain(
        adaptive_model=adaptive_model,
        source_dataloader=source_loader,
        target_dataloader=target_loader,
        epochs=EPOCHS_ADAPT,
        device=DEVICE,
        adaptation_lambda=CORAL_LAMBDA
    )
    
    # --- 5. Save Final Adaptive Model ---
    
    ADAPTIVE_MODEL_PATH = "adaptive_eegnet_final_weights.pth"
    torch.save(final_adaptive_model.state_dict(), ADAPTIVE_MODEL_PATH)
    print(f"\nFinal Adaptive model saved to {ADAPTIVE_MODEL_PATH}")


if __name__ == '__main__':
    main()
