#!/usr/bin/env python
"""
Main pipeline script for NeuroAdapt-X

This script runs the complete pipeline:
1. Generate realistic training data
2. Train baseline EEGNet model
3. Test on stressed data
4. Perform domain adaptation
5. Evaluate adapted model

Usage:
    python scripts/run_pipeline.py [--config CONFIG_PATH]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from config import load_config, get_config
from src.utils.logger import setup_logger, get_logger
from src.data.realistic_simulator import generate_realistic_mi_dataset
from src.data.add_stressors import inject_stressors_into_epochs
from src.models.eegnet import EEGNet
from src.models.adaptive import AdaptiveEEGNet, CORALLoss
from src.models.train import train_source_domain, train_adaptation_domain
from src.utils.metrics import calculate_accuracy, OnlineAccuracyTracker


def generate_data(config, logger):
    """Generate training and test data"""
    logger.info("Generating realistic training data...")
    
    # Generate source (clean) data
    X_source, y_source = generate_realistic_mi_dataset(
        n_epochs=config.training.batch_size * 100,  # ~6400 epochs
        n_channels=config.data.n_channels,
        sfreq=config.data.sfreq,
        n_times=config.data.n_times,
        n_classes=config.data.n_classes,
        noise_level=config.data.noise_level,
        erd_strength=config.data.erd_strength,
        random_seed=42
    )
    
    logger.info(f"Generated {len(X_source)} source epochs")
    
    # Generate target (stressed) data
    X_target, y_target = generate_realistic_mi_dataset(
        n_epochs=config.training.batch_size * 20,  # ~1280 epochs
        n_channels=config.data.n_channels,
        sfreq=config.data.sfreq,
        n_times=config.data.n_times,
        n_classes=config.data.n_classes,
        noise_level=config.data.noise_level * 2.0,  # Higher noise
        erd_strength=config.data.erd_strength * 0.5,  # Weaker ERD (stressed)
        random_seed=43
    )
    
    logger.info(f"Generated {len(X_target)} target epochs")
    
    return (X_source, y_source), (X_target, y_target)


def create_dataloaders(X, y, batch_size, shuffle=True):
    """Create PyTorch DataLoaders"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model, dataloader, device, logger):
    """Evaluate model on a dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            if isinstance(model, AdaptiveEEGNet):
                logits, _ = model(X)
            else:
                logits = model(X)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = calculate_accuracy(np.array(all_labels), np.array(all_preds))
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Run NeuroAdapt-X pipeline")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and load existing models'
    )
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    logger = setup_logger(
        name="neuroadapt_pipeline",
        log_dir=config.paths.logs_dir,
        level=20  # INFO
    )
    
    device = config.get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config.model.__dict__}")
    
    # Generate data
    (X_source, y_source), (X_target, y_target) = generate_data(config, logger)
    
    # Create data loaders
    source_loader = create_dataloaders(
        X_source, y_source, config.training.batch_size, shuffle=True
    )
    target_loader = create_dataloaders(
        X_target, y_target, config.training.batch_size, shuffle=False
    )
    
    # Initialize model
    base_model = EEGNet(
        n_channels=config.model.n_channels,
        n_times=config.model.n_times,
        n_classes=config.model.n_classes,
        F1=config.model.F1,
        D=config.model.D,
        F2=config.model.F2,
        kernel_T=config.model.kernel_T,
        P1=config.model.P1,
        P2=config.model.P2,
        dropout_rate=config.model.dropout_rate
    )
    
    model_path = config.paths.checkpoints_dir / "eegnet_baseline.pth"
    
    # Train baseline model
    if not args.skip_training:
        logger.info("Training baseline EEGNet model...")
        train_source_domain(
            base_model,
            source_loader,
            config.training.epochs_source,
            device,
            str(model_path)
        )
    else:
        logger.info("Loading existing baseline model...")
        try:
            base_model.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            logger.error(f"Model not found at {model_path}. Run without --skip-training")
            return
    
    # Evaluate baseline on source data
    logger.info("Evaluating baseline model on source data...")
    source_accuracy = evaluate_model(base_model, source_loader, device, logger)
    logger.info(f"Baseline source accuracy: {source_accuracy:.4f}")
    
    # Evaluate baseline on target data (should be lower)
    logger.info("Evaluating baseline model on target (stressed) data...")
    target_accuracy = evaluate_model(base_model, target_loader, device, logger)
    logger.info(f"Baseline target accuracy: {target_accuracy:.4f}")
    logger.info(f"Domain shift: {source_accuracy - target_accuracy:.4f}")
    
    # Create adaptive model
    adaptive_model = AdaptiveEEGNet(base_model)
    adaptive_model_path = config.paths.checkpoints_dir / "adaptive_eegnet.pth"
    
    # Train adaptation
    if not args.skip_training:
        logger.info("Training adaptive model...")
        train_adaptation_domain(
            adaptive_model,
            source_loader,
            target_loader,
            config.training.epochs_adapt,
            device,
            config.training.coral_lambda
        )
        torch.save(adaptive_model.state_dict(), adaptive_model_path)
    else:
        logger.info("Loading existing adaptive model...")
        try:
            adaptive_model.load_state_dict(
                torch.load(adaptive_model_path, map_location=device)
            )
        except FileNotFoundError:
            logger.error(f"Adaptive model not found at {adaptive_model_path}")
            return
    
    # Evaluate adapted model on target data
    logger.info("Evaluating adapted model on target data...")
    adapted_accuracy = evaluate_model(adaptive_model, target_loader, device, logger)
    logger.info(f"Adapted target accuracy: {adapted_accuracy:.4f}")
    logger.info(f"Improvement: {adapted_accuracy - target_accuracy:.4f}")
    
    logger.info("Pipeline complete!")


if __name__ == '__main__':
    main()

