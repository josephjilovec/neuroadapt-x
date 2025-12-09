"""
Tests for realistic EEG simulator
"""

import pytest
import numpy as np
from src.data.realistic_simulator import (
    RealisticEEGSimulator,
    generate_realistic_mi_dataset
)


class TestRealisticEEGSimulator:
    """Test cases for RealisticEEGSimulator"""
    
    def test_simulator_initialization(self):
        """Test simulator initialization"""
        simulator = RealisticEEGSimulator(
            n_channels=3,
            sfreq=100.0,
            n_times=400,
            random_seed=42
        )
        
        assert simulator.n_channels == 3
        assert simulator.sfreq == 100.0
        assert simulator.n_times == 400
        assert len(simulator.ch_names) == 3
    
    def test_generate_epoch_shape(self):
        """Test epoch generation shape"""
        simulator = RealisticEEGSimulator(
            n_channels=3,
            sfreq=100.0,
            n_times=400,
            random_seed=42
        )
        
        epoch = simulator.generate_epoch(task_type=0)
        
        assert epoch.shape == (3, 400)
        assert epoch.dtype == np.float64 or epoch.dtype == np.float32
    
    def test_generate_epoch_values(self):
        """Test epoch generation produces reasonable values"""
        simulator = RealisticEEGSimulator(
            n_channels=3,
            sfreq=100.0,
            n_times=400,
            random_seed=42
        )
        
        epoch = simulator.generate_epoch(task_type=0, noise_level=2.0)
        
        # Values should be in reasonable range (microvolts)
        assert np.abs(epoch).max() < 100  # Should not exceed 100 µV
        assert not np.isnan(epoch).any()
        assert not np.isinf(epoch).any()
    
    def test_generate_dataset_balanced(self):
        """Test balanced dataset generation"""
        simulator = RealisticEEGSimulator(
            n_channels=3,
            sfreq=100.0,
            n_times=400,
            random_seed=42
        )
        
        X, y = simulator.generate_dataset(
            n_epochs=100,
            n_classes=2,
            balanced=True
        )
        
        assert X.shape == (100, 3, 400)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 2
        assert np.allclose(np.bincount(y), [50, 50])  # Balanced
    
    def test_generate_dataset_unbalanced(self):
        """Test unbalanced dataset generation"""
        simulator = RealisticEEGSimulator(
            n_channels=3,
            sfreq=100.0,
            n_times=400,
            random_seed=42
        )
        
        X, y = simulator.generate_dataset(
            n_epochs=100,
            n_classes=2,
            balanced=False
        )
        
        assert X.shape == (100, 3, 400)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 2
    
    def test_to_mne_epochs(self):
        """Test conversion to MNE Epochs"""
        simulator = RealisticEEGSimulator(
            n_channels=3,
            sfreq=100.0,
            n_times=400,
            random_seed=42
        )
        
        X, y = simulator.generate_dataset(n_epochs=10, n_classes=2)
        epochs = simulator.to_mne_epochs(X, y)
        
        assert len(epochs) == 10
        assert epochs.info['sfreq'] == 100.0
        assert len(epochs.ch_names) == 3
    
    def test_erd_application(self):
        """Test ERD pattern application"""
        simulator = RealisticEEGSimulator(
            n_channels=3,
            sfreq=100.0,
            n_times=400,
            random_seed=42
        )
        
        # Generate epochs with and without ERD
        epoch_no_erd = simulator.generate_epoch(task_type=0, erd_strength=0.0)
        epoch_with_erd = simulator.generate_epoch(task_type=0, erd_strength=0.4)
        
        # ERD should reduce power in certain channels
        # This is a basic check - in practice, ERD reduces mu/beta power
        assert epoch_no_erd.shape == epoch_with_erd.shape


class TestGenerateRealisticMIDataset:
    """Test convenience function"""
    
    def test_generate_dataset_function(self):
        """Test the convenience function"""
        X, y = generate_realistic_mi_dataset(
            n_epochs=50,
            n_channels=3,
            sfreq=100.0,
            n_times=400,
            random_seed=42
        )
        
        assert X.shape == (50, 3, 400)
        assert y.shape == (50,)
        assert len(np.unique(y)) == 2

