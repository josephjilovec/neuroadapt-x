"""
Tests for configuration management
"""

import pytest
import os
import json
from pathlib import Path
from config import (
    Config,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    AdaptationConfig,
    PathConfig,
    get_config,
    load_config
)


class TestConfig:
    """Test configuration classes"""
    
    def test_model_config_defaults(self):
        """Test ModelConfig defaults"""
        config = ModelConfig()
        
        assert config.n_channels == 3
        assert config.n_times == 400
        assert config.n_classes == 2
        assert config.F1 == 8
        assert config.D == 2
        assert config.F2 == 16
    
    def test_data_config_defaults(self):
        """Test DataConfig defaults"""
        config = DataConfig()
        
        assert config.sfreq == 100.0
        assert config.n_channels == 3
        assert config.noise_level == 2.0
    
    def test_training_config_defaults(self):
        """Test TrainingConfig defaults"""
        config = TrainingConfig()
        
        assert config.batch_size == 64
        assert config.epochs_source == 50
        assert config.learning_rate == 1e-3
    
    def test_path_config_creation(self):
        """Test PathConfig directory creation"""
        config = PathConfig()
        
        assert config.data_dir.exists()
        assert config.models_dir.exists()
        assert config.logs_dir.exists()
    
    def test_config_from_dict(self):
        """Test Config creation from dictionary"""
        config_dict = {
            'model': {'n_channels': 5, 'n_times': 500},
            'data': {'sfreq': 250.0},
            'training': {'batch_size': 32}
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.model.n_channels == 5
        assert config.model.n_times == 500
        assert config.data.sfreq == 250.0
        assert config.training.batch_size == 32
    
    def test_config_to_dict(self):
        """Test Config conversion to dictionary"""
        config = Config()
        config_dict = config.to_dict()
        
        assert 'model' in config_dict
        assert 'data' in config_dict
        assert 'training' in config_dict
        assert config_dict['model']['n_channels'] == 3
    
    def test_config_json_roundtrip(self, tmp_path):
        """Test Config JSON save/load"""
        config = Config()
        config.model.n_channels = 5
        
        json_path = tmp_path / "test_config.json"
        config.to_json(str(json_path))
        
        loaded_config = Config.from_json(str(json_path))
        
        assert loaded_config.model.n_channels == 5
    
    def test_get_config_singleton(self):
        """Test get_config returns singleton"""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_get_device(self):
        """Test device selection"""
        config = Config()
        device = config.get_device()
        
        assert device is not None
        assert str(device) in ['cpu', 'cuda']

