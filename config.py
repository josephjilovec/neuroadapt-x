"""
Configuration management for NeuroAdapt-X

Centralized configuration for all components including:
- Model hyperparameters
- Data generation parameters
- Training settings
- Adaptation parameters
- Paths and directories
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class ModelConfig:
    """EEGNet model configuration"""
    n_channels: int = 3
    n_times: int = 400
    n_classes: int = 2
    F1: int = 8  # Number of temporal filters
    D: int = 2  # Depth multiplier
    F2: int = 16  # Number of separable conv filters (F1 * D)
    kernel_T: int = 64  # Temporal kernel size
    P1: int = 8  # Pooling factor for Block 1
    P2: int = 4  # Pooling factor for Block 2
    dropout_rate: float = 0.25


@dataclass
class DataConfig:
    """Data generation and preprocessing configuration"""
    sfreq: float = 100.0  # Sampling frequency (Hz)
    n_channels: int = 3
    n_times: int = 400  # 4 seconds at 100 Hz
    n_classes: int = 2
    ch_names: list = field(default_factory=lambda: ['C3', 'Cz', 'C4'])
    
    # Realistic simulation parameters
    noise_level: float = 2.0  # Noise amplitude in microvolts
    erd_strength: float = 0.4  # ERD strength (0-1)
    
    # Preprocessing parameters
    low_cutoff_hz: float = 8.0
    high_cutoff_hz: float = 30.0
    t_min: float = 0.0
    t_max: float = 4.0
    baseline_period: tuple = (None, 0)
    
    # Stress simulation parameters
    shift_amplitude_uv: float = 50.0
    shift_freq_range_hz: tuple = (0.1, 0.5)
    emg_amplitude_uv: float = 15.0
    emg_freq_range_hz: tuple = (50.0, 150.0)
    emg_probability: float = 0.3


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 64
    epochs_source: int = 50
    epochs_adapt: int = 5
    learning_rate: float = 1e-3
    learning_rate_adapt: float = 1e-4
    weight_decay: float = 1e-5
    coral_lambda: float = 0.5  # Weight for CORAL loss
    device: str = "auto"  # "auto", "cuda", or "cpu"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Checkpointing
    save_best: bool = True
    save_frequency: int = 5  # Save every N epochs


@dataclass
class AdaptationConfig:
    """Online adaptation configuration"""
    confidence_threshold: float = 0.8
    adaptation_buffer_size: int = 20
    adaptation_batches: int = 4
    coral_lambda: float = 0.5
    learning_rate_online: float = 1e-5
    
    # Real-time processing
    epoch_duration: float = 4.0  # seconds
    block_duration: float = 0.1  # seconds
    decoding_rate: float = 10.0  # Hz


@dataclass
class PathConfig:
    """Paths and directories configuration"""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            adaptation=AdaptationConfig(**config_dict.get('adaptation', {})),
            paths=PathConfig(**config_dict.get('paths', {}))
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'adaptation': self.adaptation.__dict__,
            'paths': {k: str(v) for k, v in self.paths.__dict__.items() if isinstance(v, Path)}
        }
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_device(self):
        """Get PyTorch device based on configuration"""
        import torch
        
        if self.training.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.training.device)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config):
    """Set the global configuration instance"""
    global _config
    _config = config


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path: Path to JSON config file (optional)
        
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        config = Config.from_json(config_path)
    else:
        config = Config()
    
    # Override with environment variables if present
    if os.getenv('NEUROADAPT_N_CHANNELS'):
        config.model.n_channels = int(os.getenv('NEUROADAPT_N_CHANNELS'))
        config.data.n_channels = int(os.getenv('NEUROADAPT_N_CHANNELS'))
    
    if os.getenv('NEUROADAPT_N_TIMES'):
        config.model.n_times = int(os.getenv('NEUROADAPT_N_TIMES'))
        config.data.n_times = int(os.getenv('NEUROADAPT_N_TIMES'))
    
    if os.getenv('NEUROADAPT_DEVICE'):
        config.training.device = os.getenv('NEUROADAPT_DEVICE')
    
    set_config(config)
    return config


# Default configuration
DEFAULT_CONFIG = Config()

