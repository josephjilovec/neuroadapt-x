"""
Model definitions for NeuroAdapt-X
"""

from .eegnet import EEGNet
from .adaptive import AdaptiveEEGNet, AdaBN2d, CORALLoss
from .train import train_source_domain, train_adaptation_domain

__all__ = [
    'EEGNet',
    'AdaptiveEEGNet',
    'AdaBN2d',
    'CORALLoss',
    'train_source_domain',
    'train_adaptation_domain',
]

