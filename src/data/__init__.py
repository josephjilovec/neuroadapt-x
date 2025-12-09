"""
Data loading and preprocessing modules for NeuroAdapt-X
"""

from .load_datasets import fetch_and_load_raw_data, generate_simulated_raw_data
from .preprocess import preprocess_raw_data, process_all_subjects
from .add_stressors import inject_stressors_into_epochs
from .realistic_simulator import (
    RealisticEEGSimulator,
    generate_realistic_mi_dataset
)

__all__ = [
    'fetch_and_load_raw_data',
    'generate_simulated_raw_data',
    'preprocess_raw_data',
    'process_all_subjects',
    'inject_stressors_into_epochs',
    'RealisticEEGSimulator',
    'generate_realistic_mi_dataset',
]

