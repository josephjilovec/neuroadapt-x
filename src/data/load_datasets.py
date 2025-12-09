import mne
import numpy as np
import os
from typing import List, Tuple, Dict, Union, Optional

# Optional MOABB import (for real data)
try:
    from moabb.datasets import PhysioNetMI
    from moabb.utils import set_download_dir
    MOABB_AVAILABLE = True
except ImportError:
    MOABB_AVAILABLE = False
    PhysioNetMI = None

# Import realistic simulator
try:
    from .realistic_simulator import RealisticEEGSimulator, generate_realistic_mi_dataset
except ImportError:
    from realistic_simulator import RealisticEEGSimulator, generate_realistic_mi_dataset

# --- Configuration ---
# Set the MOABB download directory to the local 'data/raw' folder
RAW_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'raw'
))

# Initialize the PhysioNet MI dataset object (if MOABB is available)
if MOABB_AVAILABLE:
    set_download_dir(RAW_DATA_PATH)
    MI_DATASET = PhysioNetMI()
else:
    MI_DATASET = None

# --- Core Functions ---

def fetch_and_load_raw_data(
    subject_ids: Union[List[int], str] = 'all',
    use_realistic_simulator: bool = True
) -> Dict[int, Dict[str, mne.io.Raw]]:
    """
    Downloads and loads the raw EEG data for specified subjects from the PhysioNet MI dataset,
    or generates realistic simulated data if MOABB is not available.

    Data is loaded into MNE-Python Raw objects.

    Args:
        subject_ids: A list of subject numbers (1 to 109) to load, or 'all'.
        use_realistic_simulator: If True and MOABB unavailable, use realistic simulator.

    Returns:
        A dictionary mapping subject ID to a nested dictionary of session names
        and their corresponding MNE Raw objects.
    """
    if not MOABB_AVAILABLE or use_realistic_simulator:
        # Use realistic simulator to generate data
        print("Using realistic EEG simulator (MOABB not available or simulator requested)")
        return generate_simulated_raw_data(subject_ids)
    
    if subject_ids == 'all':
        # Default to a reasonable subset or all subjects supported by MOABB/MNE
        subjects_to_load = MI_DATASET.subject_list
    else:
        subjects_to_load = subject_ids
    
    print(f"Starting data fetch for subjects: {subjects_to_load}")
    
    try:
        # The MOABB dataset.get_data method handles downloading, caching, and loading
        # into a dictionary structure: {subject_id: {session_name: raw_data}}
        raw_data_dict = MI_DATASET.get_data(subjects=subjects_to_load)
        print("Data loaded successfully.")
        return raw_data_dict
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        print("Falling back to realistic simulator...")
        return generate_simulated_raw_data(subject_ids)


def generate_simulated_raw_data(
    subject_ids: Union[List[int], str] = 'all',
    n_epochs_per_subject: int = 200
) -> Dict[int, Dict[str, mne.io.Raw]]:
    """
    Generate realistic simulated raw EEG data using the realistic simulator.
    
    Args:
        subject_ids: List of subject IDs or 'all' (default: [1, 2, 3, 4, 5])
        n_epochs_per_subject: Number of epochs to generate per subject
        
    Returns:
        Dictionary mapping subject ID to session data
    """
    if subject_ids == 'all':
        subject_ids = [1, 2, 3, 4, 5]
    elif isinstance(subject_ids, int):
        subject_ids = [subject_ids]
    
    raw_data_dict = {}
    
    simulator = RealisticEEGSimulator(
        n_channels=3,
        sfreq=100.0,
        n_times=400,
        random_seed=42
    )
    
    for subj_id in subject_ids:
        print(f"Generating simulated data for subject {subj_id}...")
        
        # Generate epochs for this subject
        X, y = simulator.generate_dataset(
            n_epochs=n_epochs_per_subject,
            n_classes=2,
            balanced=True,
            subject_ids=[subj_id],
            noise_level=2.0,
            erd_strength=0.4
        )
        
        # Convert to MNE Epochs
        epochs = simulator.to_mne_epochs(X, y)
        
        # Convert epochs to Raw (concatenate all epochs)
        raw = epochs.to_data_frame().T
        raw_array = raw.values
        
        # Create MNE Raw object
        info = mne.create_info(
            ch_names=simulator.ch_names,
            sfreq=simulator.sfreq,
            ch_types='eeg'
        )
        
        raw_mne = mne.io.RawArray(raw_array, info, verbose=False)
        
        # Store in dictionary format matching MOABB structure
        raw_data_dict[subj_id] = {'session_1': raw_mne}
    
    print(f"Generated simulated data for {len(subject_ids)} subjects")
    return raw_data_dict


def load_local_eeg_data(filepath: str) -> mne.io.Raw:
    """
    Loads a single EEG file (e.g., .fif, .edf) from a local path.
    
    This is a fallback for data not managed by MOABB.

    Args:
        filepath: The absolute path to the local EEG data file.

    Returns:
        An MNE Raw object, or None if loading fails.
    """
    if not os.path.exists(filepath):
        print(f"Error: Local file not found at {filepath}")
        return None

    print(f"Loading local file: {os.path.basename(filepath)}")
    try:
        # MNE-Python has readers for various formats (.fif, .edf, .eeg, etc.)
        if filepath.endswith('.fif'):
            raw = mne.io.read_raw_fif(filepath, preload=True, verbose='error')
        elif filepath.endswith(('.edf', '.bdf', '.gdf')):
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='error')
        else:
            print("Unsupported local file format. Only .fif, .edf, .bdf, .gdf supported.")
            return None
        
        return raw
    except Exception as e:
        print(f"Failed to load local EEG file: {e}")
        return None

# --- Main Execution for Testing ---

if __name__ == '__main__':
    # 1. Test fetching data for two subjects (e.g., subject 1 and subject 2)
    # The first time this runs, it will download the data to data/raw.
    print("--- Testing MOABB Dataset Fetch (Subject 1 & 2) ---")
    
    # We load only the first two subjects to keep the download small for a quick test
    test_subjects = [1, 2] 
    
    raw_data = fetch_and_load_raw_data(test_subjects)
    
    if raw_data:
        print("\n--- Summary of Loaded Data ---")
        for subj_id, sessions in raw_data.items():
            print(f"Subject {subj_id} loaded with {len(sessions)} sessions.")
            for session_name, raw_obj in sessions.items():
                print(f"  - Session '{session_name}': {raw_obj.info['sfreq']} Hz, {raw_obj.n_channels} channels.")
                # The data itself is a numpy array (channels x time)
                # raw_obj.get_data() 
                
        # Example of saving the data to the 'processed' folder (optional - usually done in preprocess.py)
        # Note: We skip the full preprocessing pipeline here, just showing the raw access
        
    else:
        print("\nFailed to load any data.")
    
    print("\n--- All done with data loading test. ---")
