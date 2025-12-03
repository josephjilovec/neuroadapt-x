import mne
import numpy as np
import os
from moabb.datasets import PhysioNetMI
from moabb.utils import set_download_dir
from typing import List, Tuple, Dict, Union

# --- Configuration ---
# Set the MOABB download directory to the local 'data/raw' folder
RAW_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'raw'
))
# Ensure MOABB uses our designated raw data path
set_download_dir(RAW_DATA_PATH) 

# Initialize the PhysioNet MI dataset object
# This dataset contains 109 subjects performing motor imagery (MI) tasks.
# We focus on the core MI tasks: Hands vs. Feet (T1 vs. T2 events)
MI_DATASET = PhysioNetMI()
# Note: PhysioNetMI typically uses 64 channels, but only C3, Cz, C4 are usually critical for MI.

# --- Core Functions ---

def fetch_and_load_raw_data(
    subject_ids: Union[List[int], str] = 'all'
) -> Dict[int, Dict[str, mne.io.Raw]]:
    """
    Downloads and loads the raw EEG data for specified subjects from the PhysioNet MI dataset.

    Data is loaded into MNE-Python Raw objects.

    Args:
        subject_ids: A list of subject numbers (1 to 109) to load, or 'all'.

    Returns:
        A dictionary mapping subject ID to a nested dictionary of session names
        and their corresponding MNE Raw objects.
    """
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
        print("Ensure you have a stable internet connection for the initial download.")
        return {}


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
