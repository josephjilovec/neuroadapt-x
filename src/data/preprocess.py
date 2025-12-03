import mne
import numpy as np
import os
from mne.epochs import Epochs
from typing import Dict, Union, List, Tuple
from collections import defaultdict

# Importing the load function from the sibling file for testing purposes
try:
    from .load_datasets import fetch_and_load_raw_data
except ImportError:
    # Fallback for direct script execution
    def fetch_and_load_raw_data(*args, **kwargs):
        print("Warning: load_datasets.py not imported. Cannot run full test block.")
        return None

# --- Configuration for Preprocessing ---

# Frequency band for Motor Imagery (MI) analysis: Mu (Alpha) and Beta Rhythms
LOW_CUTOFF_HZ = 8.0
HIGH_CUTOFF_HZ = 30.0

# Time window for epochs (MI typically starts 0-4 seconds after cue)
T_MIN = 0.0     # Start time of the epoch relative to the event (seconds)
T_MAX = 4.0     # End time of the epoch relative to the event (seconds)
BASELINE_PERIOD = (None, 0) # Baseline correction: from start of epoch to event (in this case, 0 seconds, but we often use a pre-stimulus window)

# Channels relevant for Motor Imagery (C3, Cz, C4 are central motor cortex)
MI_CHANNELS = ['C3', 'Cz', 'C4'] 

# Events mapping for PhysioNet MI dataset (T1=Left Hand, T2=Right Hand/Feet)
# We map these to numerical IDs (1 and 2) and their corresponding labels
EVENT_IDS = {
    'T1': 1, # Left Hand MI
    'T2': 2, # Right Hand or Feet MI (PhysioNet uses T2 for both)
}

# --- Core Preprocessing Functions ---

def preprocess_raw_data(raw: mne.io.Raw) -> Union[mne.Epochs, None]:
    """
    Applies the full preprocessing pipeline to a single MNE Raw object.
    
    Steps include:
    1. Setting standard 10-20 montage.
    2. Filtering to the MI band (8-30 Hz).
    3. Selecting relevant motor channels (C3, Cz, C4).
    4. Epoching based on T1/T2 events.
    5. Artifact rejection via simple peak-to-peak amplitude threshold.
    
    Args:
        raw: The raw EEG data (MNE Raw object).
        
    Returns:
        The processed MNE Epochs object, or None if processing fails.
    """
    try:
        # 1. Setting Montage and Channel Types (Important for visualization and spatial filtering)
        # Using a standard 10-20 montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # 2. Filtering
        # Apply a sharp bandpass filter to focus on MI-relevant frequencies (Mu and Beta)
        raw_filtered = raw.copy().filter(
            l_freq=LOW_CUTOFF_HZ, 
            h_freq=HIGH_CUTOFF_HZ, 
            picks='eeg', 
            n_jobs=1, 
            verbose='error'
        )
        print(f"  - Filtered data from {LOW_CUTOFF_HZ} to {HIGH_CUTOFF_HZ} Hz.")

        # 3. Channel Selection
        # Drop all channels not in our motor imagery set
        picks_eeg = mne.pick_channels(raw_filtered.ch_names, MI_CHANNELS, ordered=True)
        raw_selected = raw_filtered.copy().pick(picks_eeg)
        print(f"  - Selected channels: {raw_selected.ch_names}")
        
        # 4. Epoching
        # Find events (triggers) in the data
        events, event_id = mne.events_from_annotations(raw_selected)
        
        # Filter events to include only T1 and T2, matching our EVENT_IDS map
        epochs = mne.Epochs(
            raw_selected, 
            events, 
            event_id=EVENT_IDS, 
            tmin=T_MIN, 
            tmax=T_MAX,
            baseline=BASELINE_PERIOD, # Apply baseline correction
            preload=True,             # Load all data into memory
            verbose='error'
        )
        print(f"  - Epoching complete. Found {len(epochs)} epochs.")
        
        if len(epochs) == 0:
            print("Warning: No MI epochs found after selection.")
            return None
        
        # 5. Simple Artifact Rejection (Peak-to-Peak Amplitude Threshold)
        # Drop epochs where the peak-to-peak amplitude exceeds a threshold (e.g., 100 µV)
        # Note: A more robust approach would use ICA or dedicated artifact rejection tools
        # We use a simple threshold here for robustness
        # This threshold is highly dataset-dependent. 100e-6 V (100 µV) is a common starting point.
        reject_criteria = dict(eeg=100e-6) # 100 microvolts peak-to-peak
        
        n_before = len(epochs)
        epochs.drop_bad(reject=reject_criteria)
        n_after = len(epochs)
        
        print(f"  - Rejected {n_before - n_after} epochs due to amplitude threshold.")
        
        return epochs.copy()

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def process_all_subjects(raw_data_dict: Dict[int, Dict[str, mne.io.Raw]]) -> Dict[int, Epochs]:
    """
    Iterates through all subjects and sessions, processes the raw data, and aggregates epochs.
    
    Args:
        raw_data_dict: Dictionary from load_datasets.py: {subject_id: {session_name: raw_data}}
        
    Returns:
        A dictionary mapping subject ID to the concatenated MNE Epochs object for that subject.
    """
    processed_epochs_dict = {}
    
    for subj_id, sessions in raw_data_dict.items():
        print(f"\n--- Processing Subject {subj_id} ---")
        
        all_epochs_for_subject = []
        
        for session_name, raw_obj in sessions.items():
            print(f"  > Session: {session_name}")
            
            # 1. Ensure channels are EEG (MOABB handles this well, but good practice)
            raw_obj.set_channel_types({ch: 'eeg' for ch in raw_obj.ch_names if ch.startswith('E')})
            
            # 2. Run the full preprocessing pipeline
            epochs = preprocess_raw_data(raw_obj)
            
            if epochs is not None and len(epochs) > 0:
                all_epochs_for_subject.append(epochs)
            
        # Concatenate all sessions for the subject into a single Epochs object
        if all_epochs_for_subject:
            # Note: We must adjust event codes before concatenation if MOABB didn't unify them
            # For PhysioNetMI, the event IDs are consistent across sessions (T1/T2)
            
            # Concatenate the sessions
            combined_epochs = mne.concatenate_epochs(all_epochs_for_subject)
            print(f"\nSubject {subj_id} successfully processed.")
            print(f"Total valid epochs: {len(combined_epochs)} ({len(EVENT_IDS)} classes)")
            processed_epochs_dict[subj_id] = combined_epochs
        else:
            print(f"\nSubject {subj_id} failed to produce any valid epochs.")
            
    return processed_epochs_dict

# --- Main Execution for Testing ---

if __name__ == '__main__':
    print("--- Running Preprocessing Test (Requires Data Download) ---")
    
    # Load raw data for a small subset of subjects
    # Note: If running this directly, ensure you have the necessary dependencies (MNE, MOABB)
    # This calls the function defined in load_datasets.py
    test_subjects = [1, 2] 
    raw_data = fetch_and_load_raw_data(test_subjects)
    
    if raw_data:
        # Run the processing pipeline
        processed_data = process_all_subjects(raw_data)
        
        print("\n\n--- Final Processed Data Summary ---")
        if processed_data:
            for subj_id, epochs_obj in processed_data.items():
                print(f"Subject {subj_id}: {len(epochs_obj)} epochs ready for modeling.")
                # The data shape should be (n_epochs, n_channels, n_times)
                print(f"Data shape: {epochs_obj.get_data(copy=True).shape}")
                
                # Simple visualization of one epoch
                print("Showing a quick plot of one epoch from Subject 1...")
                if 1 in processed_data and len(processed_data[1]) > 0:
                    try:
                        # Plot the first epoch (optional, requires matplotlib)
                        processed_data[1][0].plot(
                            events=epochs_obj.events, 
                            scalings=dict(eeg=100e-6), 
                            block=False, 
                            show=True
                        )
                        print("Close the plot window to continue.")
                        # input("Press Enter to finish test...") # Block until user closes
                    except Exception as e:
                        print(f"Could not display plot. Ensure matplotlib is installed. Error: {e}")
                
        else:
            print("Processing failed for all test subjects.")
    
    print("\n--- Preprocessing test finished. ---")
