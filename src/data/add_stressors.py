import numpy as np
import mne
from mne.epochs import Epochs
from typing import Union, Tuple

# --- Configuration for Stress Simulation ---

# 1. Baseline Shift/Motion Artifact Parameters (Low Frequency Drift)
SHIFT_AMPLITUDE_UV = 50.0  # Max amplitude of the low-frequency shift in microvolts (µV)
SHIFT_FREQ_RANGE_HZ = (0.1, 0.5) # The frequency range of the simulated drift

# 2. EMG Noise Burst Parameters (High Frequency Artifact)
EMG_AMPLITUDE_UV = 15.0    # Max amplitude of the EMG burst in microvolts (µV)
EMG_FREQ_RANGE_HZ = (50.0, 150.0) # EMG frequency band
EMG_PROBABILITY = 0.3      # Probability that an epoch will contain an EMG burst

# Conversion factor: V to uV
V_TO_UV = 1e6 
UV_TO_V = 1e-6

# --- Noise Generation Helpers ---

def generate_baseline_shift(n_times: int, sfreq: float) -> np.ndarray:
    """
    Generates a low-frequency sinusoidal drift (simulating slow motion/sweat artifacts).

    Args:
        n_times: The number of time points (samples) in the epoch.
        sfreq: The sampling frequency (Hz).

    Returns:
        A 1D array of shift values (in Volts) with shape (n_times,).
    """
    t = np.arange(n_times) / sfreq
    
    # Randomly select a frequency within the shift range
    freq = np.random.uniform(*SHIFT_FREQ_RANGE_HZ)
    # Randomly select an amplitude up to the max configured amplitude
    amp_uv = np.random.uniform(10.0, SHIFT_AMPLITUDE_UV) 
    
    # Generate a slow sine wave
    shift_uv = amp_uv * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))
    
    return shift_uv * UV_TO_V

def generate_emg_burst(n_times: int, sfreq: float) -> np.ndarray:
    """
    Generates a high-frequency, band-limited noise burst (simulating muscle activity).

    Args:
        n_times: The number of time points (samples) in the epoch.
        sfreq: The sampling frequency (Hz).

    Returns:
        A 1D array of EMG noise values (in Volts) with shape (n_times,).
    """
    # Create white noise
    noise = np.random.randn(n_times)
    
    # Convert to MNE Raw object to use MNE's filtering capabilities
    info = mne.create_info(ch_names=['noise'], sfreq=sfreq, ch_types=['eeg'])
    raw_noise = mne.io.RawArray(noise[None, :], info, verbose='error')
    
    # Bandpass filter the noise into the EMG frequency range
    raw_filtered = raw_noise.filter(
        l_freq=EMG_FREQ_RANGE_HZ[0], 
        h_freq=EMG_FREQ_RANGE_HZ[1], 
        picks='eeg', 
        n_jobs=1, 
        verbose='error'
    )
    emg_noise = raw_filtered.get_data(copy=True)[0]
    
    # Normalize and scale the noise to the desired amplitude range (in Volts)
    max_val = np.max(np.abs(emg_noise))
    if max_val > 0:
        # Scale to match the desired uV amplitude
        amp_uv = np.random.uniform(5.0, EMG_AMPLITUDE_UV)
        emg_noise = (emg_noise / max_val) * (amp_uv * UV_TO_V)
    
    return emg_noise

# --- Main Injection Function ---

def inject_stressors_into_epochs(
    epochs: Epochs, 
    add_shift: bool = True, 
    add_emg: bool = True
) -> Epochs:
    """
    Injects simulated motion artifacts and EMG noise into the EEG epochs.

    This creates the "target domain" data for adaptation experiments.

    Args:
        epochs: The clean MNE Epochs object (the "source domain" data).
        add_shift: If True, adds a low-frequency baseline shift.
        add_emg: If True, adds high-frequency EMG bursts stochastically.

    Returns:
        A new MNE Epochs object containing the clean data plus the simulated stressors.
    """
    # Create a deep copy to ensure the original clean data remains untouched
    stressed_epochs = epochs.copy()
    
    # Get the underlying data array (n_epochs, n_channels, n_times)
    data = stressed_epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = data.shape
    sfreq = stressed_epochs.info['sfreq']
    
    print(f"\n--- Injecting Stressors into {n_epochs} Epochs ---")
    
    n_emg_added = 0

    for i in range(n_epochs):
        # 1. Apply Baseline Shift / Drift
        if add_shift:
            # Generate a new shift signal for each channel and epoch
            shift_signal = generate_baseline_shift(n_times, sfreq)
            # Apply the shift to all channels
            data[i, :, :] += shift_signal[np.newaxis, :]
        
        # 2. Apply EMG Burst (Stochastic)
        if add_emg and np.random.rand() < EMG_PROBABILITY:
            # Generate a new EMG signal for each channel
            for ch in range(n_channels):
                emg_noise = generate_emg_burst(n_times, sfreq)
                data[i, ch, :] += emg_noise
            n_emg_added += 1
            
    print(f"  - Baseline shift applied to all epochs.")
    print(f"  - EMG bursts applied to {n_emg_added} out of {n_epochs} epochs.")
    
    # Update the Epochs object with the new, stressed data
    stressed_epochs._data = data
    
    # Optional: Log the amplitude change (not necessary but useful for debugging)
    
    return stressed_epochs

# --- Main Execution for Demonstration ---

if __name__ == '__main__':
    print("--- Stressor Simulation Test ---")
    
    # 1. Create a dummy MNE Epochs object for demonstration
    sfreq = 100.0
    n_channels = 3
    n_times = int(sfreq * 4.0) # 4 seconds of data
    n_epochs = 10
    
    # Dummy data: simple sine wave + random noise (simulating "clean" MI activity)
    t = np.arange(n_times) / sfreq
    base_signal = 10 * UV_TO_V * np.sin(2 * np.pi * 10 * t) # 10 Hz sinusoid, 10 uV
    dummy_data = base_signal[np.newaxis, np.newaxis, :] * np.ones((n_epochs, n_channels, n_times))
    dummy_data += 5 * UV_TO_V * np.random.randn(n_epochs, n_channels, n_times)
    
    # Dummy info and events
    ch_names = ['C3', 'Cz', 'C4']
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    events = np.c_[np.arange(n_epochs) * (n_times + 10), np.zeros(n_epochs, dtype=int), np.tile([1, 2], n_epochs // 2)]
    event_id = {1: 1, 2: 2}

    clean_epochs = mne.EpochsArray(
        dummy_data, 
        info, 
        events=events, 
        event_id=event_id, 
        tmin=0.0, 
        verbose='error'
    )
    print(f"Created dummy clean epochs: {len(clean_epochs)}")
    
    # 2. Inject stressors
    stressed_epochs = inject_stressors_into_epochs(clean_epochs, add_shift=True, add_emg=True)

    # 3. Compare amplitudes (simple check)
    clean_amplitude = np.max(np.abs(clean_epochs.get_data())) * V_TO_UV
    stressed_amplitude = np.max(np.abs(stressed_epochs.get_data())) * V_TO_UV
    
    print(f"\nMax clean amplitude: {clean_amplitude:.2f} µV")
    print(f"Max stressed amplitude: {stressed_amplitude:.2f} µV")
    
    if stressed_amplitude > clean_amplitude:
        print("Stress injection successful: Amplitude increased.")
    else:
        print("Warning: Stressed amplitude is not greater than clean amplitude. Check noise parameters.")
        
    # Optional: Plotting one clean vs. one stressed epoch to visualize the difference
    try:
        # Get data from the first epoch
        clean_data_plot = clean_epochs.get_data(copy=True)[0] * V_TO_UV
        stressed_data_plot = stressed_epochs.get_data(copy=True)[0] * V_TO_UV
        
        print("\nDisplaying C3 channel comparison (clean vs. stressed)...")
        # In a real setup, you would plot this using matplotlib/viz.py
        
    except Exception as e:
        print(f"Could not perform visualization comparison. Error: {e}")
    
    print("\n--- Stressor simulation test finished. ---")
