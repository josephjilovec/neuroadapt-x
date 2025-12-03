import numpy as np
from scipy.signal import butter, lfilter
import os
from typing import Callable, Dict, Union

# --- Configuration Constants ---
# Assuming a standard neurophysiological recording setup (e.g., EEG)
SFREQ = 250  # Sample frequency in Hz
N_CHANNELS = 32  # Number of channels (electrodes)
DURATION_SECONDS = 10  # Length of the simulated data epoch
N_SAMPLES = SFREQ * DURATION_SECONDS

# Define the path where mock clean data would be loaded and stressed data saved
DATA_DIR = os.path.join(os.path.dirname(__file__), 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


# --- Utility Functions ---

def load_clean_data(n_samples: int, n_channels: int) -> np.ndarray:
    """
    Simulates loading a clean neurophysiological dataset (e.g., EEG).
    
    The array shape is (n_channels, n_samples).
    
    In a real application, this function would load a file (e.g., .fif, .edf)
    from the 'raw' directory.
    """
    print(f"Simulating clean data ({n_channels} channels, {DURATION_SECONDS}s)...")
    
    # Generate a simple mock EEG signal: Alpha wave (10Hz) + background noise
    t = np.linspace(0, DURATION_SECONDS, n_samples, endpoint=False)
    
    clean_data = np.zeros((n_channels, n_samples))
    
    for i in range(n_channels):
        # Base signal: Sine wave (Alpha band)
        base_signal = 5 * np.sin(2 * np.pi * 10 * t)
        
        # Channel variability: Small Gaussian random walk to simulate baseline variance
        walk = np.cumsum(np.random.normal(0, 0.1, n_samples))
        
        # Add low-amplitude white noise
        noise = np.random.normal(0, 1.5, n_samples)
        
        # The clean signal is the sum of these components
        clean_data[i, :] = base_signal + walk + noise + (i * 0.5) # Slight offset per channel
        
    return clean_data

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


# --- Stressor Simulation Functions ---

def simulate_ocular_artifact(data: np.ndarray, channels: list, amplitude_factor: float = 10.0) -> np.ndarray:
    """
    Simulates a characteristic Electrooculogram (EOG) artifact, like an eye blink.
    
    EOG artifacts are typically slow, high-amplitude deflections, most prominent
    in frontal channels (e.g., Fp1, Fp2).
    """
    print(f"Applying Ocular Artifact (EOG) to channels {channels}...")
    
    # Create a transient, slow-frequency 'blink' signal
    blink_duration_samples = int(SFREQ * 0.25)
    blink_start = np.random.randint(SFREQ, N_SAMPLES - SFREQ - blink_duration_samples)
    
    # Create a Hann window shape for a smooth blink
    blink_signal = amplitude_factor * np.hanning(blink_duration_samples)
    
    for ch_idx in channels:
        # Add the blink at a random location
        data[ch_idx, blink_start:blink_start + blink_duration_samples] += blink_signal
        
        # Add a corresponding slow drift after the blink (common EOG effect)
        slow_drift = np.linspace(0, amplitude_factor / 3, N_SAMPLES - blink_start)
        data[ch_idx, blink_start:] += slow_drift[:data.shape[1] - blink_start]
        
    return data

def simulate_muscle_artifact(data: np.ndarray, channels: list, intensity: float = 0.5) -> np.ndarray:
    """
    Simulates Electromyogram (EMG) artifact, which is high-frequency noise 
    often seen in temporal/frontal channels due to muscle tension.
    """
    print(f"Applying Muscle Artifact (EMG) to channels {channels}...")
    
    # Muscle noise is broadband, usually above 20 Hz (beta/gamma range)
    # 1. Generate high-frequency noise
    high_freq_noise = np.random.normal(0, 1, N_SAMPLES)
    
    # 2. Filter it to emphasize high frequencies (e.g., 50-100 Hz bandpass)
    emg_kernel = bandpass_filter(high_freq_noise, 50, 100, SFREQ)
    
    # 3. Apply the burst randomly across a time window
    burst_duration = int(SFREQ * 3) # 3 second burst
    burst_start = np.random.randint(0, N_SAMPLES - burst_duration)
    
    # Taper the beginning and end of the burst
    taper = np.hanning(burst_duration)
    
    # Scale and apply the burst to selected channels
    emg_burst = intensity * emg_kernel[burst_start:burst_start + burst_duration] * taper
    
    for ch_idx in channels:
        data[ch_idx, burst_start:burst_start + burst_duration] += emg_burst
        
    return data


def simulate_line_noise(data: np.ndarray, frequency: float = 60.0, amplitude: float = 5.0) -> np.ndarray:
    """
    Simulates 50/60 Hz power line interference (technical artifact).
    """
    print(f"Applying {frequency} Hz Line Noise...")
    t = np.linspace(0, DURATION_SECONDS, N_SAMPLES, endpoint=False)
    
    # Create a sine wave at the line frequency
    line_noise = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add the noise to all channels
    data += line_noise
    
    return data


def simulate_slow_drift(data: np.ndarray, channels: list, max_amplitude: float = 15.0) -> np.ndarray:
    """
    Simulates very slow baseline drift, common due to sweat, respiration, or loose electrodes.
    """
    print(f"Applying Slow Baseline Drift to channels {channels}...")
    
    # Use a random walk or a very low frequency sine wave to simulate drift
    t = np.linspace(0, DURATION_SECONDS, N_SAMPLES, endpoint=False)
    drift_signal = max_amplitude * (np.sin(2 * np.pi * 0.1 * t) + np.random.rand(N_SAMPLES))
    drift_signal = (drift_signal - drift_signal.min()) / (drift_signal.max() - drift_signal.min()) # Normalize [0, 1]
    drift_signal *= max_amplitude # Scale back up
    
    for ch_idx in channels:
        data[ch_idx, :] += drift_signal
        
    return data


def generate_stressed_data(clean_data: np.ndarray, stressor_config: Dict[str, Union[list, dict]]) -> np.ndarray:
    """
    Applies a combination of stressors based on the configuration.
    """
    stressed_data = clean_data.copy()
    
    # --- 1. Line Noise (Always technical, affects all channels) ---
    if 'line_noise' in stressor_config:
        cfg = stressor_config['line_noise']
        stressed_data = simulate_line_noise(
            stressed_data, 
            frequency=cfg.get('frequency', 60.0), 
            amplitude=cfg.get('amplitude', 5.0)
        )

    # --- 2. Ocular Artifact (Physiological, frontal channels) ---
    if 'eog' in stressor_config:
        cfg = stressor_config['eog']
        # Assign a few random frontal channels to be affected
        affected_channels = np.random.choice(N_CHANNELS, size=cfg.get('num_channels', 2), replace=False).tolist()
        stressed_data = simulate_ocular_artifact(
            stressed_data, 
            channels=affected_channels, 
            amplitude_factor=cfg.get('amplitude_factor', 12.0)
        )

    # --- 3. Muscle Artifact (Physiological, high-frequency, local) ---
    if 'emg' in stressor_config:
        cfg = stressor_config['emg']
        # Assign a few random temporal/motor channels
        affected_channels = np.random.choice(N_CHANNELS, size=cfg.get('num_channels', 3), replace=False).tolist()
        stressed_data = simulate_muscle_artifact(
            stressed_data, 
            channels=affected_channels, 
            intensity=cfg.get('intensity', 0.8)
        )

    # --- 4. Slow Drift (Physiological/Technical, varying channels) ---
    if 'slow_drift' in stressor_config:
        cfg = stressor_config['slow_drift']
        affected_channels = np.random.choice(N_CHANNELS, size=cfg.get('num_channels', 5), replace=False).tolist()
        stressed_data = simulate_slow_drift(
            stressed_data, 
            channels=affected_channels, 
            max_amplitude=cfg.get('max_amplitude', 15.0)
        )
        
    return stressed_data


# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Load the clean (mock) data
    clean_data = load_clean_data(N_SAMPLES, N_CHANNELS)
    
    # 2. Define the stressors to apply and their parameters
    # This configuration can be easily modified to stress-test specific algorithms
    stress_profile_config = {
        'line_noise': {
            'frequency': 60.0,
            'amplitude': 6.0
        },
        'eog': {
            'amplitude_factor': 15.0, # High amplitude blink
            'num_channels': 3
        },
        'emg': {
            'intensity': 1.0, 
            'num_channels': 5
        },
        'slow_drift': {
            'max_amplitude': 18.0, 
            'num_channels': 4
        }
    }
    
    # 3. Generate the stressed version
    stressed_data = generate_stressed_data(clean_data, stress_profile_config)
    
    # 4. Save the results
    # In a real scenario, you'd save this with channel names and sample info
    output_filename = os.path.join(PROCESSED_DIR, 'stressed_data.npy')
    np.save(output_filename, stressed_data)
    
    print("-" * 50)
    print(f"Successfully generated stressed data!")
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Stressed data shape: {stressed_data.shape}")
    print(f"Data saved to: {output_filename}")
    print("Run this script to generate a new, randomly artifacted dataset.")
