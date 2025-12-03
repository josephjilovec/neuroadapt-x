import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet
from typing import List

# --- LSL Stream Configuration (Must match consumer expectations) ---
STREAM_NAME = 'MockEEGStream'
STREAM_TYPE = 'EEG'
# The streamer is expected to look for these channels, which are common for MI
CH_NAMES: List[str] = ['C3', 'Cz', 'C4', 'Pz', 'O1'] 
N_CHANNELS = len(CH_NAMES)
# We match the assumed high sampling rate to simulate real hardware
SFREQ = 250 
DTYPE = 'float32'

# --- Simulation Parameters ---
# Data chunk size (determines how often data is pushed)
CHUNK_SIZE = 25 
# Base noise level (simulates microvolts)
BASE_NOISE_STD = 5.0 
# Time in seconds before the "stress" condition starts
STRESS_START_TIME = 20
# Duration of the "stress" period in seconds
STRESS_DURATION = 30 
# Additional drift/noise during stress
STRESS_DRIFT_AMPLITUDE = 10.0
STRESS_NOISE_FACTOR = 3.0

def generate_eeg_data(n_samples: int, t: float, is_stressed: bool) -> np.ndarray:
    """
    Generates synthetic EEG data with optional stress-induced noise and drift.

    Args:
        n_samples (int): Number of time points to generate.
        t (float): Current simulation time for generating drift.
        is_stressed (bool): If True, introduces high noise and baseline drift.

    Returns:
        np.ndarray: Generated data array of shape (n_samples, N_CHANNELS).
    """
    # 1. Base EEG activity (low frequency, non-zero mean)
    base_signal = np.sin(2 * np.pi * 10 * np.arange(n_samples) / SFREQ) * 1.0 
    
    # 2. Random noise (baseline noise)
    noise_std = BASE_NOISE_STD * (STRESS_NOISE_FACTOR if is_stressed else 1.0)
    noise = np.random.normal(0, noise_std, size=(n_samples, N_CHANNELS))
    
    # 3. Simulated MI component (subtle 12Hz or 20Hz rhythm, mostly in C3/C4)
    # We use a subtle 20Hz signal, simulating a brief activity burst
    mi_freq = 20.0 
    time_points = np.arange(n_samples) / SFREQ
    mi_signal = np.zeros((n_samples, N_CHANNELS))
    
    # Simulate a 1-second burst of 20Hz activity every 10 seconds (for testing)
    if (int(t) % 10 < 1):
        mi_signal[:, 0] = np.sin(2 * np.pi * mi_freq * time_points) * 3.0 # C3
        mi_signal[:, 2] = np.sin(2 * np.pi * mi_freq * time_points) * 3.0 # C4

    # 4. Stress drift (low-frequency baseline shift)
    drift = 0.0
    if is_stressed:
        # Simulate a slow, low-frequency drift component specific to stress/fatigue
        drift_factor = np.sin(2 * np.pi * 0.1 * t) * STRESS_DRIFT_AMPLITUDE
        drift = drift_factor * np.ones((n_samples, N_CHANNELS))
        
    # Combine signals: Base + MI + Noise + Drift
    data = base_signal[:, np.newaxis] + mi_signal + noise + drift
    
    return data

def stream_data():
    """
    Initializes the LSL Outlet and streams synthetic data continuously.
    """
    print("Initializing LSL Stream Outlet...")
    
    # 1. Create StreamInfo
    info = StreamInfo(
        name=STREAM_NAME, 
        type=STREAM_TYPE, 
        channel_count=N_CHANNELS, 
        nominal_srate=SFREQ, 
        channel_format=DTYPE, 
        source_id='mock_eeg_generator'
    )
    
    # Add channel names metadata
    channels = info.desc().append_child("channels")
    for ch_name in CH_NAMES:
        chan = channels.append_child("channel")
        chan.append_child_value("label", ch_name)
        chan.append_child_value("unit", "microvolts")
        chan.append_child_value("type", STREAM_TYPE)
        
    # 2. Create StreamOutlet
    outlet = StreamOutlet(info)
    print(f"LSL Stream '{STREAM_NAME}' is now open and streaming at {SFREQ} Hz.")
    print(f"Channels: {CH_NAMES}")
    
    start_time = time.time()
    total_samples = 0
    
    try:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check for the stress condition based on elapsed time
            is_stressed = (elapsed_time >= STRESS_START_TIME and 
                           elapsed_time < STRESS_START_TIME + STRESS_DURATION)
            
            if is_stressed:
                # Add a clear marker for when the stress period is active
                print(f"\r[Time: {elapsed_time:.1f}s] STREAMING STRESSED DATA...", end="")
            elif elapsed_time >= STRESS_START_TIME + STRESS_DURATION:
                # After stress period ends
                print(f"\r[Time: {elapsed_time:.1f}s] Streaming normal data (Recovery).", end="")
            else:
                # Normal period before stress
                print(f"\r[Time: {elapsed_time:.1f}s] Streaming normal data...", end="")

            # 3. Generate data chunk (N_samples, N_channels)
            chunk = generate_eeg_data(CHUNK_SIZE, elapsed_time, is_stressed)
            
            # 4. Push the chunk to the LSL stream
            # push_chunk expects a list of samples, where each sample is a list of channel values
            # Our chunk is (samples x channels), which is what push_chunk expects if it's a numpy array
            outlet.push_chunk(chunk)
            total_samples += CHUNK_SIZE
            
            # 5. Maintain the correct sampling rate
            # Calculate the time to wait until the next chunk is due
            time_per_chunk = CHUNK_SIZE / SFREQ
            time_spent = time.time() - current_time
            sleep_time = time_per_chunk - time_spent
            
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nLSL Stream simulation stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print(f"Total samples streamed: {total_samples}")

if __name__ == '__main__':
    # NOTE: You need the 'pylsl' library installed (pip install pylsl) to run this script.
    # The LSLStreamer (in realtime_processor.py) will connect to this stream.
    stream_data()
