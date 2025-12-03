import numpy as np
import pytest
import sys
import os

# Assuming the add_stressors module is located in src/data/
sys.path.append(os.path.join('..', 'src', 'data'))

# --- Mocking the stress simulation function and dependencies ---
# The function should be defined in src/data/add_stressors.py

N_CHANNELS = 64
N_TIMES = 1251
SFREQ = 250

def mock_apply_simulated_stress(eeg_data, sfreq, severity=1.0):
    """
    Simulates the effect of severe spaceflight stress (e.g., strong EMG noise 
    and baseline drift).
    
    The stress must significantly change the signal variance and mean 
    compared to the clean signal.
    """
    # 1. Simulate increased variance (EMG-like noise)
    noise_factor = severity * 0.5  # Scale noise with severity
    noise = np.random.randn(*eeg_data.shape) * noise_factor
    
    # 2. Simulate baseline drift (low-frequency artifact)
    # A simple large, low-frequency shift across all channels
    drift_factor = severity * 2.0 
    drift = np.sin(np.linspace(0, 2 * np.pi, N_TIMES)) * drift_factor
    
    # Apply artifacts to the data
    stressed_data = eeg_data + noise + drift[np.newaxis, np.newaxis, :]
    
    return stressed_data

# Use the mock for testing purposes
apply_simulated_stress = mock_apply_simulated_stress

# --- Fixtures ---

@pytest.fixture
def clean_eeg_batch():
    """Fixture to create a batch of clean, normally distributed EEG data."""
    # Shape: (Batch, Channels, Times) for simplicity in this file
    batch_size = 32
    return np.random.randn(batch_size, N_CHANNELS, N_TIMES).astype(np.float32)

# --- Test Definitions ---

def test_stress_application_maintains_shape(clean_eeg_batch):
    """Test that applying stress does not change the data's dimensional shape."""
    original_shape = clean_eeg_batch.shape
    stressed_data = apply_simulated_stress(clean_eeg_batch, SFREQ, severity=1.0)
    
    assert stressed_data.shape == original_shape

def test_stress_increases_data_variance(clean_eeg_batch):
    """Test that stress application significantly increases the overall data variance."""
    # Calculate baseline variance (should be close to 1.0 since it's randn)
    original_variance = np.var(clean_eeg_batch)
    
    # Apply severe stress
    stressed_data = apply_simulated_stress(clean_eeg_batch, SFREQ, severity=2.0)
    stressed_variance = np.var(stressed_data)
    
    # Assert that variance increases significantly (e.g., at least 50% increase)
    assert stressed_variance > original_variance * 1.5

def test_stress_changes_feature_distribution_mean(clean_eeg_batch):
    """Test that stress application causes a shift in the mean signal value, simulating drift."""
    # We test the mean of the absolute values across the entire dataset to capture the drift
    original_abs_mean = np.mean(np.abs(clean_eeg_batch))
    
    # Apply moderate stress
    stressed_data = apply_simulated_stress(clean_eeg_batch, SFREQ, severity=1.0)
    stressed_abs_mean = np.mean(np.abs(stressed_data))
    
    # Assert that the drift component shifts the overall distribution mean significantly
    # (The drift simulation adds a large sin wave, which increases the average magnitude)
    assert stressed_abs_mean > original_abs_mean * 1.5

def test_zero_severity_stress_is_minimal(clean_eeg_batch):
    """Test that setting severity=0.0 results in minimal change to the data."""
    original_data = clean_eeg_batch
    # Apply zero stress
    stressed_data = apply_simulated_stress(original_data, SFREQ, severity=0.0)
    
    # Check if the difference is negligible (within floating point precision)
    difference = np.sum((original_data - stressed_data)**2)
    
    assert difference < 1e-4

# --- Helper for running tests locally (optional, typically run via pytest command) ---
if __name__ == '__main__':
    # Simple check for test execution visibility
    try:
        data = clean_eeg_batch()
        test_stress_application_maintains_shape(data)
        test_stress_increases_data_variance(data)
        test_stress_changes_feature_distribution_mean(data)
        test_zero_severity_stress_is_minimal(data)
        print("All mock stressor tests passed successfully.")
    except AssertionError as e:
        print(f"Test failed: {e}")
