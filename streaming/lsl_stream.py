import mne
import mne_lsl
import time
import numpy as np
from typing import Optional, Dict

# --- Configuration (Must match preprocessing expectations) ---
# Note: These values should ideally be loaded from a config file or determined dynamically
# from the LSL stream info, but are hardcoded for initial structure.
TARGET_CHANNELS = ['C3', 'Cz', 'C4'] # Motor Imagery related channels
TARGET_SFREQ = 100                   # Target sampling rate after downsampling (if needed)

class LSLStreamer:
    """
    Manages connection to a live LSL stream and retrieves raw data blocks.
    
    This class wraps the mne-lsl StreamLSL utility for easy integration into 
    the real-time processing pipeline.
    """
    
    def __init__(self, stream_name: Optional[str] = None, bufsize: float = 5.0):
        """
        Initializes the streamer.

        Args:
            stream_name (str, optional): The name of the LSL stream to connect to. 
                                         If None, waits for the first available stream.
            bufsize (float): The maximum duration (in seconds) of data to buffer.
        """
        self.stream_name = stream_name
        self.bufsize = bufsize
        self.stream = None
        self.sfreq = None
        self.ch_names = None
        self.is_connected = False
        print(f"LSL Streamer initialized. Buffering up to {bufsize} seconds.")

    def find_stream(self) -> bool:
        """
        Searches for a suitable EEG stream on the network.
        """
        print(f"Searching for LSL stream: '{self.stream_name if self.stream_name else 'any EEG stream'}'...")
        
        # Use mne_lsl.lsl.resolve_streams to find available streams
        streams = mne_lsl.lsl.resolve_streams(wait_max=10.0, stype='EEG')
        
        if not streams:
            print("Error: No EEG stream found after 10 seconds.")
            return False
            
        if self.stream_name:
            # Filter by name if specified
            found_streams = [s for s in streams if self.stream_name in s.name]
            if not found_streams:
                print(f"Error: Stream '{self.stream_name}' not found. Available names: {[s.name for s in streams]}")
                return False
            selected_stream = found_streams[0]
        else:
            # Select the first one found
            selected_stream = streams[0]

        print(f"Found stream: {selected_stream.name} (Source: {selected_stream.source_id})")
        
        try:
            # Create the MNE-LSL stream object
            self.stream = mne_lsl.stream.StreamLSL(
                info=selected_stream, 
                bufsize=self.bufsize,
                high_pass=0.5, # Apply a light high-pass filter to remove DC offset/drift early
                low_pass=TARGET_SFREQ / 2 - 1 # Ensure data is pre-filtered for desired output rate
            )
            
            # Extract metadata
            self.sfreq = self.stream.info['sfreq']
            self.ch_names = self.stream.info['ch_names']
            
            # Check for channel availability
            available_channels = set(self.ch_names)
            missing_channels = [ch for ch in TARGET_CHANNELS if ch not in available_channels]
            
            if missing_channels:
                print(f"Warning: Missing required channels {missing_channels}. Available: {self.ch_names[:5]}...")
                # We can proceed but preprocessing will need to handle this.

            self.is_connected = True
            print(f"Connected successfully. Actual SFreq: {self.sfreq:.2f}Hz, Channels: {len(self.ch_names)}")
            return True
            
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            self.is_connected = False
            return False

    def get_latest_data(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Gets the latest n_samples from the stream's buffer.

        Args:
            n_samples (int): The number of time points (samples) to retrieve.

        Returns:
            Optional[np.ndarray]: A numpy array of shape (n_channels, n_samples) 
                                  containing the most recent data block, or None if not connected.
        """
        if not self.is_connected:
            print("Error: Streamer not connected.")
            return None
        
        # Pull the data from the LSL stream buffer
        # This function returns (channels x samples)
        data, timestamps = self.stream.get_data(
            n_samples=n_samples, 
            picks='eeg', 
            return_times=True
        )

        if data is None or data.shape[1] < n_samples:
            # Data might be None at startup or if stream is slow
            # print(f"Warning: Only received {data.shape[1] if data is not None else 0}/{n_samples} samples.")
            return None

        # Transpose to (samples x channels) if needed, but MNE typically uses (channels x samples)
        # We will keep (channels x samples) and handle final reshaping in realtime_processor.py
        return data

    def close(self):
        """
        Closes the LSL stream connection.
        """
        if self.stream:
            self.stream.close()
            print("LSL Streamer closed.")
        self.is_connected = False
    
    def get_info(self) -> Dict:
        """Returns the stream metadata."""
        return {
            'sfreq': self.sfreq,
            'ch_names': self.ch_names,
            'target_channels': TARGET_CHANNELS,
            'target_sfreq': TARGET_SFREQ
        }

# --- Main Execution for Testing ---

if __name__ == '__main__':
    # This block requires a running LSL stream (e.g., OpenBCI, Muse)
    
    # 1. Instantiate the streamer
    # If you know your stream name, use it, e.g., LSLStreamer(stream_name="OpenBCI_EEG")
    streamer = LSLStreamer(bufsize=5.0) 
    
    # 2. Try to connect
    if streamer.find_stream():
        info = streamer.get_info()
        print("\n--- Stream Info ---")
        print(info)
        
        # 3. Calculate the required number of samples for a 1-second block
        # In a real scenario, this should be TARGET_SFREQ (100)
        n_samples_per_block = int(info['sfreq']) 
        
        print(f"\nPulling {n_samples_per_block} samples per second...")
        
        # 4. Start receiving data blocks
        for i in range(5):
            block = streamer.get_latest_data(n_samples_per_block)
            
            if block is not None:
                # Shape is (n_channels, n_samples)
                print(f"Received block {i+1}: Shape {block.shape}, Max/Min value: {block.max():.2f}/{block.min():.2f} V")
            else:
                print(f"Received block {i+1}: Not enough data yet.")
                
            time.sleep(1) # Wait for 1 second before pulling the next block
            
    # 5. Close the stream
    streamer.close()
    print("Testing complete.")
