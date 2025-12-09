import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from typing import Optional, Tuple

# --- Project Imports (Mocks used for standalone execution) ---
# We assume these are available in their respective directories:

try:
    from ..data.preprocess import DataProcessor, preprocess_raw_data # Functions for filtering, epoching
    from ..models.adaptive import AdaptiveEEGNet, CORALLoss        # Adaptive model and loss
    from ..models.eegnet import EEGNet                            # Base EEGNet model
    from .lsl_stream import LSLStreamer, TARGET_CHANNELS, TARGET_SFREQ # LSL Interface
except ImportError:
    print("Warning: Running realtime_processor.py standalone. Using Mock classes.")

    # Mock Classes for standalone execution
    class MockDataProcessor:
        def __init__(self, sfreq, target_sfreq, ch_names): 
            self.target_sfreq = target_sfreq
            self.target_channels = TARGET_CHANNELS
            self.n_channels = len(TARGET_CHANNELS)
            self.epoch_duration = 4.0 # 4 seconds
            self.n_times = int(self.epoch_duration * self.target_sfreq)

        def preprocess_raw_data(self, data, ch_names):
            """Mocks the preprocessing steps: filtering, downsampling, selecting channels."""
            data = data[:self.n_channels, :]
            # Mock downsampling: keep only n_times samples 
            # In a real scenario, interpolation/resampling would happen here.
            
            # Pad or truncate to match expected time dimension for the model (N_TIMES)
            if data.shape[1] < self.n_times:
                 # This is highly simplified and unrealistic for real-time, but sufficient for the mock
                X_epoch = np.zeros((self.n_channels, self.n_times))
                X_epoch[:, :data.shape[1]] = data
            else:
                X_epoch = data[:, -self.n_times:]
                
            # X_epoch shape is (C, T) -> (3, 400)
            return X_epoch

    class MockModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.output = nn.Linear(10, 2)
            self.n_features_before_fc = 400 * 16 # Mock feature size
        def forward(self, x):
            # Mock returns logits and features
            batch_size = x.size(0)
            logits = self.output(torch.randn(batch_size, 10))
            features = torch.randn(batch_size, self.n_features_before_fc)
            return logits, features
        def set_adaptation_mode(self, is_adapt: bool): pass
        def set_adaptation_mode(self, is_adapt: bool): pass
    
    class MockLSLStreamer:
        def __init__(self, *args, **kwargs): 
            self.is_connected = False
            self.info = {'sfreq': 250, 'ch_names': TARGET_CHANNELS, 'target_sfreq': 100}
        def find_stream(self): 
            self.is_connected = True
            print("Mock Streamer connected.")
            return True
        def get_info(self): return self.info
        def get_latest_data(self, n_samples):
            """Mock data: (C, T)"""
            if n_samples is None: return None
            # Shape: (3 channels, n_samples) - simulating actual LSL output
            return np.random.randn(len(TARGET_CHANNELS), n_samples) * 1e-5
        def close(self): pass

    DataProcessor = MockDataProcessor
    preprocess_raw_data = MockDataProcessor.preprocess_raw_data
    AdaptiveEEGNet = MockModel
    LSLStreamer = MockLSLStreamer
    
    # We still need a real PyTorch Loss class to run the adaptation step
    class CORALLoss(nn.Module):
        def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
            return torch.tensor(1.0) # Mock loss


# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The duration of a single BCI command epoch (4.0s)
EPOCH_DURATION = 4.0 
# The amount of new data to fetch in each loop iteration (0.1s is 10Hz decoding rate)
BLOCK_DURATION = 0.1 
# Threshold for adaptation: If model confidence drops below this, adaptation begins.
CONFIDENCE_THRESHOLD = 0.8 
# Max size of the adaptation buffer (e.g., last 20 epochs)
ADAPTATION_BUFFER_SIZE = 20 
# How many batches to use for online adaptation
ADAPTATION_BATCHES = 4 
CORAL_LAMBDA = 0.5 

class RealTimeProcessor:
    """
    Core engine for real-time BCI decoding, managing data acquisition, 
    preprocessing, classification, and online domain adaptation.
    """
    def __init__(self, model_path: str, stream_name: Optional[str] = None):
        
        # --- LSL Setup ---
        # The buffer size must be at least the epoch duration
        self.streamer = LSLStreamer(stream_name=stream_name, bufsize=EPOCH_DURATION * 1.5)
        if not self.streamer.find_stream():
            raise ConnectionError("Failed to connect to LSL stream.")
        
        self.stream_info = self.streamer.get_info()
        self.actual_sfreq = int(self.stream_info['sfreq'])
        
        # --- Data & Preprocessing Setup ---
        self.data_processor = DataProcessor(
            sfreq=self.actual_sfreq, 
            target_sfreq=TARGET_SFREQ, 
            ch_names=self.stream_info['ch_names']
        )
        self.N_CHANNELS = len(TARGET_CHANNELS)
        self.N_TIMES = int(EPOCH_DURATION * TARGET_SFREQ)
        
        # Calculate samples to fetch in each iteration
        self.samples_to_fetch = int(BLOCK_DURATION * self.actual_sfreq)
        
        # Initialize ring buffer for storing the current epoch
        # Stores (N_CHANNELS, N_TIMES)
        self.epoch_buffer = deque(maxlen=self.N_TIMES)
        
        # --- Model Setup ---
        
        # Load the base model (assuming it's a pre-trained EEGNet)
        base_model = EEGNet(self.N_CHANNELS, self.N_TIMES, n_classes=2)
        
        # Wrap it in the Adaptive model
        self.model = AdaptiveEEGNet(base_model).to(DEVICE)
        
        # Load weights from offline training (if available)
        try:
            # Note: We must load the AdaptiveEEGNet state dict, not the base one
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded adaptive model weights from {model_path}")
        except Exception as e:
            print(f"Error loading model weights from {model_path}: {e}. Using random weights.")

        self.model.eval()
        self.model.set_adaptation_mode(False) # Start in standard inference mode
        
        # --- Adaptation Buffer ---
        self.adaptation_buffer = deque(maxlen=ADAPTATION_BUFFER_SIZE)
        self.is_adaptation_active = False

    def _get_current_epoch(self) -> Optional[np.ndarray]:
        """
        Fetches the latest block of data, preprocesses it, and forms the current epoch.
        Returns the processed epoch of shape (C, T) or None.
        """
        raw_block = self.streamer.get_latest_data(self.samples_to_fetch)
        
        if raw_block is None:
            return None

        # 1. Update the ring buffer with the new block (C, T_block)
        for t in range(raw_block.shape[1]):
            # Append samples one-by-one to maintain time order in the deque
            self.epoch_buffer.append(raw_block[:, t]) 
        
        # 2. Check if we have enough data for a full epoch (C, T)
        if len(self.epoch_buffer) < self.N_TIMES:
            return None
            
        # 3. Assemble the raw epoch: (N_TIMES, N_CHANNELS) -> transpose to (N_CHANNELS, N_TIMES)
        raw_epoch = np.array(self.epoch_buffer).T 

        # 4. Preprocess (filtering, downsampling, channel selection)
        processed_epoch = self.data_processor.preprocess_raw_data(
            raw_epoch, self.stream_info['ch_names']
        )
        
        # Final shape check (C, T) -> (3, 400)
        if processed_epoch.shape != (self.N_CHANNELS, self.N_TIMES):
            print(f"Error: Processed epoch shape mismatch: {processed_epoch.shape}")
            return None

        # Only keep the last N_TIMES samples in the buffer for the next iteration (sliding window)
        # This is implicitly handled by the deque maxlen, but we need to remove the oldest BLOCK_DURATION 
        # of data for the next fetch cycle to represent a true sliding window.
        
        # NOTE: A simpler, more robust method for LSL is often just to ask for the 
        # last N_SAMPLES in every call, which is what LSLStreamer.get_latest_data() 
        # already does if N_SAMPLES == N_TIMES. However, since we're simulating 
        # classification *every* BLOCK_DURATION, we use the buffer for sliding window.
        
        # Since we use BLOCK_DURATION, we need to remove the oldest block amount  
        # of data from the start of the buffer to simulate a sliding window step.
        for _ in range(self.samples_to_fetch):
            if self.epoch_buffer:
                self.epoch_buffer.popleft() 
        
        return processed_epoch

    def _classify_epoch(self, epoch: np.ndarray) -> Tuple[int, float]:
        """
        Classifies the single epoch and returns the prediction and confidence.
        """
        # Convert numpy array (C, T) to PyTorch tensor (1, C, T)
        X_tensor = torch.from_numpy(epoch).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # AdaptiveEEGNet returns (logits, features)
            logits, features = self.model(X_tensor) 

        # Get confidence (Softmax or LogSoftmax) and prediction
        import torch.nn.functional as F
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        # Add the features and the epoch to the adaptation buffer if in adaptation mode
        if self.is_adaptation_active:
            # We store the features needed for CORAL loss calculation later
            self.adaptation_buffer.append({
                'features': features.squeeze(0).detach().cpu(), # (D,)
                'epoch': epoch # (C, T)
            })

        return predicted_class.item(), confidence.item()

    def _perform_online_adaptation(self):
        """
        Trains the AdaptiveEEGNet on the accumulated target data (stressed features)
        using unsupervised CORAL loss.
        """
        if len(self.adaptation_buffer) < ADAPTATION_BATCHES:
            # Need a minimum number of samples to form batches
            return 
        
        print(f"\n[ADAPTATION] Starting online adaptation with {len(self.adaptation_buffer)} epochs.")
        
        # Prepare target data (features for CORAL)
        target_features_list = [d['features'] for d in self.adaptation_buffer]
        target_features = torch.stack(target_features_list).to(DEVICE) # (N, D)

        # Prepare dummy source data (We need to use *real* source data covariance C_S,
        # but for this script, we mock it as the first few target features for simplicity)
        # NOTE: In a real system, C_S would be computed once from the clean training set.
        source_features = target_features[:ADAPTATION_BATCHES].clone() 
        
        # --- Adaptation Training Setup ---
        self.model.train()
        self.model.set_adaptation_mode(True) # Use batch statistics for target domain
        
        import torch.optim as optim
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=1e-5 # Very low learning rate for gentle online adaptation
        )
        criterion_coral = CORALLoss()
        
        # Train for a few batches (simulating online fine-tuning)
        for i in range(ADAPTATION_BATCHES):
            optimizer.zero_grad()
            
            # Use chunks of the target features as batches
            batch_T = target_features[i:i+1]
            batch_S = source_features[i:i+1] 
            
            # --- CORAL Loss: Align feature covariance (Unsupervised) ---
            loss_coral = criterion_coral(batch_S, batch_T)
            
            # Since this is pure unsupervised DA, the loss is just CORAL loss
            total_loss = CORAL_LAMBDA * loss_coral
            
            total_loss.backward()
            optimizer.step()
        
        # Cleanup
        self.adaptation_buffer.clear()
        self.model.eval()
        self.model.set_adaptation_mode(False)
        print("[ADAPTATION] Complete. Model returned to standard inference mode.")

    def run_loop(self, adaptation_enabled: bool = True):
        """
        The main real-time loop for fetching data, classifying, and adapting.
        """
        
        try:
            print("\n--- Starting Real-Time BCI Loop ---")
            print(f"Decoding rate: 1 command/{BLOCK_DURATION}s")
            
            # Pre-populate the buffer with initial data for the first classification
            print("Pre-buffering initial data...")
            for _ in range(int(EPOCH_DURATION / BLOCK_DURATION)):
                time.sleep(BLOCK_DURATION)
                raw_block = self.streamer.get_latest_data(self.samples_to_fetch)
                if raw_block is not None:
                    for t in range(raw_block.shape[1]):
                        self.epoch_buffer.append(raw_block[:, t])
            
            
            start_time = time.time()
            i = 0
            while True:
                i += 1
                loop_start = time.time()
                
                # 1. Get the latest epoch (sliding window)
                epoch = self._get_current_epoch()
                
                if epoch is None:
                    # Not enough data for a full epoch yet, wait for the next block
                    time.sleep(BLOCK_DURATION)
                    continue

                # 2. Classify
                command_idx, confidence = self._classify_epoch(epoch)
                
                # 3. Output Command
                commands = {0: "LEFT (Simulated Rover Control)", 1: "RIGHT (Simulated Rover Control)"}
                command_text = commands.get(command_idx, "REST/UNKNOWN")
                
                print(f"[{i:04d}] Command: {command_text} | Confidence: {confidence:.3f}", end="")
                
                # 4. Stress Resilience / Adaptation Logic
                if adaptation_enabled:
                    if confidence < CONFIDENCE_THRESHOLD:
                        if not self.is_adaptation_active:
                            self.is_adaptation_active = True
                            print(f" | !!! LOW CONFIDENCE DETECTED ({confidence:.3f}) - ACTIVATING ADAPTATION !!!", end="")
                        
                        # Accumulate features for online adaptation
                        if len(self.adaptation_buffer) >= ADAPTATION_BUFFER_SIZE:
                            self._perform_online_adaptation()
                            self.is_adaptation_active = False # Deactivate after a successful cycle
                    elif self.is_adaptation_active:
                         # Still active but confidence recovered (still accumulate)
                        print(f" | Adaptation Active (Buf Size: {len(self.adaptation_buffer)})", end="")
                    
                print() # Newline

                # 5. Wait for the next block period to maintain decoding rate
                processing_time = time.time() - loop_start
                sleep_time = BLOCK_DURATION - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nProcessor stopped by user.")
        except ConnectionError as e:
            print(f"\nFatal Error: {e}")
        finally:
            self.streamer.close()
            print("Real-Time Processor shut down.")

# --- Main Execution ---

if __name__ == '__main__':
    # This path should point to the file saved by src/models/train.py
    MODEL_WEIGHTS_PATH = "adaptive_eegnet_final_weights.pth" 
    
    # NOTE: To run this successfully, you need an LSL stream running 
    # (e.g., OpenBCI GUI, or the fallback_stream.py we will create next).
    
    try:
        processor = RealTimeProcessor(
            model_path=MODEL_WEIGHTS_PATH,
            # stream_name="OpenBCI-XXXX" # Uncomment to specify a stream name
        )
        processor.run_loop(adaptation_enabled=True)
        
    except ConnectionError:
        print("\nCANNOT RUN DEMO: Please start an LSL stream (e.g., OpenBCI) or use the fallback_stream.py script.")
        print("Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
