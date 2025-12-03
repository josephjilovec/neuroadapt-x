import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

class OnlineAccuracyTracker:
    """
    Tracks classification accuracy over a sliding window or a fixed number 
    of recent trials. Useful for real-time performance monitoring in the demo.
    """
    def __init__(self, window_size: int = 50):
        """
        Initializes the tracker.

        Args:
            window_size (int): The maximum number of trials to keep track of 
                               for calculating the sliding accuracy.
        """
        self.window_size = window_size
        self.predictions: List[int] = []
        self.true_labels: List[int] = []
        self.timestamps: List[float] = []
        
        # Performance metrics
        self.current_accuracy: float = 0.0
        self.current_f1: float = 0.0

    def record_trial(self, prediction: int, true_label: int):
        """
        Records the results of a single trial and updates the window.

        Args:
            prediction (int): The model's output class label.
            true_label (int): The ground truth class label.
        """
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.timestamps.append(time.time())

        # Enforce the sliding window size
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.true_labels.pop(0)
            self.timestamps.pop(0)
            
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculates accuracy and F1 score based on the current window."""
        if not self.predictions:
            self.current_accuracy = 0.0
            self.current_f1 = 0.0
            return

        # Calculate Accuracy
        self.current_accuracy = accuracy_score(self.true_labels, self.predictions)
        
        # Calculate F1 Score (assuming binary or multiclass setup)
        # Use 'weighted' for multiclass scenarios
        try:
            self.current_f1 = f1_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        except ValueError:
            # Handle cases where there might be only one class present in the window
            self.current_f1 = 0.0 

    def get_accuracy(self) -> float:
        """Returns the current sliding window accuracy."""
        return self.current_accuracy

    def get_f1_score(self) -> float:
        """Returns the current sliding window F1 score."""
        return self.current_f1

    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Returns the confusion matrix for the current window."""
        if not self.predictions:
            return None
        return confusion_matrix(self.true_labels, self.predictions)
    
    def get_window_size(self) -> int:
        """Returns the current number of samples in the window."""
        return len(self.predictions)

    def reset(self):
        """Clears all recorded data."""
        self.predictions = []
        self.true_labels = []
        self.timestamps = []
        self.current_accuracy = 0.0
        self.current_f1 = 0.0
        
# --- Utility Functions for Offline Evaluation ---

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the classification accuracy.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.

    Returns:
        float: The accuracy score.
    """
    return accuracy_score(y_true, y_pred)

def calculate_full_f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """
    Calculates the F1 score.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        average (str): Averaging method ('binary', 'micro', 'macro', 'weighted').

    Returns:
        float: The F1 score.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)

if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Testing Metrics Utilities ---")
    
    # 1. Test Offline Utilities
    true = np.array([0, 1, 2, 0, 1, 2])
    pred = np.array([0, 1, 1, 0, 1, 0])
    
    acc = calculate_accuracy(true, pred)
    f1 = calculate_full_f1_score(true, pred)
    
    print(f"Offline Accuracy: {acc:.4f}") # Should be 4/6 = 0.6667
    print(f"Offline F1 Score (weighted): {f1:.4f}")

    # 2. Test OnlineAccuracyTracker
    tracker = OnlineAccuracyTracker(window_size=5)
    
    # Sequence of trials (1=Correct, 0=Incorrect)
    trials = [
        (1, 1), (0, 0), # Correct
        (1, 0), (0, 1), # Incorrect
        (1, 1), # Correct
        (0, 0), # Correct (Window is full, first trial removed)
    ]
    
    print("\n--- Testing Online Tracker (Window Size = 5) ---")
    
    for i, (pred, true) in enumerate(trials):
        tracker.record_trial(pred, true)
        
        # Check window content (should be last 5)
        current_preds = tracker.predictions
        current_truths = tracker.true_labels
        
        print(f"Trial {i+1}: Pred={pred}, True={true}")
        print(f"  Window Size: {tracker.get_window_size()}")
        print(f"  Window Acc: {tracker.get_accuracy():.2f}, F1: {tracker.get_f1_score():.2f}")
        
    print("\nFinal Metrics:")
    print(f"Final Window Accuracy: {tracker.get_accuracy():.2f}") 
    print(f"Confusion Matrix:\n{tracker.get_confusion_matrix()}")
