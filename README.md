NeuroAdapt-X: Dynamic Resilience for BCI Decoders

Project Overview

NeuroAdapt-X is a proof-of-concept project demonstrating a solution for Domain Shift in Brain-Computer Interfaces (BCIs). BCI decoders, often trained on clean, lab-collected data (the Source Domain), fail catastrophically when deployed in real-world, high-noise environments like spaceflight (the Target Domain).

This system implements Unsupervised Domain Adaptation (UDA), specifically using the Correlation Alignment (CORAL) Loss, to dynamically realign the decoder's feature space when environmental stress is detected. The result is a BCI system that maintains high decoding accuracy even under severe signal degradation.

This is a mock data demonstration. All raw data is generated synthetically to establish the methodology. Users must integrate real EEG data (e.g., from an MNE-compatible source) to transition to a production-ready model.

🚀 Key Features

EEGNet Baseline: Uses the compact and efficient EEGNet architecture.

Simulated Stress: Functions to mimic severe spaceflight artifacts (EMG, baseline drift).

Unsupervised Adaptation: Implements the CORAL loss to minimize domain discrepancy without requiring labeled stressed data.

Live Resilience Demo: Simulates real-time operation, showing accuracy drop and subsequent adaptation-driven recovery.

🛠️ Setup and Installation

Prerequisites

Git

Python 3.8+

Conda or Miniforge (recommended for environment management)

1. Environment Setup

It is highly recommended to use the provided environment.yml for reproducible results:

# Create the environment using the provided file
conda env create -f environment.yml

# Activate the environment
conda activate neuroadapt-x


If you prefer pip, use the requirements.txt:

pip install -r requirements.txt


2. Project Structure

The core workflow is executed via the sequence of notebooks in the notebooks/ directory.

NeuroAdapt-X/
├── notebooks/
│   ├── 00_environment_setup.ipynb  # Verify dependencies
│   ├── 01_data_preprocessing.ipynb # Load/Mock and process raw data
│   ├── 02_baseline_training.ipynb  # Train EEGNet on clean data (Source)
│   ├── 03_stress_simulation.ipynb  # Test baseline performance on stressed data (Target)
│   ├── 04_adaptation_demo.ipynb    # Offline proof-of-concept for CORAL loss
│   └── 05_live_test.ipynb          # Dynamic, real-time adaptation simulation (The main demo)
├── src/
│   ├── data/
│   │   ├── add_stressors.py        # Logic for simulated noise
│   │   └── preprocess.py           # Data loading and filtering utilities (placeholder)
│   ├── models/
│   │   └── eegnet.py               # The EEGNet model definition
│   └── utils/
│       ├── metrics.py              # Accuracy calculation, etc.
│       └── viz.py                  # Plotting tools
├── models/                         # Trained model checkpoints (`eegnet_baseline.pth`)
├── tests/
│   ├── test_preprocess.py          # Unit tests for data handling
│   ├── test_model.py               # Unit tests for model I/O
│   └── test_stressors.py           # Unit tests for stress simulation
└── ... (other support files)


🏃 Running the Demo (Mock Data)

The entire project is designed as a linear demonstration. You must run the notebooks sequentially.

Launch Jupyter:

jupyter lab


Execute the Notebooks (00 to 05):

00_environment_setup.ipynb: Confirms all Python dependencies are correctly installed.

01_data_preprocessing.ipynb: Generates and prepares the mock EEG data arrays (X and Y).

02_baseline_training.ipynb: Trains the baseline EEGNet decoder on the clean, mock data and saves the checkpoint to models/eegnet_baseline.pth.

03_stress_simulation.ipynb: Loads the baseline model and proves the domain shift by applying simulated stress. Observe the dramatic drop in accuracy.

04_adaptation_demo.ipynb: Implements the CORAL loss adaptation in an offline setting, showing the recovery of accuracy on a static batch of stressed data.

05_live_test.ipynb (The Main Demo): Runs the dynamic simulation loop, where accuracy is tracked as the data shifts between clean and stressed. The resulting plot visually demonstrates the dynamic accuracy drop, followed by rapid recovery due to on-the-fly adaptation.

✅ Testing

Unit tests are provided to verify critical components like data shape and stressor behavior.

# Run tests from the project root directory
pytest tests/


📈 Path to Production: Integrating Real Data

To evolve NeuroAdapt-X into a production-ready solution, the mock data generation logic must be replaced with real EEG data loading and MNE-based preprocessing.

Required Code Changes

Update src/data/preprocess.py:

Replace the mock data generation logic (np.random.randn(...)) in the data loading function with code that uses MNE to read your specific EEG file format (e.g., .fif, .edf).

Ensure the output of your custom loading function matches the expected format: (n_epochs, n_channels, n_times) and a corresponding (n_epochs,) label array.

Key Variables to Update: Modify N_CHANNELS, N_TIMES, SFREQ, and N_CLASSES in 01_data_preprocessing.ipynb and all subsequent notebooks to match your real data's configuration.

Refine src/data/add_stressors.py:

The mock stress simulation is currently designed to induce maximum domain shift.

For a production model, you should either:
a) Define real-world noise profiles based on your specific environment (e.g., measured vibration/EMI in a vehicle).
b) Use a clean/baseline dataset and a genuinely stressed/noisy dataset (which acts as the Target Domain for adaptation).

Production Deployment Notes

Online Monitoring: In a deployed system, the stress detection logic in 05_live_test.ipynb (which is currently simplified to if domain_label == "Stressed":) would be replaced by a statistical process control (SPC) mechanism (e.g., Drift Detection Method (DDM) or CUSUM). This mechanism would monitor a moving average of classification confidence or feature distribution distance (e.g., MMD or CORAL distance) and trigger adaptation when a statistical threshold is crossed.

Model Checkpointing: After successful adaptation in a live setting, the adapted model state (adapted_model.state_dict()) should be saved periodically to ensure rapid recovery upon system restart.

By following these steps, you can leverage the proven adaptation methodology of NeuroAdapt-X with your proprietary BCI data.
