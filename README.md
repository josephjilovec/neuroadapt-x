# NeuroAdapt-X: Dynamic Resilience for BCI Decoders

## Project Overview

NeuroAdapt-X is a **production-ready** system demonstrating a solution for Domain Shift in Brain-Computer Interfaces (BCIs). BCI decoders, often trained on clean, lab-collected data (the Source Domain), fail catastrophically when deployed in real-world, high-noise environments like spaceflight (the Target Domain).

This system implements **Unsupervised Domain Adaptation (UDA)**, specifically using the **Correlation Alignment (CORAL) Loss**, to dynamically realign the decoder's feature space when environmental stress is detected. The result is a BCI system that maintains high decoding accuracy even under severe signal degradation.

**Production-Ready with Realistic Simulation**: This repository includes a sophisticated realistic EEG signal simulator that generates physiologically accurate motor imagery data with proper spectral characteristics, ERD patterns, and realistic noise. The system is production-ready and can seamlessly transition to real EEG data when available.

## 🚀 Key Features

- **EEGNet Baseline**: Compact and efficient EEGNet architecture for motor imagery classification
- **Realistic Data Simulation**: Physiologically accurate EEG signal generation with:
  - Mu (8-13 Hz) and Beta (13-30 Hz) rhythms
  - Event-Related Desynchronization (ERD) patterns
  - Realistic pink noise and spatial topography
  - Inter-subject variability
- **Simulated Stress**: Functions to mimic severe spaceflight artifacts (EMG, baseline drift)
- **Unsupervised Adaptation**: CORAL loss implementation for domain alignment without labeled target data
- **Adaptive Batch Normalization (AdaBN)**: Dynamic normalization for target domain adaptation
- **Live Resilience Demo**: Real-time operation simulation showing accuracy drop and adaptation-driven recovery
- **Configuration Management**: Centralized config system for easy customization
- **Comprehensive Logging**: Structured logging for monitoring and debugging

## 🛠️ Setup and Installation

### Prerequisites

- Git
- Python 3.8+
- Conda or Miniforge (recommended for environment management)

### 1. Environment Setup

**Option A: Using Conda (Recommended)**

```bash
# Create the environment using the provided file
conda env create -f environment.yml

# Activate the environment
conda activate neuroadapt-x
```

**Option B: Using pip**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Project Structure

```
NeuroAdapt-X/
├── src/
│   ├── data/
│   │   ├── realistic_simulator.py  # Advanced EEG signal simulator
│   │   ├── load_datasets.py         # Data loading (MOABB + simulator)
│   │   ├── preprocess.py            # Preprocessing pipeline
│   │   └── add_stressors.py         # Stress injection
│   ├── models/
│   │   ├── eegnet.py                # EEGNet architecture
│   │   ├── adaptive.py              # AdaptiveEEGNet + CORAL + AdaBN
│   │   └── train.py                 # Training functions
│   ├── streaming/
│   │   ├── realtime_processor.py    # Real-time BCI processing
│   │   ├── lsl_stream.py            # Lab Streaming Layer interface
│   │   └── fallback_stream.py      # Mock stream for testing
│   └── utils/
│       ├── metrics.py               # Accuracy tracking
│       ├── viz.py                   # Visualization tools
│       └── logger.py                # Logging configuration
├── notebooks/                       # Interactive notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_stress_simulation.ipynb
│   ├── 04_adaptation_demo.ipynb
│   └── 05_live_test.ipynb
├── scripts/
│   └── run_pipeline.py             # Complete pipeline script
├── tests/                           # Unit tests
├── config.py                        # Configuration management
├── requirements.txt                 # Python dependencies
└── setup.py                        # Package setup
```

## 🏃 Running the Demo

### Option 1: Quick Pipeline Script (Recommended)

Run the complete pipeline with a single command:

```bash
python scripts/run_pipeline.py
```

This will:
1. Generate realistic training data using the advanced simulator
2. Train baseline EEGNet model
3. Test on stressed data (demonstrating domain shift)
4. Perform domain adaptation
5. Evaluate adapted model performance

**Options:**
```bash
# Use custom configuration
python scripts/run_pipeline.py --config config.json

# Skip training (load existing models)
python scripts/run_pipeline.py --skip-training
```

### Option 2: Jupyter Notebooks

The entire project can also be run via notebooks for interactive exploration:

```bash
# Launch Jupyter
jupyter lab
```

Execute the Notebooks sequentially:

- **01_data_exploration.ipynb**: Generates and explores realistic EEG data
- **02_baseline_training.ipynb**: Trains the baseline EEGNet decoder on clean data
- **03_stress_simulation.ipynb**: Tests baseline on stressed data (shows domain shift)
- **04_adaptation_demo.ipynb**: Offline CORAL loss adaptation demonstration
- **05_live_test.ipynb**: Dynamic real-time adaptation simulation (main demo)

## ✅ Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_realistic_simulator.py
```

## 📈 Production Features

### Realistic Data Generation

The repository includes a sophisticated `RealisticEEGSimulator` that generates physiologically accurate EEG signals:

- **Mu (8-13 Hz) and Beta (13-30 Hz) rhythms**: Characteristic motor cortex activity
- **Event-Related Desynchronization (ERD)**: Proper spatial and temporal patterns
  - Left hand MI: ERD in C4 (right hemisphere) and Cz
  - Right hand MI: ERD in C3 (left hemisphere) and Cz
- **Realistic noise**: Pink noise (1/f) + white noise components
- **Inter-subject variability**: Multiple subject profiles for robust training
- **Spatial topography**: Proper C3, Cz, C4 channel characteristics

### Configuration Management

Centralized configuration via `config.py`:
- Model hyperparameters
- Data generation parameters
- Training settings
- Adaptation parameters
- Path management

Load/save configurations:
```python
from config import Config, load_config

# Load default config
config = load_config()

# Load from JSON
config = load_config('my_config.json')

# Save config
config.to_json('saved_config.json')
```

### Logging and Monitoring

Structured logging system:
```python
from src.utils.logger import get_logger

logger = get_logger("my_module")
logger.info("Training started")
logger.error("Error occurred")
```

### Integrating Real Data

To use real EEG data instead of simulation:

1. **Update `src/data/load_datasets.py`**:
   - The `fetch_and_load_raw_data()` function supports both MOABB datasets and realistic simulation
   - For custom data, implement a loader that returns MNE Raw objects
   - Ensure output format: `(n_epochs, n_channels, n_times)` arrays

2. **Update Configuration**:
   - Modify `config.py` or use JSON config files
   - Set `N_CHANNELS`, `N_TIMES`, `SFREQ`, `N_CLASSES` to match your data

3. **Stress Profiles**:
   - Customize `src/data/add_stressors.py` for your specific noise profiles
   - Or use real clean/stressed dataset pairs

### Production Deployment

**Online Monitoring**: The system includes confidence-based adaptation triggers. For production, consider:
- Statistical Process Control (SPC) mechanisms
- Drift Detection Methods (DDM)
- CUSUM monitoring
- Feature distribution distance metrics (MMD, CORAL distance)

**Model Checkpointing**: Automatic checkpointing is implemented. Models are saved to `models/checkpoints/` with best model selection.

**Real-time Processing**: The `src/streaming/realtime_processor.py` module provides:
- LSL stream integration
- Online adaptation
- Sliding window processing
- Confidence monitoring

## 📚 Documentation

- **API Documentation**: See docstrings in source files
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **License**: MIT License (see [LICENSE](LICENSE))

## 🔬 Scientific Background

This implementation is based on:

1. **EEGNet**: Lawhern et al. (2018) - Compact CNN for EEG-based BCIs
2. **CORAL**: Sun et al. (2016) - Correlation Alignment for Domain Adaptation
3. **AdaBN**: Li et al. (2016) - Adaptive Batch Normalization for Domain Adaptation

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNE-Python for EEG processing tools
- MOABB for dataset access
- PyTorch for deep learning framework

---

**By following these guidelines, you can seamlessly transition from simulation to production with real BCI data.**
