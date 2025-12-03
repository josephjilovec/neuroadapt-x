import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any
import mne
from mne.time_frequency import psd_multitaper
from sklearn.manifold import TSNE
import seaborn as sns

# Configuration for plotting
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def plot_eeg_raw(data: np.ndarray, sfreq: float, ch_names: List[str], title: str = "Raw EEG Data"):
    """
    Plots raw EEG time series data for visual inspection.

    Args:
        data (np.ndarray): EEG data array of shape (n_channels, n_times).
        sfreq (float): Sampling frequency of the data.
        ch_names (List[str]): List of channel names.
        title (str): Title of the plot.
    """
    n_channels, n_times = data.shape
    time = np.arange(n_times) / sfreq

    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True)
    fig.suptitle(title, fontsize=16)

    for i in range(n_channels):
        ax = axes[i] if n_channels > 1 else axes
        ax.plot(time, data[i, :], color='C0')
        ax.set_ylabel(f'{ch_names[i]} (µV)')
        ax.axhline(0, color='gray', linestyle='--')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    if n_channels > 1:
        axes[-1].set_xlabel('Time (s)')
    else:
        axes.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_psd(data: np.ndarray, sfreq: float, ch_names: List[str], fmax: float = 40.0, 
             title: str = "Power Spectral Density (PSD)"):
    """
    Calculates and plots the Power Spectral Density (PSD) for each channel.

    Uses MNE's multitaper method for robust PSD estimation.

    Args:
        data (np.ndarray): EEG data array of shape (n_channels, n_times).
        sfreq (float): Sampling frequency.
        ch_names (List[str]): List of channel names.
        fmax (float): Maximum frequency to plot.
        title (str): Title of the plot.
    """
    n_channels = data.shape[0]
    
    # Create mock MNE Info object for compatibility with MNE functions
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # Create an MNE EpochsArray for PSD calculation
    # MNE expects shape (n_epochs, n_channels, n_times)
    data_3d = data[np.newaxis, :, :]
    epochs = mne.EpochsArray(data_3d * 1e-6, info, tmin=0) # Convert to Volts (MNE standard)

    psds, freqs = psd_multitaper(epochs, fmax=fmax, n_jobs=1, verbose=False)
    
    # psds shape is (n_epochs, n_channels, n_freqs) -> (1, n_channels, n_freqs)
    psds = psds[0] * 1e12 # Convert back to power units (arbitrary scaling for visualization)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=16)

    for i in range(n_channels):
        ax.plot(freqs, psds[i, :], label=ch_names[i])
        
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power ($\mu V^2/Hz$)')
    ax.set_xlim(0, fmax)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_tsne_features(source_features: np.ndarray, target_features: np.ndarray, 
                       title: str = "t-SNE Visualization of Feature Space Alignment"):
    """
    Uses t-SNE to visualize the feature distribution from source and target domains.

    Args:
        source_features (np.ndarray): Features from the source domain (clean data). Shape (N_s, D).
        target_features (np.ndarray): Features from the target domain (stressed data). Shape (N_t, D).
        title (str): Title of the plot.
    """
    
    if source_features.ndim != 2 or target_features.ndim != 2:
        print("Error: Feature inputs must be 2D arrays (N, D).")
        return

    # Combine data and create labels
    all_features = np.vstack([source_features, target_features])
    labels = np.array(['Source (Clean)'] * source_features.shape[0] + 
                      ['Target (Stressed)'] * target_features.shape[0])
    
    # Check minimum samples required for t-SNE
    if all_features.shape[0] < 5:
        print("Not enough samples for t-SNE visualization.")
        return

    print("Computing t-SNE projection...")
    # Reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_features.shape[0] - 1), n_jobs=-1)
    try:
        tsne_results = tsne.fit_transform(all_features)
    except Exception as e:
        print(f"t-SNE fitting failed: {e}. Try reducing the data size or increasing n_iter.")
        return

    df = {
        'tsne-2d-one': tsne_results[:, 0],
        'tsne-2d-two': tsne_results[:, 1],
        'Domain': labels
    }
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue='Domain',
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.6
    )
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

def plot_metrics_history(history: List[Dict[str, float]], 
                         metrics: List[str] = ['accuracy', 'loss'],
                         title: str = "Training and Adaptation Metrics"):
    """
    Plots the history of metrics (e.g., accuracy, loss) recorded over time/epochs.

    Args:
        history (List[Dict[str, float]]): List of dictionaries, where each dict 
                                           contains metrics for a step/epoch.
        metrics (List[str]): Keys from the history dictionaries to plot.
        title (str): Title of the plot.
    """
    if not history:
        print("History list is empty. Nothing to plot.")
        return

    epochs = range(1, len(history) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    for metric in metrics:
        values = [step.get(metric) for step in history if metric in step]
        if values:
            ax.plot(epochs[:len(values)], values, label=metric, marker='o', linestyle='--')
        
    ax.set_xlabel('Step / Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    # --- Example Usage for standalone testing ---
    print("Running visualization script examples...")
    
    SFREQ = 250
    CH_NAMES = ['C3', 'Cz', 'C4']
    N_CHANNELS = len(CH_NAMES)
    N_TIMES = 5 * SFREQ # 5 seconds of data
    
    # 1. Generate Mock Data (Simulated Clean EEG)
    time_series = np.linspace(0, N_TIMES / SFREQ, N_TIMES, endpoint=False)
    data = np.zeros((N_CHANNELS, N_TIMES))
    
    # C3: 10Hz Alpha wave + noise
    data[0, :] = 10 * np.sin(2 * np.pi * 10 * time_series) + np.random.randn(N_TIMES)
    # Cz: 20Hz Beta wave + noise
    data[1, :] = 5 * np.sin(2 * np.pi * 20 * time_series) + np.random.randn(N_TIMES)
    # C4: Pink noise
    data[2, :] = np.random.randn(N_TIMES) * 2

    # Plot Raw EEG
    plot_eeg_raw(data, SFREQ, CH_NAMES, "Mock Clean EEG Data Example")
    
    # Plot PSD
    plot_psd(data, SFREQ, CH_NAMES, title="PSD of Mock Clean EEG Data")
    
    # 2. Generate Mock Features for t-SNE
    D = 64 # Feature dimension
    N_SOURCE = 100
    N_TARGET = 80
    
    # Source features (clustered around 1, 1)
    source_features = np.random.randn(N_SOURCE, D) + 1 
    # Target features (shifted due to stress, clustered around -1, -1)
    target_features = np.random.randn(N_TARGET, D) - 1 
    
    # Plot t-SNE (Will show two distinct clusters)
    plot_tsne_features(source_features, target_features, 
                       "t-SNE Before Adaptation (Source vs. Stressed Target)")

    # 3. Generate Mock Metrics History
    mock_history = [
        {'accuracy': 0.65, 'loss': 0.8},
        {'accuracy': 0.68, 'loss': 0.75},
        {'accuracy': 0.72, 'loss': 0.69},
        {'accuracy': 0.78, 'loss': 0.60},
    ]
    
    # Plot Metrics
    plot_metrics_history(mock_history, metrics=['accuracy'], title="Adaptation Progress (Mock)")
