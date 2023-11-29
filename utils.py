import numpy as np

def calculate_fano_factor(spike_times, window_sizes, duration):
    """
    Calculate the Fano factor of the spike count over different window sizes.

    Parameters:
    spike_times (np.array): Times of spikes.
    window_sizes (list): List of window sizes in seconds.
    duration (float): Duration of spike train in seconds.

    Returns:
    dict: Fano factors for each window size.
    """
    fano_factors = {}
    for window in window_sizes:
        window_steps = int(window / 1e-3)  # Convert window size to steps
        num_windows = int(duration / window)
        spike_counts = [np.sum((spike_times >= i * window) & (spike_times < (i + 1) * window)) for i in range(num_windows)]
        variance = np.var(spike_counts)
        mean = np.mean(spike_counts)
        fano_factors[window] = variance / mean if mean > 0 else float('nan')
    
    return fano_factors

def calculate_coefficient_of_variation(spike_times):
    """
    Calculate the coefficient of variation of the inter-spike intervals.

    Parameters:
    spike_times (np.array): Times of spikes.

    Returns:
    float: Coefficient of variation.
    """
    if len(spike_times) < 2:
        return float('nan')
    
    isi = np.diff(spike_times)
    return np.std(isi) / np.mean(isi)
