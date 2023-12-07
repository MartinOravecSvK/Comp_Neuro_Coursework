import numpy as np

def calculate_fano_factor(spike_times, window_sizes, duration):
    fano_factors = {}
    for window in window_sizes:
        num_windows = int(duration / window)
        spike_counts = [np.sum((spike_times >= i * window) & (spike_times < (i + 1) * window)) for i in range(num_windows)]
        variance = np.var(spike_counts)
        mean = np.mean(spike_counts)
        fano_factors[window] = variance / mean if mean > 0 else float('nan')

    return fano_factors

def calculate_coefficient_of_variation(spike_times):
    if len(spike_times) < 2:
        return float('nan')
    
    isi = np.diff(spike_times)
    return np.std(isi) / np.mean(isi)

def load_rho():
    spike_data = np.genfromtxt('ExtendedCoursework/rho.dat')
    spike_times = np.array([i*2 for i in range(len(spike_data)) if spike_data[i] == 1]) 
    return spike_times