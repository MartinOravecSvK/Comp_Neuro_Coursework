import numpy as np
import matplotlib.pyplot as plt

def question4():
    # Load the spike data
    spike_data = np.genfromtxt('ExtendedCoursework/rho.dat')
    spike_times = np.array([i*2 for i in range(len(spike_data)) if spike_data[i] == 1])

    # Load the stimulus data
    stimulus = np.genfromtxt('ExtendedCoursework/stim.dat')

    # Parameters
    sampling_rate = 500  # Hz
    window_size_ms = 100  # milliseconds
    window_size_points = window_size_ms * sampling_rate // 1000  # convert ms to data points

    # Initialize an array to store the segments of the stimulus
    sta_segments = []

    # Calculate the STA
    for spike_time in spike_times:
        index = spike_time // 2  # convert ms to data points
        if index >= window_size_points:
            segment = stimulus[index - window_size_points:index]
            sta_segments.append(segment)

    # Calculate the average of the segments
    sta = np.mean(sta_segments, axis=0)

    # Time vector for plotting
    time_vector = np.linspace(-window_size_ms, 0, window_size_points, endpoint=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, sta)
    plt.xlabel('Time before spike (ms)')
    plt.ylabel('Stimulus')
    plt.title('Spike-Triggered Average over 100 ms')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(12, 6))

    spike_rate_segments = []
    for i in range(window_size_points, len(spike_data) - window_size_points):
        segment = spike_data[i - window_size_points:i]
        spike_rate = np.sum(segment) / (window_size_ms / 1000)  # Spikes per second
        spike_rate_segments.append(spike_rate)
    average_spike_rate = np.mean(spike_rate_segments, axis=0)

    # Plot STA
    plt.plot(time_vector, sta, label='Spike-Triggered Average', color='blue')

    # Assuming average_spike_rate needs to be plotted over the same time_vector
    plt.plot(time_vector, [average_spike_rate] * len(time_vector), label='Average Spike Rate', color='red')

    plt.xlabel('Time before spike (ms)')
    plt.ylabel('Stimulus / Spike Rate')
    plt.title('Spike-Triggered Average and Average Spike Rate over 100 ms')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    question4()