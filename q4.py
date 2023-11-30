import numpy as np
import matplotlib.pyplot as plt
import utils

def question4():
    spike_times = utils.load_rho()
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


if __name__ == "__main__":
    question4()