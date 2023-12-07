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
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_vector, sta)
    # plt.xlabel('Time before spike (ms)')
    # plt.ylabel('Stimulus')
    # plt.title('Spike-Triggered Average over 100 ms')
    # plt.grid(True)
    # plt.show()
    # plt.figure(figsize=(12, 6))

    # plt.figure(figsize=(10, 6))
    # plt.plot(time_vector, sta, color='royalblue', linestyle='-', linewidth=2, marker='o', markersize=4)
    # plt.xlabel('Time before spike (ms)', fontsize=12)
    # plt.ylabel('Stimulus', fontsize=12)
    # plt.title('Spike-Triggered Average over 100 ms', fontsize=14)
    # plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
    # plt.tight_layout()
    # plt.savefig('STA_plot.png', dpi=300)  # Save the figure in high-resolution
    # plt.show()

    plt.figure(figsize=(10, 6))

    # Plot the STA curve
    plt.plot(time_vector, sta, color='royalblue', linestyle='-', linewidth=2)

    # Indicate the maximum value on the curve
    max_sta = np.max(sta)
    max_time = time_vector[np.argmax(sta)]
    plt.plot(max_time, max_sta, 'ro')  # 'ro' plots a red circle at the max point
    plt.annotate(f'Maximum ({max_time} ms, {max_sta:.2f})', xy=(max_time, max_sta),
                 xytext=(max_time+5, max_sta), arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, ha='center')

    # Set the x-axis ticks
    plt.xticks(np.arange(-100, 1, 20))

    # Enhanced text for titles and labels
    plt.xlabel('Time before spike (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('Stimulus', fontsize=12, fontweight='bold')
    plt.title('Neural Response Characterization of Fly H1 Neuron', fontsize=14, fontweight='bold')

    # Additional styling
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig('/mnt/data/Enhanced_STA_plot.png', dpi=300)  # Save the figure in high-resolution
    plt.show()


if __name__ == "__main__":
    question4()