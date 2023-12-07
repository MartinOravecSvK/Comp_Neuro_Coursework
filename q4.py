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

    # Plot the STA curve
    plt.plot(time_vector, sta, color='royalblue', linestyle='-', linewidth=2, marker='o', markersize=4)

    # Find and indicate the maximum value on the curve
    max_sta = np.max(sta)
    max_time = time_vector[np.argmax(sta)]
    max_time_rounded = round(max_time, 2)  # Round the time to two decimal places
    plt.axvline(x=max_time_rounded, color='red', linestyle='--')  # Vertical line for the maximum

    # Annotate the maximum value on the curve
    plt.text(max_time_rounded-1, 0, f'Max: {max_time_rounded} ms\nSTA : {max_sta:.2f}', 
             color='black', verticalalignment='bottom', horizontalalignment='right')

    # Set the x-axis ticks
    plt.xticks(np.arange(-100, 1, 10))

    # Enhanced text for titles and labels
    plt.xlabel('Time before spike (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('STA of Stimulus', fontsize=12, fontweight='bold')
    plt.title('Neural Response Characterization of Fly H1 Neuron', fontsize=14, fontweight='bold')

    # Additional styling
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig('STA_plot.png', dpi=300)  # Save the figure in high-resolution
    plt.show()


if __name__ == "__main__":
    question4()