import numpy as np
import matplotlib.pyplot as plt
import utils

def find_spike_pairs(spike_times, interval, adjacent_only):
    pairs = []
    if adjacent_only:
        for i in range(len(spike_times) - 1):
            if (spike_times[i + 1] - spike_times[i] == interval):
                pairs.append(i)
    else:
        for i in range(len(spike_times) - 1):
            for j in range(len(spike_times) - i - 1):
                if (spike_times[i+j] - spike_times[i] == interval):
                    pairs.append(i)
                    break
                if (i+j >= len(spike_times) or spike_times[i+j] - spike_times[i] > interval):
                    break
    return pairs

def calculate_triggered_stimulus(stimulus, spike_times, intervals, adjacent_only=False):
    triggered_stimuli = {}
    for interval in intervals:
        segments = []
        segment_length = 100
        pairs = find_spike_pairs(spike_times, interval, adjacent_only)
        for pair in pairs:
            index = spike_times[pair] - segment_length
            index = index // 2
            if index >= 0 and index < len(stimulus):
                segment = stimulus[index:spike_times[pair]//2]
                segments.append(segment)

        triggered_stimuli[interval] = np.mean(segments, axis=0) if segments else None

    return triggered_stimuli

def question5():
    stimulus = np.genfromtxt('ExtendedCoursework/stim.dat')
    spike_times = utils.load_rho()

    # Define intervals in ms
    intervals = [2, 10, 20, 50]

    # Calculate the average stimulus for each interval (not necessarily adjacent spike_times)
    triggered_stimuli = calculate_triggered_stimulus(stimulus, spike_times, intervals)

    # Repeat for adjacent spike_times only
    triggered_stimuli_adjacent = calculate_triggered_stimulus(stimulus, spike_times, intervals, adjacent_only=True)

    # Display the results
    print("Calculated average stimmulus for each interval (not necessarily adjacent spike_times):\n", triggered_stimuli)
    print("\n")
    print("Calculated average stimmulus for each interval\n", triggered_stimuli_adjacent)

    # Plotting
    plt.figure(figsize=(10, 6))
    for interval in intervals:
        plt.plot(np.linspace(-100, 0), triggered_stimuli[interval], label=f'{interval}ms')
    plt.legend()
    plt.xlabel('Time before spike (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('Triggered Stimulus', fontsize=12, fontweight='bold')
    plt.title('Triggered Stimulus', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # plt.savefig('Triggered_Stimulus_plot.png', dpi=300)  # Save the figure in high-resolution
    plt.show()

if __name__ == "__main__":
    question5()