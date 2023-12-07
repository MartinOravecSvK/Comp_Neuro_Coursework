import numpy as np
import matplotlib.pyplot as plt
import utils

def find_spike_pairs(spike_times, interval, adjacent_only):
    pairs = []
    if adjacent_only:
        for i in range(len(spike_times) - 1):
            if (spike_times[i + 1] - spike_times[i] == interval):
                # pairs = np.append(pairs, i)
                pairs.append(i)
    else:
        for i in range(len(spike_times) - 1):
            for j in range(interval//2+1):
                if (spike_times[i+j] - spike_times[i] == interval):
                    # pairs = np.append(pairs, i)
                    pairs.append(i)
                    break
                if (i+j >= len(spike_times) - 1):
                    break
    return pairs

def calculate_triggered_stimulus(stimulus, spike_times, intervals, sampling_rate=500, adjacent_only=False):
    triggered_stimuli = {}
    for interval in intervals:
        segments = []
        pairs = find_spike_pairs(spike_times, interval, adjacent_only)
        for pair in pairs:
            index = spike_times[pair] // 2

            if index >= interval * sampling_rate // 1000:
                segment = stimulus[index - (interval * sampling_rate // 1000):index]
                segments.append(segment)
                print(segment)

        triggered_stimuli[interval] = np.mean(segments, axis=0) if segments else None
        break

    return triggered_stimuli

def question5():
    stimulus = np.genfromtxt('ExtendedCoursework/stim.dat')
    spike_times = utils.load_rho()

    # Define intervals in ms
    intervals = [2, 10, 20, 50]

    # Calculate the average stimulus for each interval (not necessarily adjacent spikes)
    triggered_stimuli = calculate_triggered_stimulus(stimulus, spike_times, intervals)

    # Repeat for adjacent spikes only
    triggered_stimuli_adjacent = calculate_triggered_stimulus(stimulus, spike_times, intervals, adjacent_only=True)

    # Display the results
    print("Calculated average stimmulus for each interval (not necessarily adjacent spikes):\n", triggered_stimuli)
    print("\n")
    print("Calculated average stimmulus for each interval\n", triggered_stimuli_adjacent)

if __name__ == "__main__":
    question5()