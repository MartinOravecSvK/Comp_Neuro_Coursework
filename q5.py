import numpy as np
import matplotlib.pyplot as plt
import utils
import q4
import seaborn as sns

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
                segment = stimulus[int(index):int(spike_times[pair]//2)]
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
    # print("Calculated average stimmulus for each interval (not necessarily adjacent spike_times):\n", triggered_stimuli)
    # print("\n")
    # print("Calculated average stimmulus for each interval\n", triggered_stimuli_adjacent)

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # for interval in intervals:
    #     plt.plot(np.linspace(-100, 0), triggered_stimuli[interval], label=f'{interval}ms')
    # plt.legend()
    # plt.xlabel('Time before spike (ms)', fontsize=12, fontweight='bold')
    # plt.ylabel('Triggered Stimulus', fontsize=12, fontweight='bold')
    # plt.title('Triggered Stimulus', fontsize=14, fontweight='bold')
    # plt.tight_layout()
    # # plt.savefig('Triggered_Stimulus_plot.png', dpi=300)  # Save the figure in high-resolution
    # plt.show()

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # for interval in intervals:
    #     plt.plot(np.linspace(-100, 0), triggered_stimuli_adjacent[interval], label=f'{interval}ms')
    # plt.legend()
    # plt.xlabel('Time before spike (ms)', fontsize=12, fontweight='bold')
    # plt.ylabel('Triggered Stimulus', fontsize=12, fontweight='bold')
    # plt.title('Triggered Stimulus (Adjacent Spike Times Only)', fontsize=14, fontweight='bold')
    # plt.tight_layout()
    # # plt.savefig('Triggered_Stimulus_plot.png', dpi=300)  # Save the figure in high-resolution
    # plt.show()

    # plt.figure(figsize=(10, 6))

    # colors = ['blue', 'green', 'red', 'purple']  # Different colors for each interval
    # intervals = [2, 10, 20, 50]

    # # Plot for non-adjacent spike times
    # for i, interval in enumerate(intervals):
    #     plt.plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
    #             triggered_stimuli[interval], 
    #             label=f'{interval}ms Non-Adjacent', linestyle='--', color=colors[i])

    # # Plot for adjacent spike times
    # for i, interval in enumerate(intervals):
    #     plt.plot(np.linspace(-100, 0, len(triggered_stimuli_adjacent[interval])), 
    #             triggered_stimuli_adjacent[interval], 
    #             label=f'{interval}ms Adjacent', linestyle='-', color=colors[i])

    # plt.legend()
    # plt.xlabel('Time before spike (ms)', fontsize=12, fontweight='bold')
    # plt.ylabel('Triggered Stimulus', fontsize=12, fontweight='bold')
    # plt.title('Comparison of Triggered Stimulus (Adjacent vs. Non-Adjacent)', fontsize=14, fontweight='bold')
    # plt.tight_layout()
    # # plt.savefig('Combined_Triggered_Stimulus_plot.png', dpi=300)  # Save the figure in high-resolution
    # plt.show()

    sta_single_spike = q4.question4(show_graph=False)

    fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # More appealing color palette
    intervals = [2, 10, 20, 50]

    # Plot for non-adjacent spike times with enhanced aesthetics
    for i, interval in enumerate(intervals):
        axs[0].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                    triggered_stimuli[interval], 
                    label=f'{interval} ms', 
                    color=colors[i],
                    linewidth=2)
    axs[0].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                sta_single_spike, 
                label='single spike', 
                color='black',
                linewidth=2)

    # Plot for adjacent spike times with enhanced aesthetics
    for i, interval in enumerate(intervals):
        axs[1].plot(np.linspace(-100, 0, len(triggered_stimuli_adjacent[interval])), 
                    triggered_stimuli_adjacent[interval], 
                    label=f'{interval} ms', 
                    color=colors[i],
                    linewidth=2)
    axs[1].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                sta_single_spike, 
                label='single spike', 
                color='black',
                linewidth=2)

    # Common x-axis label
    fig.text(0.5, 0.02, 'Time Before Pairs of Spikes (ms)', ha='center', fontsize=16, fontweight='bold')

    # Shared y-axis label
    fig.text(0, 0.5, 'Stimulus Average', va='center', rotation='vertical', fontsize=16, fontweight='bold')

    # Individual subplot titles
    axs[0].set_title('Non-Adjacent Spike Times', fontsize=17, fontweight='bold')
    axs[1].set_title('Adjacent Spike Times', fontsize=17, fontweight='bold')

    # Add grid, legend, and tick parameters to each subplot
    for ax in axs:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, loc='best')
        ax.tick_params(labelsize=12)

    # Common main title
    plt.suptitle('Comparative Triggered Stimulus Analysis', fontsize=18, fontweight='bold', y=0.98)

    # Tight layout often improves the spacing between plot elements
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to prevent overlap of suptitle and x/y labels

    # Optionally, save the figure with a transparent background
    plt.savefig('STA_plot_pairs.png', dpi=300, transparent=True)

    plt.show()

def question5custom(spike_times, constant):
    stimulus = np.genfromtxt('ExtendedCoursework/stim.dat')
    
    # Define intervals in ms
    intervals = [2, 10, 20, 50]

    # Calculate the average stimulus for each interval (not necessarily adjacent spike_times)
    print("Calculating triggered stimuli for non-adjacent spike times...")
    for stimuli in stimulus:
        stimuli = stimuli + constant
        
    triggered_stimuli = calculate_triggered_stimulus(stimulus, spike_times, intervals)

    # Repeat for adjacent spike_times only
    print("Calculating triggered stimuli for adjacent spike times...")
    triggered_stimuli_adjacent = calculate_triggered_stimulus(stimulus, spike_times, intervals, adjacent_only=True)

    sta_single_spike1 = q4.question4(show_graph=False)
    sta_single_spike2 = q4.question4(show_graph=False, spike_times=spike_times[1:])

    print("Plotting...")
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # More appealing color palette
    colors = ['blue', 'orange', 'green', 'red']
    intervals = [2, 10, 20, 50]

    # non-adjacent spike times
    for i, interval in enumerate(intervals):
        if i == 1:
            if triggered_stimuli[interval] is not None:
                axs[0].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                            triggered_stimuli[interval], 
                            label=f'{interval} ms', 
                            color=colors[i],
                            linewidth=2,
                            alpha=0.8)
    axs[0].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                sta_single_spike1, 
                label='single spike', 
                color='black',
                linewidth=2,
                linestyle='--',)
    axs[0].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                sta_single_spike2, 
                label='single spike', 
                color='black',
                linewidth=2,
                linestyle='-',)
    
    # adjacent spike times
    for i, interval in enumerate(intervals):
        if i == 1:
            if triggered_stimuli_adjacent[interval] is not None:
                axs[1].plot(np.linspace(-100, 0, len(triggered_stimuli_adjacent[interval])), 
                            triggered_stimuli_adjacent[interval], 
                            label=f'{interval} ms', 
                            color=colors[i],
                            linewidth=2,
                            alpha=0.8,)
    axs[1].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                sta_single_spike1, 
                label='single spike', 
                color='black',
                linewidth=2,
                linestyle='--',)
    axs[1].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                sta_single_spike2, 
                label='single spike', 
                color='black',
                linewidth=2,
                linestyle='-',)
    
    # reset spike_times
    spike_times = utils.load_rho()

    triggered_stimuli = calculate_triggered_stimulus(stimulus, spike_times, intervals)
    triggered_stimuli_adjacent = calculate_triggered_stimulus(stimulus, spike_times, intervals, adjacent_only=True)

    for i, interval in enumerate(intervals):
        if i == 1:
            if triggered_stimuli[interval] is not None:
                axs[0].plot(np.linspace(-100, 0, len(triggered_stimuli[interval])), 
                            triggered_stimuli[interval], 
                            label=f'{interval} ms', 
                            color=colors[i],
                            linewidth=2,
                            linestyle='--',)

    for i, interval in enumerate(intervals):
        if i == 1:
            if triggered_stimuli_adjacent[interval] is not None:
                axs[1].plot(np.linspace(-100, 0, len(triggered_stimuli_adjacent[interval])), 
                            triggered_stimuli_adjacent[interval], 
                            label=f'{interval} ms', 
                            color=colors[i],
                            linewidth=2,
                            linestyle='--')


    fig.text(0.5, 0.05, 'Time Before Pairs of Spikes (ms)', ha='center', fontsize=16, fontweight='bold')
    fig.text(0, 0.5, 'Stimulus Average', va='center', rotation='vertical', fontsize=16, fontweight='bold')

    axs[0].set_title('Non-Adjacent Spike Times', fontsize=18, fontweight='bold')
    axs[1].set_title('Adjacent Spike Times', fontsize=18, fontweight='bold')

    for ax in axs:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, loc='best')
        ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95], pad=3)

    plt.savefig('STA_plot_pairs.png', dpi=300, transparent=True)

    plt.show()


if __name__ == "__main__":
    question5()