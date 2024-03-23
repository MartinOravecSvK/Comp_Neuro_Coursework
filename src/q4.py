import numpy as np
import matplotlib.pyplot as plt
from utils import utils

def question4(show_graph=True, spike_times = None, save_fig=False):
    if spike_times is None:
        spike_times = utils.load_rho()
    stimulus = np.genfromtxt('ExtendedCoursework/stim.dat')

    sampling_rate = 500  # Hz
    window_size_ms = 100  # ms
    window_size_points = window_size_ms * sampling_rate // 1000  

    sta_segments = []

    # Calculate the STA
    for spike_time in spike_times:
        index = spike_time // 2 
        if index >= window_size_points:
            segment = stimulus[int(index) - window_size_points:int(index)]
            sta_segments.append(segment)

    sta = np.mean(sta_segments, axis=0)

    if show_graph:
        time_vector = np.linspace(-window_size_ms, 0, window_size_points, endpoint=True)

        plt.figure(figsize=(10, 6))
        plt.plot(time_vector, sta, color='royalblue', linestyle='-', linewidth=2, marker='o', markersize=4)

        max_sta = np.max(sta)
        max_time = time_vector[np.argmax(sta)]
        plt.axvline(x=round(max_time, 2), color='red', linestyle='--')

        plt.text(round(max_time, 2)-1, 0, f'Max: {round(max_time, 2)} ms\nSTA : {max_sta:.2f}', 
                color='black', verticalalignment='bottom', horizontalalignment='right')

        plt.xticks(np.arange(-100, 1, 10))

        plt.xlabel('Time before spike (ms)', fontsize=12, fontweight='bold')
        plt.ylabel('STA of Stimulus', fontsize=12, fontweight='bold')
        plt.title('Neural Response Characterization of Fly H1 Neuron', fontsize=14, fontweight='bold')

        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
        if save_fig:
            plt.savefig('STA_plot.png', dpi=300) 
    
    return sta

if __name__ == "__main__":
    show_graph = True
    save_fig = False

    question4(show_graph=show_graph, save_fig=save_fig)