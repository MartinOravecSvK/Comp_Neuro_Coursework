import numpy as np
import utils

def generate_spike_train(firing_rate, interval, refractory_period=0):
    current_time = 0#; spike_times = []
    spike_times = []

    while current_time < interval:
        if refractory_period > 0 and spike_times:
            current_time += refractory_period

        next_spike = np.random.exponential(1 / firing_rate)
        current_time += next_spike

        if current_time < interval:
            spike_times.append(current_time)

    return np.array(spike_times)

def question1():
    # Parameters
    firing_rate = 35  # Hz
    duration = 1000  # seconds
    window_sizes = [0.01, 0.05, 0.1]  # 10 ms, 50 ms, 100 ms

    # Generate spike trains
    spike_train_no_refractory = generate_spike_train(firing_rate, duration)
    spike_train_with_refractory = generate_spike_train(firing_rate, duration, 0.005)  # 5 ms refractory period

    # Calculate Fano factors and coefficients of variation
    fano_no_refractory = utils.calculate_fano_factor(spike_train_no_refractory, window_sizes, duration)
    fano_with_refractory = utils.calculate_fano_factor(spike_train_with_refractory, window_sizes, duration)
    cv_no_refractory = utils.calculate_coefficient_of_variation(spike_train_no_refractory)
    cv_with_refractory = utils.calculate_coefficient_of_variation(spike_train_with_refractory)

    # Display the results
    print("No Refractory Period:")
    print("Fano Factors:", fano_no_refractory)
    print("Coefficient of Variation (CV) of ISI:", cv_no_refractory)

    print("\nWith 5 ms Refractory Period:")
    print("Fano Factors:", fano_with_refractory)
    print("Coefficient of Variation (CV) of ISI:", cv_with_refractory)

if __name__ == "__main__":
    question1()