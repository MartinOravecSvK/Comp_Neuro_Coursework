import numpy as np
import utils

"""
Question 1: Fano Factor and Coefficient of Variation

    Code to generate spike trains with and without a refractory period 
    and calculate the Fano factor and coefficient of variation of the interspike intervals (ISI)
"""

# Parameters:
#   firing_rate: The average firing rate of the neuron in Hz 
#   interval: The duration of the spike train in seconds
#   refractory_period: The refractory period of the neuron in seconds (Default: 0)
# Returns:
#   spike_times: An array of spike times in seconds

def generate_spike_train(firing_rate: float, interval: float, refractory_period: float = 0):
    current_time = 0
    spike_times = []

    while current_time < interval:
        if refractory_period > 0 and spike_times:
            current_time += refractory_period

        next_spike = np.random.exponential(1 / firing_rate)
        current_time += next_spike

        if current_time < interval:
            spike_times.append(current_time)

    return np.array(spike_times)


# Main function to answer question 1
# - Generates spike trains with and without a refractory period 
#   and calculates the Fano factor and coefficient of variation of the ISI
# - Displays the results
def question1():
    firing_rate = 35  # Hz
    duration = 1000  # seconds
    window_sizes = [0.01, 0.05, 0.1]  # seconds

    # Generate spike trains
    spike_train_no_refractory = generate_spike_train(firing_rate, duration)
    spike_train_with_refractory = generate_spike_train(firing_rate, duration, 0.005) 

    # Calculate Fano factors and coefficients of variation
    fano_no_refractory = utils.calculate_fano_factor(spike_train_no_refractory, window_sizes, duration)
    fano_with_refractory = utils.calculate_fano_factor(spike_train_with_refractory, window_sizes, duration)
    cv_no_refractory = utils.calculate_coefficient_of_variation(spike_train_no_refractory)
    cv_with_refractory = utils.calculate_coefficient_of_variation(spike_train_with_refractory)

    # Display the results
    print("No Refractory Period:")
    print("Fano Factors:", fano_no_refractory)
    print("Coefficient of Variation (CV) of the ISI:", cv_no_refractory)

    print("\nWith 5 ms Refractory Period:")
    print("Fano Factors:", fano_with_refractory)
    print("Coefficient of Variation (CV) of the ISI:", cv_with_refractory)

if __name__ == "__main__":
    question1()