import numpy as np
import utils

def question2():
    # Parameters
    firing_rate = 35  # Hz
    duration = 1000  # seconds
    window_sizes = [0.01, 0.05, 0.1]  # 10 ms, 50 ms, 100 ms

    spike_train_no_refractory = np.array()
    spike_train_with_refractory = np.array()

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
