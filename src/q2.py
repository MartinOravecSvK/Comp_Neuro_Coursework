import numpy as np
from utils import utils

"""
Question 2: Fano Factor and Coefficient of Variation of rho.dat (H1 fly neuron data)

    Code to calculate the Fano factor and coefficient of variation of the interspike intervals (ISI)
"""

# Main function to answer question 2
# - Loads the spike train data from rho.dat
# - Calculates the Fano factor and coefficient of variation of the ISI
# - Displays the results
def question2():
    duration = 1000  # 20 minutes in seconds
    window_sizes = [0.01, 0.05, 0.1] # seconds

    raw_data = np.genfromtxt('ExtendedCoursework/rho.dat')
    spike_train =  np.array([i*0.002 for i in range(len(raw_data)) if raw_data[i] == 1])

    # Calculate Fano factors and coefficients of variation
    fano_no_refractory = utils.calculate_fano_factor(spike_train, window_sizes, duration)
    cv_no_refractory = utils.calculate_coefficient_of_variation(spike_train)

    # Display the results
    print("From rho.dat:")
    print("Fano Factors:", fano_no_refractory)
    print("Coefficient of Variation (CV) of ISI:", cv_no_refractory)

if __name__ == "__main__":
    question2()