import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import threading
import time

class IntegrateAndFireNeuron:
    def __init__(self, threshold=-55.0, tau=10.0, R=1.0, E=-70.0, absolut_refractory_period=1.0, relative_refractory_period=4.0, reset_voltage=-80.0):
        self.threshold = threshold  # Spike threshold
        self.tau = tau  # Membrane time constant
        self.R = R  # Membrane resistance
        self.V = -70.0  # Membrane potential
        self.E = E  # Resting potential
        self.spikeVol = 20.0 # Spike value
        self.restVol = -70.0 # Reset value
        self.resetVol = reset_voltage # Reset voltage
        self.timeElapsed = 0.0 # Time elapsed
        self.arf = absolut_refractory_period # Refractory period
        self.rrf =  relative_refractory_period # Refractory period
        self.spikes = []

    def step(self, RI, dt):
        """
        Update the membrane potential based on the input current
        and time step size.
        """
        if (len(self.spikes) > 0 and self.timeElapsed+dt < self.spikes[-1]+self.arf):
            self.timeElapsed += dt
            if (self.V == self.spikeVol):
                self.V = self.resetVol
            return self.V
        
        elif (len(self.spikes) > 0 and self.timeElapsed+dt < self.spikes[-1]+self.rrf):
            self.timeElapsed += dt
            return self.V
        
        elif (self.V == self.resetVol):
            self.V = self.restVol

        self.timeElapsed += dt

        if (self.V == self.spikeVol):
            self.V = self.restVol

        dV = dt / self.tau * (self.E - self.V + RI)
        self.V += dV

        if self.V >= self.threshold:
            self.V = 20.0  # Set membrane potential to 20 mV
            self.spikes.append(self.timeElapsed)

        return self.V

    def getTimeElapsed(self):
        return self.timeElapsed

def poisson_neuron(firing_rate, duration, refractory_period=0, strength=6.0):
    current_time = 0
    spike_times = []

    while current_time < duration:
        # If a refractory period is set and there was a previous spike, add the refractory period to the current time.
        if refractory_period > 0 and spike_times:
            current_time += refractory_period

        # Draw the next spike time from an exponential distribution
        next_spike = np.random.exponential(1 / firing_rate)
        current_time += next_spike

        # Record the spike time if it's within the duration
        if current_time < duration:
            spike_times.append(current_time)

    return np.array(spike_times)

def simulateConstant():
    # Simulation parameters
    T = 1000  # Total time to simulate (ms)
    dt = 1.0  # Time step (ms)
    time = np.arange(0, T, dt)  # Time array
    RI = 16  # RmIe (mV)
    spike_voltage = 20.0 # Spike Voltage (mV)

    neuron = IntegrateAndFireNeuron()

    membrane_potentials = []
    spikes = np.array([])

    for t in time:
        current_voltage = neuron.step(RI, dt)
        if (current_voltage == spike_voltage):
            spikes = np.append(spikes, t)
        membrane_potentials.append(neuron.V)
    print("Spike times (ms):")
    print([a for a in np.round(spikes, 2)])

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(time, membrane_potentials, label="Membrane Potential")
    plt.ylabel("Membrane Potential (V)")
    plt.xlabel("Time elapsed (ms)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def simulateMultipleOptions(verbose=True, display_graphs=False, T=100, dt=1.0):
    output = ""
    _, axs = plt.subplots(2, 1, figsize=(12, 10))
    time = np.arange(0, T, dt)
    inhibitory_firing_rate_options = [10] # Hz
    excitatory_firing_rate_options = [10] # Hz
    inhibitory_spike_strength_options = [-2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -0.5, -1.0, -1.5] # mV
    excitatory_spike_strength_options = [ 2.0,  2.5,  3.0,  3.5,  4.0,  4.5,  5.0,  0.5,  1.0,  1.5] # mV
    presynaptic_neurons_num_inhib_options = [10, 10, 10, 10, 10, 10, 10, 100, 100, 100] # Number of inhibitory poisson process simulated neurons
    presynaptic_neurons_num_excit_options = [10, 10, 10, 10, 10, 10, 10, 100, 100, 100] # Number of inhibitory poisson process simulated neurons
    multiple_options = max(len(inhibitory_firing_rate_options), len(excitatory_firing_rate_options), len(inhibitory_spike_strength_options), len(excitatory_spike_strength_options), len(presynaptic_neurons_num_inhib_options), len(presynaptic_neurons_num_excit_options))
    
    for option_i in range(multiple_options):
        print("Simulating option " + str(option_i+1) + " of " + str(multiple_options) + "...")
        inhibitory_firing_rate = inhibitory_firing_rate_options[min(option_i, len(inhibitory_firing_rate_options)-1)]
        excitatory_firing_rate = excitatory_firing_rate_options[min(option_i, len(excitatory_firing_rate_options)-1)]
        inhibitory_spike_strength = inhibitory_spike_strength_options[min(option_i, len(inhibitory_spike_strength_options)-1)]
        excitatory_spike_strength = excitatory_spike_strength_options[min(option_i, len(excitatory_spike_strength_options)-1)]
        presynaptic_neurons_num_inhib = presynaptic_neurons_num_inhib_options[min(option_i, len(presynaptic_neurons_num_inhib_options)-1)]
        presynaptic_neurons_num_excit = presynaptic_neurons_num_excit_options[min(option_i, len(presynaptic_neurons_num_excit_options)-1)]

        if (verbose):
            print("Inhibitory firing rate: " + str(inhibitory_firing_rate))
            print("Excitatory firing rate: " + str(excitatory_firing_rate))
            print("Inhibitory spike strength: " + str(inhibitory_spike_strength))
            print("Excitatory spike strength: " + str(excitatory_spike_strength))
            print("Number of inhibitory poisson process simulated neurons: " + str(presynaptic_neurons_num_inhib))
            print("Number of excitatory poisson process simulated neurons: " + str(presynaptic_neurons_num_excit))
            print("\n")

        presynaptic_neurons_inhib = [poisson_neuron(inhibitory_firing_rate, T) for _ in range(presynaptic_neurons_num_inhib)]
        presynaptic_neurons_excit = [poisson_neuron(excitatory_firing_rate, T) for _ in range(presynaptic_neurons_num_excit)]
        simulated_input = np.array([0.0 for _ in range(len(time))])
        for neuron in presynaptic_neurons_inhib:
            for spike in neuron:
                simulated_input[int(spike)] += inhibitory_spike_strength
        for neuron in presynaptic_neurons_excit:
            for spike in neuron:
                simulated_input[int(spike)] += excitatory_spike_strength
        simulated_output_potentials, simulated_output_spikes = simulatePoissonInput(simulated_input, T, dt)

        if (verbose):
            print("Observed firing rate (Hz):")
            print(len(simulated_output_spikes) / T * 1000)
            print("Observed coefficient of variation (CV) of ISI:")
            print(utils.calculate_coefficient_of_variation(simulated_output_spikes))
            print("Observed Fano factor:")
            print(utils.calculate_fano_factor(simulated_output_spikes, [0.01, 0.05, 0.1], T))
            print("Spike times (ms):")
            print([a for a in np.round(simulated_output_spikes, 2)])
            print("\n")

        output += "Inhibitory:                     Excitatory:\n"
        output += "(Hz) firing rate:     " + str(inhibitory_firing_rate) + " "*(5-len(str(inhibitory_firing_rate))) +               "| firing rate:    " + str(excitatory_firing_rate) + "\n"
        output += "(mV) spike strength: " + str(inhibitory_spike_strength) + " "*(6-len(str(inhibitory_spike_strength))) +          "| spike strength: " + str(excitatory_spike_strength) + "\n"
        output += "(N)  PPSim Neurons:   " + str(presynaptic_neurons_num_inhib) + " "*(5-len(str(presynaptic_neurons_num_inhib))) + "| PPSim neurons:  " + str(presynaptic_neurons_num_excit) + "\n"
        output += "Observed:\n"
        output += "firing rate (Hz): " + str(len(simulated_output_spikes) / T * 1000) + "\n"
        output += "coefficient of variation (CV) of ISI: " + str(utils.calculate_coefficient_of_variation(simulated_output_spikes)) + "\n"
        output += "Fano factor: " + str(utils.calculate_fano_factor(simulated_output_spikes, [0.01, 0.05, 0.1], T)) + "\n\n"
        axs[0 if presynaptic_neurons_num_excit == 10 else 1].plot(time, simulated_output_potentials, label=f'Param {excitatory_spike_strength}', color=plt.cm.viridis(option_i / multiple_options))

        if display_graphs:
            # Plotting
            plt.figure(figsize=(12, 4))
            plt.plot(time, simulated_output_potentials, label="Membrane Potential")
            plt.ylabel("Membrane Potential (V)")
            plt.xlabel("Time elapsed (ms)")

            # Add a horizontal line for the stable potential (-70 mV)
            plt.axhline(
                y=-70.0, 
                color='black', 
                linestyle='-', 
                linewidth=1, 
                alpha=0.4, 
                label='Stable Potential (-70 mV)'
            )

            # Add a horizontal dotted line for the threshold (-55 mV)
            plt.axhline(
                y=-55.0, 
                color='black', 
                linestyle='--', 
                linewidth=1, 
                alpha=0.4, 
                label='Threshold (-55 mV)'
            )

            plt.legend()

            plt.tight_layout()
            plt.show()

    print("Done!\n")
    print(output)

    # Display graphs
    axs[0].set_title('10 PPSim Neurons')
    axs[0].set_ylabel('Membrane Potential (V)')
    axs[0].set_xlabel('Time elapsed (ms)')
    axs[0].legend()
    # Add a horizontal line for the stable potential (-70 mV)
    axs[0].axhline(
        y=-70.0, 
        color='black', 
        linestyle='-', 
        linewidth=1, 
        alpha=0.4, 
        label='Stable Potential (-70 mV)'
    )

    # Add a horizontal dotted line for the threshold (-55 mV)
    axs[0].axhline(
        y=-55.0, 
        color='black', 
        linestyle='--', 
        linewidth=1, 
        alpha=0.4, 
        label='Threshold (-55 mV)'
    )
    axs[1].set_title('100 PPSim Neurons')
    axs[1].set_ylabel('Membrane Potential (V)')
    axs[1].set_xlabel('Time elapsed (ms)')
    axs[1].legend()
    # Add a horizontal line for the stable potential (-70 mV)
    axs[1].axhline(
        y=-70.0, 
        color='black', 
        linestyle='-', 
        linewidth=1, 
        alpha=1.0, 
        label='Stable Potential (-70 mV)'
    )

    # Add a horizontal dotted line for the threshold (-55 mV)
    axs[1].axhline(
        y=-55.0, 
        color='black', 
        linestyle='--', 
        linewidth=1, 
        alpha=1.0, 
        label='Threshold (-55 mV)'
    )
    plt.tight_layout()
    plt.show()

def simulateSingleOption(display_graph=True, T=100, dt=1.0):
    # Simulated neuron parameters
    time = np.arange(0, T, dt)  # Time array
    spike_voltage = 20.0 # Spike Voltage (mV)
    spike_threshold = -55.0 # Spike threshold (mV)
    absolut_refractory_period = 1.0 # Refractory period (ms)
    relative_refractory_period = 4.0 # Refractory period (ms)
    E = -70.0 # Resting potential (mV)
    E_refractory = -80.0 # Refractory potential (mV)

    integrate_fire_neuron = IntegrateAndFireNeuron(
        E=E,
        threshold=spike_threshold,
        absolut_refractory_period=absolut_refractory_period,
        relative_refractory_period=relative_refractory_period,
        reset_voltage=E_refractory,
    )

    inhibitory_firing_rate = 10 # Hz
    excitatory_firing_rate = 10 # Hz
    inhibitory_spike_strength = -2.0 # mV
    excitatory_spike_strength =  2.0 # mV
    # Set a randomness seed for reproducibility
    presynaptic_neurons_num_inhib = 10 # Number of inhibitory poisson process simulated neurons
    presynaptic_neurons_num_excit = 10 # Number of inhibitory poisson process simulated neurons    

    presynaptic_neurons_inhib = [poisson_neuron(inhibitory_firing_rate, T) for i in range(presynaptic_neurons_num_inhib)]
    presynaptic_neurons_excit = [poisson_neuron(excitatory_firing_rate, T) for i in range(presynaptic_neurons_num_excit)]
    simulated_input = np.array([0 for i in range(len(time))])
    for neuron in presynaptic_neurons_inhib:
        for spike in neuron:
            simulated_input[int(spike)] += inhibitory_spike_strength
    for neuron in presynaptic_neurons_excit:
        for spike in neuron:
            simulated_input[int(spike)] += excitatory_spike_strength

    membrane_potentials = []
    spikes = np.array([])

    for t in time:
        RI = simulated_input[int(t)]
        current_voltage = integrate_fire_neuron.step(RI, dt)
        if (current_voltage == spike_voltage):
            spikes = np.append(spikes, t)
        membrane_potentials.append(integrate_fire_neuron.V)

    # Print the results
    print("Observed firing rate (Hz):")
    print(len(spikes) / T * 1000)
    print("Observed coefficient of variation (CV) of ISI:")
    print(utils.calculate_coefficient_of_variation(spikes))
    print("Observed Fano factor:")
    print(utils.calculate_fano_factor(spikes, [0.01, 0.05, 0.1], T))
    print("Spike times (ms):")
    print([a for a in np.round(spikes, 2)])

    # Show simulation graph
    if display_graph:

        # Plotting
        plt.figure(figsize=(12, 4))
        plt.plot(time, membrane_potentials, label="Membrane Potential")
        plt.ylabel("Membrane Potential (V)")
        plt.xlabel("Time elapsed (ms)")

        # Add a horizontal line for the stable potential (-70 mV)
        plt.axhline(
            y=E, 
            color='black', 
            linestyle='-', 
            linewidth=1, 
            alpha=0.4, 
            label='Stable Potential (-70 mV)'
        )

        # Add a horizontal dotted line for the threshold (-55 mV)
        plt.axhline(
            y=spike_threshold, 
            color='black', 
            linestyle='--', 
            linewidth=1, 
            alpha=0.4, 
            label='Threshold (-55 mV)'
        )

        plt.legend()

        plt.tight_layout()
        plt.show()

def simulatePoissonInput(simulated_input, T, dt):
    # Simulated neuron parameters
    time = np.arange(0, T, dt)  # Time array
    spike_voltage = 20.0 # Spike Voltage (mV)
    spike_threshold = -55.0 # Spike threshold (mV)
    absolut_refractory_period = 1.0 # Refractory period (ms)
    relative_refractory_period = 4.0 # Refractory period (ms)
    E = -70.0 # Resting potential (mV)
    E_refractory = -80.0 # Refractory potential (mV)

    integrate_fire_neuron = IntegrateAndFireNeuron(
        E=E,
        threshold=spike_threshold,
        absolut_refractory_period=absolut_refractory_period,
        relative_refractory_period=relative_refractory_period,
        reset_voltage=E_refractory,
    )

    membrane_potentials = []
    spikes = np.array([])

    for t in time:
        RI = simulated_input[int(t)]
        current_voltage = integrate_fire_neuron.step(RI, dt)
        if (current_voltage == spike_voltage):
            spikes = np.append(spikes, t)
        membrane_potentials.append(integrate_fire_neuron.V)
    
    return membrane_potentials, spikes

def simulatePoisson(display_graph=True):
    # TODO:
    # Make these values lists to simulate multiple neuron parameters for analaysis

    np.random.seed(2024)
    # Simulated neuron parameters
    T = 1000  # Total simulation time (ms)
    dt = 0.25  # Time step (ms)
    time = np.arange(0, T, dt)  # Time array
    spike_voltage = 20.0 # Spike Voltage (mV)
    spike_threshold = -55.0 # Spike threshold (mV)
    absolut_refractory_period = 1.0 # Refractory period (ms)
    relative_refractory_period = 4.0 # Refractory period (ms)
    E = -70.0 # Resting potential (mV)
    E_refractory = -80.0 # Refractory potential (mV)

    integrate_fire_neuron = IntegrateAndFireNeuron(
        E=E,
        threshold=spike_threshold,
        absolut_refractory_period=absolut_refractory_period,
        relative_refractory_period=relative_refractory_period,
        reset_voltage=E_refractory,
    )

    inhibitory_firing_rate = 10 # Hz
    excitatory_firing_rate = 10 # Hz
    inhibitory_spike_strength = -0.5 # mV
    excitatory_spike_strength =  0.5 # mV
    # Set a randomness seed for reproducibility
    presynaptic_neurons_num_inhib = 100 # Number of inhibitory poisson process simulated neurons
    presynaptic_neurons_num_excit = 100 # Number of inhibitory poisson process simulated neurons    

    presynaptic_neurons_inhib = [poisson_neuron(inhibitory_firing_rate, T) for i in range(presynaptic_neurons_num_inhib)]
    presynaptic_neurons_excit = [poisson_neuron(excitatory_firing_rate, T) for i in range(presynaptic_neurons_num_excit)]
    simulated_input = np.array([0 for i in range(len(time))])
    for neuron in presynaptic_neurons_inhib:
        for spike in neuron:
            simulated_input[int(spike)] += inhibitory_spike_strength
    for neuron in presynaptic_neurons_excit:
        for spike in neuron:
            simulated_input[int(spike)] += excitatory_spike_strength

    membrane_potentials = []
    spikes = np.array([])

    for t in time:
        RI = simulated_input[int(t)]
        current_voltage = integrate_fire_neuron.step(RI, dt)
        if (current_voltage == spike_voltage):
            spikes = np.append(spikes, t)
        membrane_potentials.append(integrate_fire_neuron.V)

    # Print the results
    print("Observed firing rate (Hz):")
    print(len(spikes) / T * 1000)
    print("Observed coefficient of variation (CV) of ISI:")
    print(utils.calculate_coefficient_of_variation(spikes))
    print("Observed Fano factor:")
    print(utils.calculate_fano_factor(spikes, [0.01, 0.05, 0.1], T))
    print("Spike times (ms):")
    print([a for a in np.round(spikes, 2)])

    # Show simulation graph
    if display_graph:

        # Plotting
        plt.figure(figsize=(12, 4))
        plt.plot(time, membrane_potentials, label="Membrane Potential")
        plt.ylabel("Membrane Potential (V)")
        plt.xlabel("Time elapsed (ms)")

        # Add a horizontal line for the stable potential (-70 mV)
        plt.axhline(
            y=E, 
            color='black', 
            linestyle='-', 
            linewidth=1, 
            alpha=0.4, 
            label='Stable Potential (-70 mV)'
        )

        # Add a horizontal dotted line for the threshold (-55 mV)
        plt.axhline(
            y=spike_threshold, 
            color='black', 
            linestyle='--', 
            linewidth=1, 
            alpha=0.4, 
            label='Threshold (-55 mV)'
        )

        plt.legend()

        plt.tight_layout()
        plt.show()

def calculateFrequency(n_neurons, strength, neuron_frequency, T, dt):

    # Simulated neuron parameters
    input_inhibitory = [poisson_neuron(neuron_frequency, T) for _ in range(n_neurons)]
    input_excitatory = [poisson_neuron(neuron_frequency, T) for _ in range(n_neurons)]
    # input_sum = [input_excitatory[i] + input_inhibitory[i] for i in range(n_neurons)]

    time = np.arange(0, T, dt)  # Time array
    neuron = IntegrateAndFireNeuron()
    spikes = 0

    # input_sum = np.array([0.0 for _ in range(len(time))])
    # for t in time:
    #     input_sum[int(t)] = sum([input_excitatory[i][int(t)] + input_inhibitory[i][int(t)] for i in range(n_neurons)])


    input_sum = np.array([0.0 for _ in range(len(time))])

    for i in input_inhibitory:
        for spike in i:
            input_sum[int(spike)] -= strength
    for i in input_excitatory:
        for spike in i:
            input_sum[int(spike)] + strength

    for t_step in range(len(time)):
        RI = input_sum[t_step]
        current_voltage = neuron.step(RI, dt)
        if current_voltage == 20.0:
            spikes += 1

    return spikes / T * 1000 # Hz

def worker(threads, frequencies_list, n_neurons, strength, neuron_frequency, T, dt):
    global shared_i
    global lock
    for n in n_neurons:
        with lock:
            shared_i += 1
            print(f"    {shared_i}/{threads*len(n_neurons)}", end='\r')
        frequency = calculateFrequency(n, strength, neuron_frequency, T, dt)
        print(frequency)
        frequencies_list.append(frequency)

def fullsim():

    T = 100
    dt = 1.0
    # num_of_sim_neurons = np.arange(1, 251, 1)
    num_of_sim_neurons = np.arange(1, 10, 1)
    neuron_frequency = 10
    strengths = np.arange(0.5, 5.5, 0.5)
    num_of_threads = len(strengths)
    threads_data = []
    frequencies_list = [[] for _ in range(num_of_threads)]
    threads = []
    for thread in range(num_of_threads):
        threads_data.append(strengths[thread])

    start_time = time.time()

    # Run the threads
    for thread in range(num_of_threads):
        threads.append(threading.Thread(target=worker, args=(num_of_threads, frequencies_list[thread], num_of_sim_neurons, threads_data[thread], neuron_frequency, T, dt)))
        threads[thread].start()

    # Wait for the threads to finish
    for thread in range(num_of_threads):
        threads[thread].join()

    print("Done in time: " + str(round(time.time() - start_time, 2)) + "s")

    # Plot the results from the frequencies list
    plt.figure(figsize=(12, 4))
    for thread in range(num_of_threads):
        plt.plot(num_of_sim_neurons, frequencies_list[thread], label="Strength: " + str(threads_data[thread]))
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Number of neurons")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    shared_i = 0
    lock = threading.Lock()
    
    # np.random.seed(2024)

    full_sim = True
    simulate_constant = False
    simulate_poisson = False
    display_graph = False

    if simulate_constant: simulateConstant()
    if simulate_poisson: simulatePoisson(display_graph=display_graph)

    T = 100  # Total simulation time (ms)
    dt = 0.1  # Time step (ms)

    run_multiple_simulations = False
    run_single_simulation = False

    if run_multiple_simulations: simulateMultipleOptions(verbose=False, display_graphs=display_graph, T=T, dt=dt)
    if run_single_simulation : simulateSingleOption(display_graph=display_graph, T=T, dt=dt)

    if full_sim: fullsim()
