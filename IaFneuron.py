import numpy as np
import matplotlib.pyplot as plt

class IntegrateAndFireNeuron:
    def __init__(self, threshold=-55.0, tau=10.0, R=1.0, E=-70.0, refractory_period=0.0):
        self.threshold = threshold  # Spike threshold
        self.tau = tau  # Membrane time constant
        self.R = R  # Membrane resistance
        self.V = -70.0  # Membrane potential
        self.E = E  # Resting potential
        self.spikeVal = 20.0 # Spike value
        self.resetVal = -70.0 # Reset value
        self.timeElapsed = 0.0 # Time elapsed

    def step(self, RI, dt):
        """
        Update the membrane potential based on the input current
        and time step size.
        """
        self.timeElapsed += dt

        if (self.V == self.spikeVal):
            self.V = self.resetVal

        dV = dt / self.tau * (self.E - self.V + RI)
        self.V += dV

        if self.V >= self.threshold:
            self.V = 20.0  # Set membrane potential to 20 mV
        
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
    T = 100  # Total time to simulate (ms)
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

def simulatePoisson():
    # TODO:
    # Make these values lists to simulate multiple neuron parameters for analaysis
    
    # Simulation parameters
    T = 100  # Total time to simulate (ms)
    dt = 1.0  # Time step (ms)
    time = np.arange(0, T, dt)  # Time array
    spike_voltage = 20.0 # Spike Voltage (mV)
    spike_threshold = -55.0 # Spike threshold (mV)
    refractory_period = 5.0 # Refractory period (ms)
    E = -70.0 # Resting potential (mV)

    # TODO:
    # Use literature to get some values for these parameters
    # based on different neuron types and brain regions
    inhibitory_firing_rate = 20 # Hz
    excitatory_firing_rate = 20 # Hz
    inhibitory_spike_strength = -3.0 # mV
    excitatory_spike_strength = 3.0 # mV

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

    integrate_fire_neuron = IntegrateAndFireNeuron(
        threshold=spike_threshold,
        E=E,
        refractory_period=refractory_period,
    )

    membrane_potentials = []
    spikes = np.array([])

    for t in time:
        RI = simulated_input[int(t)]
        current_voltage = integrate_fire_neuron.step(RI, dt)
        if (current_voltage == spike_voltage):
            spikes = np.append(spikes, t)
        membrane_potentials.append(integrate_fire_neuron.V)

    print("Spike times (ms):")
    print([a for a in np.round(spikes, 2)])

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

if __name__ == "__main__":
    simulate_constant = False
    simulate_poisson = True
    if simulate_constant: simulateConstant()
    if simulate_poisson: simulatePoisson()