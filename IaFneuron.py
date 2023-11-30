import numpy as np
import matplotlib.pyplot as plt

class IntegrateAndFireNeuron:
    def __init__(self, threshold=1.0, tau=10.0, R=1.0):
        self.threshold = threshold  # Spike threshold
        self.tau = tau  # Membrane time constant
        self.R = R  # Membrane resistance
        self.V = 0.0  # Membrane potential

    def step(self, I, dt):
        """
        Update the membrane potential based on the input current
        and time step size.
        """
        dV = dt / self.tau * (-self.V + self.R * I)
        self.V += dV

        spike = 0
        if self.V >= self.threshold:
            spike = 1
            self.V = 0.0  # Reset membrane potential after spike

        return spike

def simulate():
    # Simulation parameters
    T = 100  # Total time to simulate (ms)
    dt = 1.0  # Time step (ms)
    time = np.arange(0, T, dt)  # Time array
    I = 1.5  # Input current (constant)

    # Create a neuron
    neuron = IntegrateAndFireNeuron()

    # Simulation
    membrane_potentials = []
    spikes = []

    for t in time:
        spike = neuron.step(I, dt)
        spikes.append(spike)
        membrane_potentials.append(neuron.V)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, membrane_potentials, label="Membrane Potential")
    plt.ylabel("Membrane Potential (V)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, spikes, label="Spikes", color='r')
    plt.xlabel("Time (ms)")
    plt.ylabel("Spikes")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate()

