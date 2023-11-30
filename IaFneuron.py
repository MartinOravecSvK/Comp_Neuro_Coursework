import numpy as np
import matplotlib.pyplot as plt

class IntegrateAndFireNeuron:
    def __init__(self, threshold=-55.0, tau=10.0, R=1.0, E=-70.0):
        self.threshold = threshold  # Spike threshold
        self.tau = tau  # Membrane time constant
        self.R = R  # Membrane resistance
        self.V = -70.0  # Membrane potential
        self.E = E  # Resting potential
        self.spikeVal = 20.0 # Spike value
        self.resetVal = -70.0 # Reset value

    def step(self, RI, dt):
        """
        Update the membrane potential based on the input current
        and time step size.
        """
        if (self.V == self.spikeVal):
            self.V = self.resetVal

        dV = dt / self.tau * (self.E - self.V + RI)
        self.V += dV

        if self.V >= self.threshold:
            self.V = 20.0  # Set membrane potential to 20 mV
        

def simulate():
    # Simulation parameters
    T = 120  # Total time to simulate (ms)
    dt = 0.1  # Time step (ms)
    time = np.arange(0, T, dt)  # Time array
    RI = 16  # RmIe (mV)

    neuron = IntegrateAndFireNeuron()

    membrane_potentials = []

    for _ in time:
        neuron.step(RI, dt)
        membrane_potentials.append(neuron.V)

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(time, membrane_potentials, label="Membrane Potential")
    plt.ylabel("Membrane Potential (V)")
    plt.xlabel("Time elapsed (ms)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate()

