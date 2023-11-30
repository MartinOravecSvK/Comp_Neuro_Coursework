import numpy as np
import matplotlib.pyplot as plt

class HodgkinHuxleyNeuron:
    """ Hodgkin-Huxley Neuron Model """
    def __init__(self):
        # Constants
        self.C_m = 1.0  # membrane capacitance, in uF/cm^2
        self.g_Na = 120.0  # maximum conducances, in mS/cm^2
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0  # Nernst reversal potentials, in mV
        self.E_K = -77.0
        self.E_L = -54.387

        # Initial membrane potential
        self.V_m = -65.0

        # Initial values for m, h, and n
        self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
        self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())
        self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())

    def alpha_m(self):
        """Rate variable alpha for m"""
        return 0.1 * (40 + self.V_m) / (1 - np.exp(-(40 + self.V_m) / 10))

    def beta_m(self):
        """Rate variable beta for m"""
        return 4.0 * np.exp(-(65 + self.V_m) / 18)

    def alpha_h(self):
        """Rate variable alpha for h"""
        return 0.07 * np.exp(-(65 + self.V_m) / 20)

    def beta_h(self):
        """Rate variable beta for h"""
        return 1 / (1 + np.exp(-(35 + self.V_m) / 10))

    def alpha_n(self):
        """Rate variable alpha for n"""
        return 0.01 * (55 + self.V_m) / (1 - np.exp(-(55 + self.V_m) / 10))

    def beta_n(self):
        """Rate variable beta for n"""
        return 0.125 * np.exp(-(65 + self.V_m) / 80)

    def I_Na(self):
        """Membrane current (in uA/cm^2) Sodium"""
        return self.g_Na * self.m**3 * self.h * (self.V_m - self.E_Na)

    def I_K(self):
        """Membrane current (in uA/cm^2) Potassium"""
        return self.g_K * self.n**4 * (self.V_m - self.E_K)

    def I_L(self):
        """Membrane current (in uA/cm^2) Leak"""
        return self.g_L * (self.V_m - self.E_L)

    def I_ion(self, I_ext):
        """Total membrane current (in uA/cm^2)"""
        return I_ext - (self.I_Na() + self.I_K() + self.I_L())

    def step(self, I_ext, dt):
        """Simulate a single time step using Euler's method"""
        self.V_m += dt * self.I_ion(I_ext) / self.C_m

        self.m += dt * (self.alpha_m() * (1.0 - self.m) - self.beta_m() * self.m)
        self.h += dt * (self.alpha_h() * (1.0 - self.h) - self.beta_h() * self.h)
        self.n += dt * (self.alpha_n() * (1.0 - self.n) - self.beta_n() * self.n)

        return self.V_m

def simulate():
    # Simulation parameters
    T = 50.0  # total time to simulate (ms)
    dt = 0.01  # simulation time step (ms)
    time = np.arange(0, T+dt, dt)

    # External current
    I_ext = 10  # external current (uA/cm^2)

    # Initialize neuron
    neuron = HodgkinHuxleyNeuron()

    # Record the membrane potentials
    V_m = []

    for t in time:
        V = neuron.step(I_ext, dt)
        V_m.append(V)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(time, V_m)
    plt.title("Hodgkin-Huxley Neuron")
    plt.ylabel("Membrane Potential (mV)")
    plt.xlabel("Time (ms)")
    plt.show()

if __name__ == "__main__":
    simulate()