"""
While technically working this class has not been used in the final implementation
of the project. The reason for this is that I ran out of time and had to focus on
finishin the project and just used the implementation in the q3q6.py file. 
"""

class IntegrateAndFireNeuron:
    def __init__(self, 
                 threshold=-55.0, 
                 tau=10.0, 
                 R=1.0, 
                 E=-70.0, 
                 absolut_refractory_period=1.0, 
                 relative_refractory_period=4.0, 
                 reset_voltage=-80.0):
        
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

    # Step function
    # Update the membrane potential of the simulated neuron
    def step(self, RI, dt):
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

