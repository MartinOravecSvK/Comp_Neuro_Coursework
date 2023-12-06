import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import time
from multiprocessing import Value, Manager, Process

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

    def step(self, RI, dt):
        """
        Update the membrane potential based on the input current
        and time step size.
        """
        # if (len(self.spikes) > 0 and self.timeElapsed+dt < self.spikes[-1]+self.arf):
        #     self.timeElapsed += dt
        #     if (self.V == self.spikeVol):
        #         self.V = self.resetVol
        #     return self.V
        
        # elif (len(self.spikes) > 0 and self.timeElapsed+dt < self.spikes[-1]+self.rrf):
        #     self.timeElapsed += dt
        #     dV = dt / self.tau * (self.E - self.V + RI*0.5)
        #     self.V += dV

        #     if self.V >= self.threshold:
        #         self.V = 20.0  # Set membrane potential to 20 mV
        #         self.spikes.append(self.timeElapsed)

        #     return self.V
        
        # elif (self.V == self.resetVol and self.V < self.restVol):
        #     self.V = self.restVol

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
    
    def getSpikes(self):
        return self.spikes

def poisson_neuron(firing_rate, duration, refractory_period=0):
    current_time = 0
    spike_times = []
    adjusted_firing_rate = 1000 / firing_rate # 1000 ms in 1 second / firing rate in Hz

    while current_time < duration:
        # If a refractory period is set and there was a previous spike, add the refractory period to the current time.
        if refractory_period > 0 and spike_times:
            current_time += refractory_period

        # Draw the next spike time from an exponential distribution
        next_spike = np.around(np.random.exponential(adjusted_firing_rate))
        current_time += next_spike

        # Record the spike time if it's within the duration
        if current_time <= duration:
            spike_times.append(current_time)

    return np.array(spike_times)

def simulate(input_strength, inhi_input, exci_input, T, dt):

    iaf_neuron = IntegrateAndFireNeuron()
    input_values = np.zeros(int(T / dt))
    constant_input = 10

    for i in inhi_input:
        for spike in i:
            input_values[int(spike)] -= input_strength
    for i in exci_input:
        for spike in i:
            input_values[int(spike)] += input_strength

    for t in np.arange(0, T, dt):
        RI = input_values[int(t / dt)] + constant_input
        iaf_neuron.step(RI, dt)

    return len(iaf_neuron.getSpikes()) / (T / 1000)

def worker(p_n, p_i, input_neuron_strength, input_exci_neuron_num_range, input_inhi_neuron_num_range, input_neuron_freq, T, dt, shared_i, lock, frequencies_list):

    print("Calculating poisson procces input neurons spikes for " + str(len(input_exci_neuron_num_range)) + " neurons")
    input_inhibitory = [poisson_neuron(input_neuron_freq, T-dt) for _ in range(max(input_inhi_neuron_num_range))]
    input_excitatory = [poisson_neuron(input_neuron_freq, T-dt) for _ in range(max(input_exci_neuron_num_range))]

    for n in input_exci_neuron_num_range:
        inhibitory_input = input_inhibitory[:n]
        excitatory_input = input_excitatory[:n]
        with lock:
            shared_i.value += 1
            print(f"    {shared_i.value}/{len(input_exci_neuron_num_range)*p_n}", end='\r')
        
        frequencies_list[p_i].append(simulate(
            input_neuron_strength, 
            inhibitory_input, 
            excitatory_input, 
            T, 
            dt,
        ))

# Main simualtion function with the same number of excitatory and inhibitory neurons
def fullsim1(T, dt, input_neuron_freq):

    # Parameters
    input_exci_neuron_num_range = np.arange(10, 2002, 2) # nnumber of excitatory neurons simulated by a poisson process
    input_inhi_neuron_num_range = np.arange(10, 2002, 2) # nnumber of inhibitory neurons simulated by a poisson process
    input_neuron_strength_range = np.arange(0.5, 5.1, 0.5) # mV
    input_neuron_freq = 35 # Hz
    
    # Proccessing parameters
    manager = Manager()
    frequencies_list = manager.list([manager.list() for _ in range(len(input_neuron_strength_range))])  # Shared list
    shared_i = Value('i', 0)  # Shared counter
    lock = manager.Lock()  # Shared lock for the coounter
    start_time = time.time()  # Start time
    processes = []  # List of processes

    # Distribute the work across threads
    for process_i in range(len(input_neuron_strength_range)):
        p = Process(target=worker, args=(
            len(input_neuron_strength_range),
            process_i,
            input_neuron_strength_range[process_i],
            input_exci_neuron_num_range,
            input_inhi_neuron_num_range,
            input_neuron_freq,
            T,
            dt,
            shared_i,
            lock,
            frequencies_list,
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Done in time: " + str(round(time.time() - start_time, 2)) + "s or " + str(round((time.time() - start_time) / 60, 2)) + "m")

    # Set plot colors
    colors = sns.cubehelix_palette(22, start=0.5, rot=-1.50)

    # Plot the results from the frequencies list
    plt.figure(figsize=(12, 4))
    # for process_i in range(len(input_neuron_strength_range)):
    #     plt.plot(input_exci_neuron_num_range, frequencies_list[process_i], label="Strength: " + str(round(input_neuron_strength_range[process_i], 1)) + " mV")
    
    plt.stackplot(input_exci_neuron_num_range, 
                  frequencies_list, 
                  labels=[str(round(input_neuron_strength_range[i], 1)) + " mV" for i in range(len(input_neuron_strength_range))], 
                  colors=colors
    )

    # Define axes limits
    plt.xlim([10, 2000])
    # plt.ylim([0, 100])
    
    # Define axes labels 
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Number of neurons")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    return

if __name__=="__main__":
    # np.random.seed(0)
    T = 1000 # ms
    dt = 1 # ms
    input_neurons_freq = 10



    fullsim1(T, dt, input_neurons_freq)