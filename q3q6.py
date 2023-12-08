import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Value, Manager, Process
import q5

class IntegrateAndFireNeuron:
    def __init__(self, 
                 threshold=-55.0, 
                 tau=10.0, 
                 R=1.0, 
                 E=-70.0, 
                 absolut_refractory_period=1.0, 
                 relative_refractory_period=4.0, 
                 reset_voltage=-80.0,
                 spikeVol=20.0,):
        self.threshold = threshold  # Spike threshold
        self.tau = tau  # Membrane time constant
        self.R = R  # Membrane resistance
        self.V = -70.0  # Membrane potential
        self.E = E  # Resting potential
        self.spikeVol = spikeVol # Spike value
        self.restVol = -70.0 # Reset value
        self.resetVol = reset_voltage # Reset voltage
        self.timeElapsed = 0.0 # Time elapsed
        self.arf = absolut_refractory_period # Refractory period
        self.rrf =  relative_refractory_period # Refractory period
        self.spikes = []

    def step(self, RI, dt):
        self.timeElapsed += dt

        if (self.V == self.spikeVol):
            self.V = self.restVol

        dV = dt / self.tau * (self.E - self.V + RI)
        self.V += dV

        if self.V >= self.threshold:
            self.V = 20.0  # Set membrane potential to 20 mV
            self.spikes.append(self.timeElapsed)

        return self.V

    # A step fnuction with an absolut and relative refractory period
    def stepRef(self, RI, dt):
        self.timeElapsed += dt

        if (self.spikes and self.timeElapsed - self.spikes[-1] < self.arf):
            return self.V 

        effective_threshold = self.threshold
        if (self.spikes and self.arf <= self.timeElapsed - self.spikes[-1] < self.arf + self.rrf):
            effective_threshold += 20 

        if (self.V == self.spikeVol):
            self.V = self.restVol

        dV = dt / self.tau * (self.E - self.V + RI)
        self.V += dV
        
        if self.V >= effective_threshold:
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

    return len(iaf_neuron.getSpikes()) / (T / 1000), iaf_neuron.getSpikes()

def worker(p_n, p_i, input_neuron_strength, input_exci_neuron_num_range, input_inhi_neuron_num_range, input_neuron_freq, T, dt, shared_i, lock, frequencies_list, fano_factors, coefficient_of_variations):

    # print("Calculating poisson procces input neurons spikes for " + str(len(input_exci_neuron_num_range)) + " neurons")
    input_inhibitory = [poisson_neuron(input_neuron_freq, T-dt) for _ in range(max(input_inhi_neuron_num_range))]
    input_excitatory = [poisson_neuron(input_neuron_freq, T-dt) for _ in range(max(input_exci_neuron_num_range))]
    i = 0
    input_range = input_exci_neuron_num_range if max(input_exci_neuron_num_range) > max(input_inhi_neuron_num_range) else input_inhi_neuron_num_range
    for n in input_range:
        inhibitory_input = input_inhibitory[:min(n, max(input_inhi_neuron_num_range))]
        excitatory_input = input_excitatory[:min(n, max(input_exci_neuron_num_range))]
        with lock:
            shared_i.value += 1
            print(f"    {shared_i.value}/{len(input_exci_neuron_num_range)*p_n}", end='\r')
        
        current_freq, spikes = simulate(
            input_neuron_strength, 
            inhibitory_input, 
            excitatory_input, 
            T, 
            dt,
        )
        frequencies_list[p_i].append(current_freq)
        if i % 8 == 0 and len(spikes) >= 1:
            # fano_factors[p_i].append([n, utils.calculate_fano_factor(np.asarray(spikes), [0.01, 0.05, 0.1], T)[0.1]])
            coefficient_of_variations[p_i].append([n, utils.calculate_coefficient_of_variation(spikes)])
        i += 1        

# Main simualtion function with the same number of excitatory and inhibitory neurons
def fullsim1(T, dt, input_neuron_freq):

    # Parameters
    input_exci_neuron_num_range = np.arange(1, 2001, 1) # nnumber of excitatory neurons simulated by a poisson process
    input_inhi_neuron_num_range = np.arange(1, 2001, 1) # nnumber of inhibitory neurons simulated by a poisson process
    input_neuron_strength_range = np.arange(0.5, 5.1, 0.5) # mV
    input_neuron_freq = 35 # Hz

    full_freq_list = [[] for _ in range(len(input_neuron_strength_range))]
    full_sim_n = 1

    # Proccessing parameters
    manager = Manager()
    frequencies_list = manager.list([manager.list() for _ in range(len(input_neuron_strength_range))])  # Shared list
    fano_factors = manager.list([manager.list() for _ in range(len(input_neuron_strength_range))])
    coefficient_of_variations = manager.list([manager.list() for _ in range(len(input_neuron_strength_range))])
    shared_i = Value('i', 0)  # Shared counter
    lock = manager.Lock()  # Shared lock for the coounter
    start_time = time.time()  # Start time
    processes = []  # List of processes

    for sim_i in range(full_sim_n):
        print("   Simulation " + str(sim_i+1) + " of " + str(full_sim_n))
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
                fano_factors,
                coefficient_of_variations,
            ))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # Initialise full freq list
        if (sim_i == 0):
            for i in range(len(input_neuron_strength_range)):
                full_freq_list[i] = frequencies_list[i][:]
        # If already initialised add the new values to the full freq list
        else:
            for i in range(len(input_neuron_strength_range)):
                for j in range(len(input_exci_neuron_num_range)):
                    full_freq_list[i][j] += frequencies_list[i][j]
        
        # Reset shared variables
        frequencies_list = manager.list([manager.list() for _ in range(len(input_neuron_strength_range))])
        # spikes_list = manager.list([manager.list() for _ in range(len(input_neuron_strength_range))])
        shared_i.value = 0
        processes = []

    # Divide the full freq list by the number of simulations
    for i in range(len(input_neuron_strength_range)):
        for j in range(len(input_exci_neuron_num_range)):
            full_freq_list[i][j] /= full_sim_n

    # degree = 3
    # model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # predictions = []
    # for i in range(len(input_neuron_strength_range)):
    #     model.fit(np.array(input_exci_neuron_num_range).reshape(-1, 1), 
    #               np.array(full_freq_list[i]).reshape(-1, 1))

    #     predictions.append(model.predict(np.array(input_exci_neuron_num_range).reshape(-1, 1)))

    print("Done in time: " + str(round(time.time() - start_time, 2)) + "s or " + str(round((time.time() - start_time) / 60, 2)) + "m")

    # Set plot colors
    colors = sns.cubehelix_palette(len(input_neuron_strength_range), start=0.5, rot=-.75)

    # Plot the results from the frequencies list
    plt.figure(figsize=(12, 4))
    plt.stackplot(np.arange(1, len(full_freq_list[0])+1),
                  full_freq_list, 
                  labels=[str(round(input_neuron_strength_range[i], 1)) + " mV" for i in range(len(input_neuron_strength_range))], 
                  colors=colors,
                #   alpha=0.7,
    )
    # for i in range(len(input_neuron_strength_range)):
    #     plt.plot(input_exci_neuron_num_range, 
    #              predictions[i], 
    #              color=colors[i], 
    #              linestyle='-')

    plt.xlim([10, 2000])

    plt.ylabel("Firing rate frequency (Hz)")
    plt.xlabel("Number of simulated input neurons")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the fano factors and coefficient of variations
    # plt.figure(figsize=(12, 4))
    # for i in range(len(input_neuron_strength_range)):
    #     plt.plot(np.array(fano_factors[i]).reshape(-1, 2)[:, 0],
    #              np.array(fano_factors[i]).reshape(-1, 2)[:, 1],
    #              label=str(round(input_neuron_strength_range[i], 1)) + " mV",
    #              color=colors[i],)
    #     # plt.plot(fano_factors[i])
    # plt.ylabel("Fano Factor")
    # plt.xlabel("Number of simulated input neurons")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Plot the coefficient of variations
    plt.figure(figsize=(12, 4))
    for i in range(len(input_neuron_strength_range)):
        plt.plot(np.array(coefficient_of_variations[i]).reshape(-1, 2)[:, 0],
                 np.array(coefficient_of_variations[i]).reshape(-1, 2)[:, 1],
                 label=str(round(input_neuron_strength_range[i], 1)) + " mV",
                 color=colors[i],)
    plt.ylabel("Coefficient of Variation", fontsize=16, fontweight='bold')
    plt.xlabel("Number of simulated input neurons", fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

# This simulation simulates different inhibitory to excitatory neuron ratios
def fullsim2(T, dt):
    # Parameters
    input_exci_neuron_num_range = np.arange(1001, 1, -1) 
    input_inhi_neuron_num_range = np.arange(1, 1001, 1)
    input_neurons_freq = 35 # Hz
    input_neuron_strength = 5.0 # mV

    frequencies_list = [[] for _ in range(3)]

    # First loop keeps the ratio 1 to 1
    input_inhibitory = [poisson_neuron(input_neurons_freq, T-dt) for _ in range(max(input_inhi_neuron_num_range))]
    input_excitatory = [poisson_neuron(input_neurons_freq, T-dt) for _ in range(max(input_exci_neuron_num_range))]
    i = 0
    input_range = input_exci_neuron_num_range if max(input_exci_neuron_num_range) > max(input_inhi_neuron_num_range) else input_inhi_neuron_num_range
    for n in range(len(input_range)):
        print(f"    {i}/{len(input_exci_neuron_num_range)} with inhibitory input: {input_inhi_neuron_num_range[n]} | excitatory input: {input_exci_neuron_num_range[n]} and latest frequency: {frequencies_list[0][-1] if frequencies_list[0] else  'nan'}", end='\r')
        inhibitory_input = input_inhibitory[:input_inhi_neuron_num_range[n]]
        excitatory_input = input_excitatory[:input_exci_neuron_num_range[n]]
        
        current_freq, spikes = simulate(
            input_neuron_strength, 
            inhibitory_input, 
            excitatory_input, 
            T, 
            dt,
        )
        frequencies_list[0].append(current_freq)
        # if i % 4 == 0 and len(spikes) >= 1:
        #     fano_factors[p_i].append([n, utils.calculate_fano_factor(np.asarray(spikes), [0.01, 0.05, 0.1], T)[0.1]])
        #     coefficient_of_variations[p_i].append(utils.calculate_coefficient_of_variation(spikes) if spikes else 0)
        i += 1  

    # input_exci_neuron_num_range = np.arange(1, 26, 1) 
    # input_inhi_neuron_num_range = np.arange(1, 501, 1)

    # i = 0
    # for n in input_range:
    #     print(f"    {i}/{len(input_exci_neuron_num_range)}", end='\r')
    #     inhibitory_input = input_inhibitory[:min(n, max(input_inhi_neuron_num_range))]
    #     excitatory_input = input_excitatory[:min(n, max(input_exci_neuron_num_range))]
        
    #     current_freq, spikes = simulate(
    #         input_neuron_strength, 
    #         inhibitory_input, 
    #         excitatory_input, 
    #         T, 
    #         dt,
    #     )
    #     frequencies_list[1].append(current_freq)
    #     # if i % 4 == 0 and len(spikes) >= 1:
    #     #     fano_factors[p_i].append([n, utils.calculate_fano_factor(np.asarray(spikes), [0.01, 0.05, 0.1], T)[0.1]])
    #     #     coefficient_of_variations[p_i].append(utils.calculate_coefficient_of_variation(spikes) if spikes else 0)
    #     i += 1 

    # input_exci_neuron_num_range = np.arange(1, 501, 1) 
    # input_inhi_neuron_num_range = np.arange(1, 26, 1)

    # i = 0
    # for n in input_range:
    #     print(f"    {i}/{len(input_exci_neuron_num_range)}", end='\r')
    #     inhibitory_input = input_inhibitory[:min(n, max(input_inhi_neuron_num_range))]
    #     excitatory_input = input_excitatory[:min(n, max(input_exci_neuron_num_range))]
        
    #     current_freq, spikes = simulate(
    #         input_neuron_strength, 
    #         inhibitory_input, 
    #         excitatory_input, 
    #         T, 
    #         dt,
    #     )
    #     frequencies_list[2].append(current_freq)
    #     # if i % 4 == 0 and len(spikes) >= 1:
    #     #     fano_factors[p_i].append([n, utils.calculate_fano_factor(np.asarray(spikes), [0.01, 0.05, 0.1], T)[0.1]])
    #     #     coefficient_of_variations[p_i].append(utils.calculate_coefficient_of_variation(spikes) if spikes else 0)
    #     i += 1 

    plt.figure(figsize=(12, 4))
    plt.plot(input_inhi_neuron_num_range, frequencies_list[0], linestyle='solid', label="1:1")
    # plt.plot(input_range, frequencies_list[1], linestyle='dashed', label="x+1:1")
    # plt.plot(input_range, frequencies_list[2], linestyle='dotted', label="1:x+1")
    plt.ylabel("Firing rate frequency (Hz)")
    plt.xlabel("Number of simulated input neurons")
    plt.tight_layout()
    plt.show()

def simulateQ6(stimulus, T, dt, constant_input, tau=25.0):

    iaf_neuron = IntegrateAndFireNeuron(
        threshold=-45.0,
        tau=tau,
        E=-65.0,
        spikeVol=20.0,
        reset_voltage=-70.0,
        absolut_refractory_period=2.0,
        relative_refractory_period=4.0,
    )
    input_values = stimulus
    stimulus_scalar = 1

    for t in range(len(input_values)):
        print(f"    {t+1}/{T//dt}", end='\r')
        RI = input_values[t]*stimulus_scalar + constant_input
        iaf_neuron.stepRef(RI, dt)

    return len(iaf_neuron.getSpikes()) / (T / 1000), iaf_neuron.getSpikes()

# Fullsim for question 6
def fullsim3():
    
    stimulus = np.genfromtxt('ExtendedCoursework/stim.dat')
    T = len(stimulus) * 2 # ms
    dt = 2 # ms
    constant = 10 # mV

    freq, spike_times =  simulateQ6(stimulus, T, dt, constant)
    print()
    print(len(spike_times))
    q5.question5custom(spike_times, constant)

def fullsim4():

    stimulus = np.genfromtxt('ExtendedCoursework/stim.dat')
    T = len(stimulus) * 2 # ms
    dt = 2 # ms
    constant = 10 # mV
    tau_range = np.arange(1, 21, 1)
    triggered_stimuli = []
    triggered_stimuli_adjacent = []

    for i, tau in enumerate(tau_range):
        print(f"    {tau}/{max(tau_range)}", end='\r')
        _, spike_times =  simulateQ6(stimulus, T, dt, constant, tau)

        intervals = [2, 10, 20, 50]

        # Calculate the average stimulus for each interval (not necessarily adjacent spike_times)
        for stimuli in stimulus:
            stimuli = stimuli + constant
            
        triggered_stimuli.append(q5.calculate_triggered_stimulus(stimulus, spike_times, intervals)[10])
        # print(triggered_stimuli[i])
        # return
        triggered_stimuli_adjacent.append(q5.calculate_triggered_stimulus(stimulus, spike_times, intervals, adjacent_only=True)[10])

    colors = sns.cubehelix_palette(len(tau_range), start=0.5, rot=-.75)
    print()
    print("Plotting...")
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # non-adjacent spike times
    if triggered_stimuli is not None:
        axs[0].stackplot(np.linspace(-100, 0, len(triggered_stimuli[0])), 
                    triggered_stimuli, 
                    labels=[str(tau_range[i]) + " ms" for i in range(len(tau_range))], 
                    colors=colors,
                    # linewidth=2,
                    # alpha=0.8
        )

        # axs[0].set_title("Triggered Stimulus for Non-Adjacent Spike Times")

        axs[0].legend()
        axs[0].set_xlim([-100, 0])
        axs[0].set_xticks(np.arange(-100, 1, 10))
        axs[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # axs[0].tight_layout()

    # adjacent spike times
    if triggered_stimuli_adjacent is not None:
        axs[1].stackplot(np.linspace(-100, 0, len(triggered_stimuli_adjacent[0])), 
                    triggered_stimuli_adjacent, 
                    labels=[str(tau_range[i]) + " ms" for i in range(len(tau_range))], 
                    colors=colors,
                    # linewidth=2,
                    # alpha=0.8
        )
        # axs[1].set_title("Triggered Stimulus for Adjacent Spike Times")

        axs[1].legend()
        axs[1].set_xlim([-100, 0])
        axs[1].set_xticks(np.arange(-100, 1, 10))
        axs[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # axs[1].tight_layout()

    fig.text(0.005, 0.5, 'Stimulus Average', va='center', rotation='vertical', fontsize=16)
    fig.text(0.5, 0.05, 'Time Before Pairs of Spikes with interval 10ms (ms)', ha='center', fontsize=16)

    axs[0].set_title('Non-Adjacent Spike Times', fontsize=18)
    axs[1].set_title('Adjacent Spike Times', fontsize=18)
    for ax in axs:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # ax.legend(loc='center left', bbox_to_anchor=(-0.1, 0.5))
        ax.legend(loc='center left')
        ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95], pad=3)
    # plt.savefig("q6.png")
    plt.show()

if __name__=="__main__":
    # np.random.seed(0)
    T = 10000 # ms
    dt = 1 # ms
    input_neurons_input_exci_neuron_num_rangefreq = 10

    # fullsim1(T, dt, input_neurons_freq)
    # fullsim2(T, dt)
    fullsim3()
    # fullsim4()