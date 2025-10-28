'''
MD.py
refactored molecular dynamics simulation using numba's JIT acceleration

required packages:
    numpy
    matplotlib
    scipy
    numba
'''

######################################################
# IMPORTS
######################################################

import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from numba import njit, float64, int32
from numba.experimental import jitclass

######################################################
# GLOBAL VARIABLES
######################################################

kb = 1.380649e-23               # joule/Kelvin

######################################################
# FUNCTIONS: I/O and plotting
# not JIT-able
######################################################

# input: .txt file with particle positions; dimensionless mass; dimensionless temperature
# output: n x d position array; n x d velocity array with system momentum 0
def initialize_particle_system(file, mass, desired_T):
    start = time.time()
    print("\nInitializing particle system...")
    positions = np.genfromtxt(file, dtype=float)
    n, d = positions.shape
    velocities = np.random.normal(0.0, desired_T, (n, d))
    # m size should be (n,)
    m = np.asarray(mass)
    if m.ndim == 0:
        m = np.full(n, m)
    elif m.ndim == 2:
        # take first column
        m = m[:, 0]
    system_momentum = np.average(velocities, axis=0, weights=m)
    # normalizes system momentum to 0
    velocities -= system_momentum
    print(f"{n} particles initialized in {d} dimensions.")
    end = time.time()
    speed = end - start
    print(f"({speed:.4f} s)\n")
    return positions, velocities

def write_positions(positions_array, species):
    print(f"Writing positions to output file \"positions.txt\":")
    start = time.time()
    n, d, i = positions_array.shape
    with open("positions.txt", "w") as f:
        f.write(f"{n}\n")
        f.write(f"Properties=species:S:1:pos:R:{d}\n")
        for frame in range(i):
            for atom in range(n):
                x, y, z = positions_array[atom, :, frame]
                f.write(f"{species} {x:.6f} {y:.6f} {z:.6f}\n")
    end = time.time()
    speed = end - start
    print(f"File writing complete ({speed:.4f} s)\n")

def write_energies(K_array, U_array, total_E_array):
    print(f"Writing energy values to output file \"energies.csv\":")
    start = time.time()
    i = len(total_E_array)
    with open("energies.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["Kinetic Energy", "Potential Energy", "Total Energy"])
        for frame in range(i):
            w.writerow([f"{K_array[frame]:.6f}", f"{U_array[frame]:.6f}", f"{total_E_array[frame]:.6f}"])
    end = time.time()
    speed = end - start
    print(f"File writing complete ({speed:.4f} s)\n")

def write_temperature_pressure(temperature_array, pressure_array):
    print(f"Writing temperature and pressure values to output file \"temp_pressure.csv\":")
    start = time.time()
    i = len(temperature_array)
    with open("temp_pressure.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["Temperature", "Pressure"])
        for frame in range(i):
            w.writerow([f"{temperature_array[frame]:.6f}", f"{pressure_array[frame]:.6f}"])
    end = time.time()
    speed = end - start
    print(f"File writing complete ({speed:4f} s)\n")

def write_time_averages(k_avgs, u_avgs, t_avgs, p_avgs):
    print(f"Writing time-averaged values to output file \"time_averages.csv\":")
    start = time.time()
    i = len(k_avgs)
    with open("time_averages.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["kinetic energy", "potential energy", "temperature", "pressure"])
        for frame in range(i):
            w.writerow([f"{k_avgs[frame]:.6f}", f"{u_avgs[frame]:.6f}", f"{t_avgs[frame]:.6f}", f"{p_avgs[frame]:.6f}"])
    end = time.time()
    speed = end - start
    print(f"File writing complete ({speed:4f} s)\n")

def write_files(system, species):
    write_positions(system.positions, species)
    write_energies(system.K_array, system.U_array, system.total_energy_array)
    write_temperature_pressure(system.temperature_array, system.pressure_array)

def final_outputs(system, num_steps, step_size, thermo_off_timestep, eqbm_timestep, 
                  species, mass, sigma, epsilon):
    draw_plots(system, step_size)
    write_files(system, species)

    final_avg_temp = system.time_average_temperature[-1]
    final_avg_kelvin = dimensional_temperature(final_avg_temp, epsilon)

    t = make_x_axis(system)
    diff_coeff = system.get_diffusion_coefficient()
    slope, intercept, lin_diff_coeff = linear_diffusion_coefficient(
        system.post_eqbm_frames, system.frames, t, system.MSD_array, system.d)
    dim_diff_coeff = dimensional_diffusion(lin_diff_coeff, system.d, sigma, epsilon, mass)
    print(f"d(MSD)/dt: {slope:.4e}")

    # write time averages iff sim ran long enough to equilibrate
    if num_steps > eqbm_timestep:
        write_time_averages(system.time_average_K, system.time_average_U, \
                            system.time_average_temperature, system.time_average_pressure)
        print(f"Average temperature after equilibration: {final_avg_temp:.4f} ({final_avg_kelvin:.4f} K)")
    
    # calculate diffusion coefficient iff thermostat was on during entire simulation
    if num_steps <= thermo_off_timestep:
        print(f"Single-value diffusion coefficient: {diff_coeff:.4e}")
        print(f"Linear regression diffusion coefficient: {lin_diff_coeff:.4e} ({dim_diff_coeff:.4f})")
        
    final_temp = system.temperature
    final_kelvin = dimensional_temperature(final_temp, epsilon)
    print(f"Final temperature: {final_temp:.4f} ({final_kelvin:.4f} K)")

def plot_E(t, K, U, E, show=False):
    # plot energy on 3 axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, 
                               layout='constrained', 
                               sharex='row')

    # plot kinetic and potential energy
    ax1.plot(t, K, '-g')
    ax1.set(ylabel='Kinetic Energy')
    y1b, y1t = ax1.get_ylim()
    ax2.plot(t, U, '-b')
    ax2.set(ylabel = 'Potential Energy')
    y2b, y2t = ax2.get_ylim()

    # choose the y limits for total energy plot based on the y limits for kinetic and potential plots
    axislimits = (y1b, y1t, y2b, y2t)

    #plot total energy
    ax3.plot(t, E, '-k')
    ax3.set(ylabel = 'Total Energy', ylim=(min(axislimits), max(axislimits)))

    # figure properties
    fig.set_size_inches(6, 8)
    fig.supxlabel('Time')
    fig.suptitle('Change in Energy over Time')
    plt.savefig('energy_plot.png')
    #print("Saved energy plots to \'energy_plot.png\'.")

    if show:
        # show plot
        plt.show()
    else:
        plt.close()

def plot_P(t, P_list, show=False):
    P_array = np.array(P_list)

    if P_array.ndim != 2:
        raise ValueError("P_list must be a 2D array-like")

    # normalize to shape (d, frames)
    if P_array.shape[0] == len(t) and P_array.shape[1] != len(t):
        P_array = P_array.T
    elif P_array.shape[1] == len(t):
        pass
    else:
        raise ValueError(f"Time length {len(t)} does not match either axis of momentum array {P_array.shape}")

    d, nframes = P_array.shape

    fig, axs = plt.subplots(d, 1, layout='constrained', sharex='row')
    axs = np.array(axs, ndmin=1).reshape(-1)

    labels = ['x', 'y', 'z']
    for i in range(d):
        lbl = labels[i] if i < len(labels) else str(i)
        axs[i].plot(t, P_array[i, :], label=f'P_{lbl}')
        axs[i].set(ylabel=f'Momentum in {lbl}', ylim=(-0.01, 0.01))
        axs[i].legend()

    fig.set_size_inches(6, 6)
    fig.supxlabel('Time')
    fig.suptitle('Total Momentum Components Over Time')
    plt.savefig("momentum_plot.png")
    #print("Saved momentum plots to 'momentum_plot.png'.")

    if show:
        plt.show()
    else:
        plt.close()

def plot_T_p(t, T_list, p_list, show=False):
    T_array = np.array(T_list)
    p_array = np.array(p_list)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t, T_array)
    axes[0].set(ylabel = 'Temperature')

    axes[1].plot(t, p_array)
    axes[1].set(ylabel = 'Pressure')

    fig.supxlabel('Time')
    fig.suptitle('Instantaneous Temperature and Pressure')
    plt.savefig('temp_pressure_plots.png')
    #print("Saved temperature and pressure plots to \'temp_pressure_plots.png\'.")

    if show:
        # show plot
        plt.show()
    else:
        plt.close()

def plot_v(velocity_array, temp, time_units, dt, show=False):
    timesteps = int(time_units // dt)
    v_array = np.array(velocity_array[-timesteps:])
    speeds = np.linalg.norm(v_array, axis=2).flatten()

    plt.hist(speeds, bins=200, density=True, alpha=0.7, color='steelblue')
    plt.xlabel("Velocity")
    plt.ylabel("Probability density")
    plt.title("Velocity distribution")

    x = np.linspace(0, speeds.max(), 200)
    plt.plot(x, sp.maxwell.pdf(x, scale=np.sqrt(float(temp))), 'r-', lw=2, label="Maxwell–Boltzmann")
    plt.legend()
    
    plt.savefig('velocity_hist.png')
    #print("Saved velocity histogram to \'velocity_hist.png\'.")

    if show:
        # show plot
        plt.show()
    else:
        plt.close()

def plot_MSD(t, MSD, show=False):
    MSD_array = np.array(MSD)
    fig, axes = plt.subplots(1, 1, figsize=(6, 8))

    axes.plot(t, MSD_array)
    axes.set(ylabel = 'Mean Squared Displacement')

    fig.supxlabel('Time')
    fig.suptitle('Mean Squared Displacement over Time')
    plt.savefig('MSD.png')
    #print("Saved MSD plot to \'MSD.png\'.")

    if show:
        # show plot
        plt.show()
    else:
        plt.close()

def plot_COM(t, COM_list, show=False):
    COM_array = np.array(COM_list)

    d, nframes = COM_array.shape

    fig, axs = plt.subplots(d, 1, layout='constrained', sharex='row')
    axs = np.array(axs, ndmin=1).reshape(-1)

    labels = ['x', 'y', 'z']
    for i in range(d):
        lbl = labels[i] if i < len(labels) else str(i)
        axs[i].plot(t, COM_array[i, :], label=f'P_{lbl}')
        axs[i].set(ylabel=f'Center of Mass in {lbl}')
        axs[i].legend()

    fig.set_size_inches(6, 6)
    fig.supxlabel('Time')
    fig.suptitle('Center of Mass Over Time')
    plt.savefig("COM_plot.png")
    #print("Saved COM plots to 'COM_plot.png'.")

    if show:
        plt.show()
    else:
        plt.close()

def make_x_axis(system):
    return np.arange(0, system.frames, 1)

def draw_plots(system, step_size):
    t = make_x_axis(system)
    plot_E(t, system.K_array, system.U_array, system.total_energy_array)
    plot_P(t, system.momentum_array)
    plot_T_p(t, system.temperature_array, system.pressure_array)
    plot_v(system.velocities, system.temperature, 100, step_size)
    plot_MSD(t, system.MSD_array)
    plot_COM(t, system.COM_array)
    print()

######################################################
# FUNCTIONS: conversions and calculations
# JIT-enabled!
######################################################

@njit
def T_kelvin_to_dimensionless(T_kelvin, epsilon):
    return T_kelvin * kb / epsilon

@njit
def dimensional_pressure(pressure, epsilon, sigma):
    return pressure * epsilon / (sigma ** 3)

@njit
def dimensional_density(n, mass, cell_side_length, sigma):
    density = n * mass / cell_side_length
    return density / sigma ** 3

@njit
def dimensional_temperature(T_dimensionless, epsilon):
    return T_dimensionless * epsilon / kb

@njit
def dimensional_diffusion(diff_coeff, d, sigma, epsilon, mass):
    D_red = diff_coeff / (2.0 * d)
    return D_red * sigma * np.sqrt(epsilon / mass)


# TODO: various other dimensionalizations

@njit
def cutoff_calcs(r_cut):
    # calculate constant cutoff terms for continuous force/continuous energy LJ
    r_cut_inv = 1 / r_cut
    r_cut_6 = r_cut_inv ** 6
    dU_r_cut = 24 * r_cut_inv * (2 * r_cut_6 ** 2 - r_cut_6)
    U_r_cut = 4 * (r_cut_6 ** 2 - r_cut_6)
    return dU_r_cut, U_r_cut

@njit
def linear_diffusion_coefficient(start, end, t, MSD_array, d):
    n = end - start
    if n <= 1:
        return 0.0, 0.0, 0.0  # slope, intercept, D

    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0

    for i in range(start, end):
        x = t[i]
        y = MSD_array[i]
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y

    denom = n * sxx - sx * sx
    if denom == 0.0:
        slope = 0.0
        intercept = sy / n
    else:
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

    D = slope / (2.0 * d)
    return slope, intercept, D


######################################################
# CLASSES: JIT methods
######################################################

spec = [
    # scalars
    ('n', int32),               # number of particles
    ('d', int32),               # number of dimensions
    ('i', int32),               # current timestep
    ('record_stride', int32),   # stride between frames
    ('frame', int32),           # current frame
    ('frames', int32),          # number of frames
    ('r_cut', float64),         # cutoff radius
    ('dU_r_cut', float64),      # cutoff force
    ('U_r_cut', float64),       # cutoff potential
    ('T0', float64),            # initial temperature guess
    ('temperature', float64),   # current temperature
    ('temp_des', float64),      # desired temperature
    ('L', float64),             # box side length
    ('volume', float64),        # simulation volume
    ('K', float64),             # kinetic energy
    ('U', float64),             # potential energy
    ('total_energy', float64),  # total energy
    ('pressure', float64),      # pressure
    ('MSD', float64),           # mean squared displacement
    ('diffusion_coeff', float64), # self-diffusion coefficient
    ('steps', int32),           # total number of steps
    ('dt', float64),            # timestep size
    ('tau_damp', float64),      # thermostat damping (off if 0)
    ('thermo_off', int32),      # when to turn off thermostat
    ('eqbm_step', int32),       # start data collection
    ('post_eqbm_frames', int32),# number of data collection frames

    # initial state
    ('x0', float64[:, :]),      # initial positions (n x d)
    ('v0', float64[:, :]),      # initial velocities (n x d)

    # current state
    ('xi', float64[:, :]),              # current positions (n x d)
    ('vi', float64[:, :]),              # current velocities (n x d)
    ('F', float64[:, :]),               # current forces (n x d)
    ('wrap_count', int32[:, :]),        # PBC wrap count (n x d)
    ('unrolled', float64[:, :]),        # unwrapped positions (n x d)
    ('momentum', float64[:]),           # system momentum (d)
    ('center_of_mass', float64[:]),     # center of mass (d)

    # time-averaged quantities
    ('time_average_K', float64[:]),             # average kinetic energy
    ('time_average_U', float64[:]),             # average potential energy
    ('time_average_temperature', float64[:]),   # average temperature
    ('time_average_pressure', float64[:]),      # average pressure

    # cached vars that rely on large rij calculation
    ('_F_cache', float64[:,:]),
    ('_U_cache', float64),
    ('_virial_cache', float64),
    ('_cache_step', int32),

    # recorded arrays
    ('positions', float64[:, :, :]),    # positions over time (n x d x frames)
    ('velocities', float64[:, :, :]),   # velocities over time (frames x n x d)
    ('wrap_history', int32[:, :, :]),   # history of PBC wraps (n x d x frames)
    ('K_array', float64[:]),            # kinetic energy over time (frames)
    ('U_array', float64[:]),            # potential energy over time (frames)
    ('total_energy_array', float64[:]), # total energy over time (frames)
    ('temperature_array', float64[:]),  # temperature over time (frames)
    ('pressure_array', float64[:]),     # pressure over time (frames)
    ('momentum_array', float64[:, :]),  # momentum over time (d x frames)
    ('MSD_array', float64[:]),          # MSD over time (frames)
    ('COM_array', float64[:, :]),       # center of mass over time (d x frames)
]

@jitclass(spec)
class ParticleSystem:
    def __init__(self, positions, velocities, r_cut, temp_guess, temp_des,
             eqbm_timestep, thermo_off_timestep, cell_side_length, num_steps, step_size,
             record_freq, thermostat_damping):

        self.n, self.d = positions.shape
        self.x0 = positions.copy()
        self.v0 = velocities.copy()
        self.xi = self.x0.copy()
        self.vi = self.v0.copy()
        self.L = float(cell_side_length)

        # simulation scalars
        self.dt = float(step_size)
        self.steps = int(num_steps)
        self.record_stride = int(max(1, record_freq))
        self.frames = int((self.steps + self.record_stride - 1) // self.record_stride)
        self.frame = 0
        self.temp_des = float(temp_des)
        self.thermo_off = int(thermo_off_timestep)
        self.eqbm_step = int(eqbm_timestep)
        self.tau_damp = float(thermostat_damping)
        self.post_eqbm_frames = int(max(1, (self.steps - self.eqbm_step) // self.record_stride))

        # integer wrap counts and unrolled
        self.wrap_count = np.zeros((self.n, self.d), dtype=np.int32)
        self.unrolled = self.xi.copy()

        # recording arrays
        self.positions = np.empty((self.n, self.d, self.frames), dtype=np.float64)
        self.velocities = np.empty((self.frames, self.n, self.d), dtype=np.float64)
        self.wrap_history = np.zeros((self.n, self.d, self.frames), dtype=np.int32)
        self.K_array = np.empty(self.frames, dtype=np.float64)
        self.U_array = np.empty(self.frames, dtype=np.float64)
        self.total_energy_array = np.empty(self.frames, dtype=np.float64)
        self.momentum_array = np.empty((self.d, self.frames), dtype=np.float64)
        self.temperature_array = np.empty(self.frames, dtype=np.float64)
        self.pressure_array = np.empty(self.frames, dtype=np.float64)
        self.MSD_array = np.empty(self.frames, dtype=np.float64)
        self.COM_array = np.empty((self.d, self.frames), dtype=np.float64)
        self.time_average_K = np.empty(self.frames, dtype=np.float64)
        self.time_average_U = np.empty(self.frames, dtype=np.float64)
        self.time_average_temperature = np.empty(self.frames, dtype=np.float64)
        self.time_average_pressure = np.empty(self.frames, dtype=np.float64)

        # system properties and caches
        self.r_cut = float(r_cut)
        self.dU_r_cut, self.U_r_cut = cutoff_calcs(self.r_cut)
        self.T0 = float(temp_guess)
        self.temperature = self.T0

        self._F_cache = np.zeros((self.n, self.d), dtype=np.float64)
        self._U_cache = 0.0
        self._virial_cache = 0.0
        self._cache_step = -1
        self.i = 0

        # initialize forces
        self.compute_forces_and_energy()
        self.F = self._F_cache.copy()

        # other scalar fields
        self.K = 0.0
        self.U = 0.0
        self.total_energy = 0.0
        self.pressure = 0.0
        self.volume = self.L ** self.d
        self.momentum = np.zeros(self.d, dtype=np.float64)
        self.MSD = 0.0
        self.center_of_mass = np.zeros(self.d, dtype=np.float64)
        self.diffusion_coeff = 0.0

    def run_simulation(self):
        for step in range(self.steps):
            self.simulation_step()
            if step % self.record_stride == 0:
                if self.frame < self.frames:
                    self.periodic_system_update()
                    self.record_data()
                    if step > self.eqbm_step:
                        self.time_averages(self.frame,100)
                    self.frame += 1
            self.i += 1
        self.final_calculations()
                        
    def simulation_step(self):
        # choose integrator
        if self.tau_damp != 0 and self.i < self.thermo_off:
            x, v, f = self.nh_step(self.vi, self.xi, self.F, self.dt)
        else:
            x, v, f = self.vv_step(self.vi, self.xi, self.F, self.dt)
        
        # update system
        self.per_step_update(x, v, f)

    def vv_step(self, v, x, f, dt):
        # first velocity half step
        v += 0.5 * dt * f
        # position full step
        x += dt * v
        self.PBC()
        # update force
        self.compute_forces_and_energy()
        f = self._F_cache
        # second velocity half step
        v += 0.5 * dt * f
        return x, v, f

    def nh_step(self, v, x, f, dt):
        # get temperature
        T_inst = self.get_system_temp()
        # first velocity half step 
        z = self.zeta(T_inst, self.temp_des, self.tau_damp)
        v += 0.5 * dt * (f - z * v) 
        # position full step 
        x += dt * v 
        self.PBC()
        # update force
        self.compute_forces_and_energy()
        f = self._F_cache
        # update temperature at t0 + dt with new KE 
        T_inst = self.get_system_temp()
        z = self.zeta(T_inst, self.temp_des, self.tau_damp) 
        # second velocity half step 
        v = (v + 0.5 * dt * f) / (1 + 0.5 * dt * z) 
        return x, v, f

    def per_step_update(self, x, v, f):
        self.xi = x
        self.vi = v
        self.F = f

    def periodic_system_update(self):
        self.momentum = self.get_system_momentum()
        self.temperature = self.get_system_temp()
        self.pressure = self.get_system_pressure()
        self.MSD = self.get_MSD()
        self.center_of_mass = self.get_center_of_mass()

    def record_data(self):
        self.positions[:, :, self.frame] = self.xi
        self.velocities[self.frame, :, :] = self.vi
        self.wrap_history[:, :, self.frame] = self.wrap_count
        self.unrolled = self.xi + self.wrap_count * self.L

        self.compute_forces_and_energy()
        K = self.get_total_K()
        self.K_array[self.frame] = K
        self.U_array[self.frame] = self._U_cache
        self.total_energy_array[self.frame] = K + self._U_cache

        self.momentum_array[:, self.frame] = self.momentum
        self.temperature_array[self.frame] = self.temperature
        self.pressure_array[self.frame] = self.pressure
        self.MSD_array[self.frame] = self.MSD
        self.COM_array[:, self.frame] = self.center_of_mass

    def time_averages(self, frame, interval):
        self.time_average_K[frame] = self.calculate_avg_K(interval)
        self.time_average_U[frame] = self.calculate_avg_U(interval)
        self.time_average_temperature[frame] = self.calculate_avg_temp(interval)
        self.time_average_pressure[frame] = self.calculate_avg_pressure(interval)

    def final_calculations(self):
        self.unrolled = self.xi + self.wrap_count * self.L
        self.diffusion_coeff = self.get_diffusion_coefficient()

    def compute_forces_and_energy(self):
        if self._cache_step != self.i:
            self.forces_and_energy(self.xi)
            self._cache_step = self.i

    def forces_and_energy(self, xi):
        # pairwise displacements
        rij = xi[:, np.newaxis, :] - xi[np.newaxis, :, :]
        # NIC
        rij -= self.L * np.round(rij / self.L)

        r2 = np.sum(rij**2, axis=2)
        r = np.sqrt(r2)
        for k in range(r.shape[0]):
            r[k, k] = np.inf

        inv_r = np.where(r < self.r_cut, 1.0 / r, 0.0)
        r6 = inv_r**6

        # LJ potential
        U_raw = 4 * (r6**2 - r6)
        shift = self.U_r_cut - (r - self.r_cut) * self.dU_r_cut
        U_pairs = np.where(r < self.r_cut, U_raw - shift, 0.0)
        self._U_cache = 0.5 * np.sum(U_pairs)

        # LJ force
        F_mag = np.where(r < self.r_cut,
                        24 * inv_r**2 * (2 * r6**2 - r6),
                        0.0)
        F_vec = (F_mag[..., np.newaxis] * rij) / r[..., np.newaxis]
        self._F_cache = np.sum(F_vec, axis=1)

        # virial pressure
        self._virial_cache = 0.5 * np.sum(np.sum(rij * F_vec, axis=-1))

    def LJ_F(self):
        self.compute_forces_and_energy()
        return self._F_cache

    def LJ_U(self):
        self.compute_forces_and_energy()
        return self._U_cache

    def zeta(self, T_inst, T_des, damping):
        damp_factor = damping ** (-2)
        ratio = T_inst / T_des
        return damp_factor * (ratio - 1)

    def each_K(self):
        return 0.5 * self.vi ** 2
    
    def get_total_K(self):
        return 0.5 * np.sum(self.vi ** 2)

    def get_total_E(self):
        return self.get_total_K() + self.LJ_U()
    
    def get_system_momentum(self):
        # system total momentum (should be ~0)
        return np.sum(self.vi, axis=0)

    def get_center_of_mass(self):
        # center of mass (should be constant)
        com = np.zeros(self.d)
        for j in range(self.d):
            s = 0.0
            for i in range(self.n):
                s += self.xi[i, j]
            com[j] = s / self.n
        return com

    def get_system_temp(self):
        return 2 * self.get_total_K() / (self.d * (self.n-1))

    def get_system_pressure(self):
        self.compute_forces_and_energy()
        P_ideal = (self.n - 1) * self.get_system_temp() / self.volume
        P_virial = self._virial_cache / (3.0 * self.volume)
        return P_ideal + P_virial

    def calculate_avg_K(self, interval):
        if self.dt <= 0.0:
            return 0.0
        last = max(1, int(interval / self.dt))
        end = self.frame
        start = max(0, end - last)
        return np.mean(self.K_array[start:end])

    def calculate_avg_U(self, interval):
        if self.dt <= 0.0:
            return 0.0
        last = max(1, int(interval / self.dt))
        end = self.frame
        start = max(0, end - last)
        return np.mean(self.U_array[start:end])

    def calculate_avg_temp(self, interval):
        if self.dt <= 0.0:
            return 0.0
        last = max(1, int(interval / self.dt))
        end = self.frame
        start = max(0, end - last)
        return np.mean(self.temperature_array[start:end])

    def calculate_avg_pressure(self, interval):
        if self.dt <= 0.0:
            return 0.0
        last = max(1, int(interval / self.dt))
        end = self.frame
        start = max(0, end - last)
        return np.mean(self.pressure_array[start:end])

    def get_MSD(self):
        displacement = self.unrolled - self.x0
        displacement_squared = np.sum(displacement ** 2, axis=1)
        return np.mean(displacement_squared)

    def get_diffusion_coefficient(self):
        t = self.dt * self.i
        divisor = 1 / (2 * self.d * t) # 1/6t
        return divisor * self.MSD

    def PBC(self):
        wrap = np.floor(self.xi / self.L).astype(np.int32)
        # record when PBC is applied
        self.wrap_count += wrap
        # apply PBC
        self.xi = self.xi % self.L

######################################################
# MAIN: run simulation
######################################################

def main():
    start_setup = time.time()

    # particle properties
    species = "Ar"
    mass = 6.634 * 1e-26
    sigma = 3.4
    epsilon = 1.66 * 1e-21

    # simulation properties
    tau_damp = 0.05
    r_cut = 2.5
    cell_side_length = 6.8

    # temperature
    guess_kelvin = 130
    des_kelvin = 100
    temp_guess = T_kelvin_to_dimensionless(guess_kelvin, epsilon)
    temp_des = T_kelvin_to_dimensionless(des_kelvin, epsilon)

    # time scale
    num_steps = 50000
    step_size = 0.002
    record_freq = 10
    time_units = num_steps * step_size
    eqbm_timestep = 10000
    thermo_off_timestep = 0
    # = 0: thermo off (use VV)
    # = num_steps: thermo on the whole time

    end_setup = time.time()
    thinking_time = end_setup - start_setup
    print(f"\nProgram setup took {thinking_time:.4f} s.")

    test_positions, test_velocities = initialize_particle_system("liquid256.txt", 1, temp_des)

    print(f"Setting up particle class...")

    start_system_setup = time.time()
    system = ParticleSystem(test_positions, test_velocities, r_cut, 
                            temp_guess, temp_des, eqbm_timestep, thermo_off_timestep, 
                            cell_side_length, num_steps, step_size, 
                            record_freq, tau_damp)
    end_system_setup = time.time()
    system_setup = end_system_setup - start_system_setup

    print(f"Starting temperature: {temp_guess:.4f} ({guess_kelvin} K)")
    print(f"Desired temperature: {temp_des:.4f} ({des_kelvin} K)")
    print(f"({system_setup:.4f} s)\n")
    print(f"Running simulation for {int(time_units)} units of time...\n")
    
    start_sim = time.time()
    system.run_simulation()
    end_sim = time.time()
    sim_time = end_sim - start_sim
    avg_per_step = sim_time / num_steps

    print("Simulation ran successfully.")
    print(f"({sim_time:.4f} s | average {avg_per_step:.4e} per timestep.)\n")

    # print("Array shapes:")
    # print(f"x: {system.positions.shape}")
    # print(f"K: {system.K_array.shape}")
    # print(f"U: {system.U_array.shape}")
    # print(f"E: {system.total_energy_array.shape}")
    # print(f"T: {system.temperature_array.shape}")
    # print(f"P: {system.pressure_array.shape}")
    # print(f"MSD: {system.MSD_array.shape}")
    # print(f"COM: {system.COM_array.shape}")
    # print()
    
    final_outputs(system, num_steps, step_size, thermo_off_timestep, eqbm_timestep, species, mass, sigma, epsilon)

if __name__ == "__main__":
    main()
