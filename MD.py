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
from numba import njit, float64, int32, int64
from numba.experimental import jitclass

######################################################
# GLOBAL VARIABLES
######################################################

kb = 1.380649e-23               # joule/Kelvin
mm_per_m = 1000
meter_per_angstrom = 1e-10
eV_per_J = 6.242e18

######################################################
# JIT FUNCTIONS: conversions and utilities
######################################################

@njit
def T_kelvin_to_dimensionless(T_kelvin, epsilon):
    return T_kelvin * kb / epsilon

@njit
def dimensional_pressure(pressure, epsilon, sigma):
    sigma_m = sigma * meter_per_angstrom
    return pressure * epsilon / (sigma_m ** 3)

@njit
def dimensional_density(n, mass, cell_side_length, sigma):
    density = n * mass / cell_side_length
    return density / sigma ** 3

@njit
def dimensional_temperature(T_dimensionless, epsilon):
    return T_dimensionless * epsilon / kb

@njit
def dimensional_diffusion(diff_coeff, sigma, epsilon, mass):
    sigma_m = sigma * meter_per_angstrom
    m2_per_sec = diff_coeff * sigma_m * np.sqrt(epsilon / mass)
    return m2_per_sec * (mm_per_m ** 2)

# TODO: various other dimensionalizations

@njit
def cutoff_calcs(r_cut):
    # calculate constant cutoff terms for continuous force/continuous energy LJ
    r_cut_inv = 1 / r_cut
    r_cut_6 = r_cut_inv ** 6
    dU_r_cut = 24 * r_cut_inv * (2 * r_cut_6**2 - r_cut_6)
    U_r_cut = 4 * (r_cut_6 ** 2 - r_cut_6)
    return dU_r_cut, U_r_cut

@njit
def make_x_axis(frames, dt, record_stride):
    t = np.empty(frames, dtype=np.float64)
    step = dt * float(record_stride)
    for k in range(frames):
        t[k] = k * step
    return t


######################################################
# NON-JIT FUNCTIONS: I/O and plotting
######################################################

def initialize_particle_system(file, mass, desired_T):
    # read initial positions from .txt file 
    # generate random velocities
    # normalize system momentum

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
    end = time.time()
    speed = end - start
    print(f"    {n} particles initialized in {d} dimensions. ({speed:.4f} s)\n")
    return positions, velocities

# TODO: add data callout when thermostat turned off
# E, T_p, MSD
def plot_E(t, K, U, E, show=False):
    # plot kinetic, potential, and total energy
    # for debugging: total energy should be constant if not using thermostat


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
    # plot momentum for all dimensions
    # for debugging: P should be zero for every dimension during entire simulation
    P_array = np.array(P_list)

    if P_array.ndim != 2:
        raise ValueError("P_list must be a 2D array-like")

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

def plot_T_p(t, T_list, p_list, T_avg_list, p_avg_list, show=False):
    # plot temperature and pressure

    T_array = np.array(T_list)
    p_array = np.array(p_list)
    T_av_array = np.array(T_avg_list)
    p_av_array = np.array(p_avg_list)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t, T_array)
    axes[0].plot(t, T_av_array)
    axes[0].set(ylabel = 'Temperature')

    axes[1].plot(t, p_array)
    axes[1].plot(t, p_av_array)
    axes[1].set(ylabel = 'Pressure')

    fig.supxlabel('Time')
    plt.savefig('temp_pressure_plots.png')
    #print("Saved temperature and pressure plots to \'temp_pressure_plots.png\'.")

    if show:
        # show plot
        plt.show()
    else:
        plt.close()

def plot_v(velocity_array, temp, time_units, dt, show=False):
    # histogram of velocities with a Maxwell-Boltzmann fit line

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

def plot_MSD(t, MSD, slope, intercept, show=False):
    # plot mean squared displacement of the particle system

    MSD_array = np.array(MSD)

    plt.plot(t, MSD_array, label='Data')

    plt.gca()
    lin_reg = intercept + slope * t
    plt.plot(t, lin_reg, '-r', label='Linear fit')

    plt.xlabel('Time')
    plt.ylabel('MSD')
    plt.title('Mean Squared Displacement over Time')
    plt.savefig('MSD.png')
    #print("Saved MSD plot to \'MSD.png\'.")

    if show:
        # show plot
        plt.show()
    else:
        plt.close()

def plot_COM(t, COM_list, show=False):
    # plot center of mass of the system over time
    # for debugging: COM should be constant throughout simulation

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

def draw_plots(system, t, step_size, MSD_slope, MSD_intercept, show=False):
    # wrapper function: draw all plots
    plot_E(t, system.K_array, system.U_array, system.total_energy_array, show)
    plot_P(t, system.momentum_array, show)
    plot_T_p(t, system.temperature_array, system.pressure_array, system.time_average_temperature, system.time_average_pressure, show)
    plot_v(system.velocities, system.temperature, 100, step_size, show)
    plot_MSD(t, system.MSD_array, MSD_slope, MSD_intercept, show)
    plot_COM(t, system.COM_array, show)

def write_positions(positions_array, sp, cell_side_length, print_time=False):
    # writes a .txt file in extended XYZ format for visualizing the simulation in OVITO
    
    if print_time:
        print('Writing positions to output file "positions.txt":')
    start = time.time()

    n, d, frames = positions_array.shape

    with open("positions.txt", "w") as f:
        for frame in range(frames):
            f.write(f"{n}\n")
            f.write(f"Lattice=\"{cell_side_length:.6f} 0 0 0 {cell_side_length:.6f} 0 0 0 {cell_side_length:.6f}\" Properties=species:S:1:pos:R:{d}\n")
            for atom in range(n):
                x = positions_array[atom, 0, frame]
                y = positions_array[atom, 1, frame] if d > 1 else 0.0
                z = positions_array[atom, 2, frame] if d > 2 else 0.0

                f.write(f"{sp} {x:.6f} {y:.6f} {z:.6f}\n")

    end = time.time()
    if print_time:
        print(f"File writing complete ({end - start:.4f} s)\n")

def write_unrolled_positions(unrolled_array, sp, print_time=False):
    # writes a .txt file in XYZ format with no periodic boundary conditions
    # for debugging

    if print_time:
        print(f"Writing positions to output file \"unrolled.txt\":")
    start = time.time()

    n, d, frames = unrolled_array.shape

    with open("unrolled.txt", "w") as f:
        for frame in range(frames):
            f.write(f"{n}\n")
            f.write(f"Properties=species:S:1:pos:R:{d}\n")
            for atom in range(n):
                x = unrolled_array[atom, 0, frame]
                y = unrolled_array[atom, 1, frame] if d > 1 else 0.0
                z = unrolled_array[atom, 2, frame] if d > 2 else 0.0

                f.write(f"{sp} {x:.6f} {y:.6f} {z:.6f}\n")

    end = time.time()
    speed = end - start
    if print_time:
        print(f"File writing complete ({speed:.4f} s)\n")

def write_energies(K_array, U_array, total_E_array, print_time=False):
    # writes a .csv file with kinetic, potential, and total energy values for every recorded simulation frame
    # energy values are in units of ε

    if print_time:
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
    if print_time:
        print(f"File writing complete ({speed:.4f} s)\n")

def write_temperature_pressure(temperature_array, pressure_array, print_time=False):
    # writes temperature and pressure values, in reduced LJ units, to a .csv file

    if print_time:
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
    if print_time:
        print(f"File writing complete ({speed:4f} s)\n")

def write_time_averages(k_avgs, u_avgs, t_avgs, p_avgs, print_time=False):
    # writes time-averaged potential and kinetic energy, temperature and pressure to a .csv file, in reduced LJ units

    if print_time:
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
    if print_time:
        print(f"File writing complete ({speed:4f} s)\n")

def write_files(system, species, print_time=False):
    # wrapper function to write files
    # only writes time averages if simulation ran long enough to equilibrate
    write_positions(system.positions, species, system.L, print_time)
    write_unrolled_positions(system.unrolled_array, species, print_time)
    write_energies(system.K_array, system.U_array, system.total_energy_array, print_time)
    write_temperature_pressure(system.temperature_array, system.pressure_array, print_time)
    if system.steps > system.eqbm_step:
        write_time_averages(system.time_average_K, system.time_average_U, \
                            system.time_average_temperature, system.time_average_pressure)

def final_outputs(system, num_steps, step_size, 
                  thermo_off_timestep, eqbm_timestep, 
                  species, mass, sigma, epsilon):
    # post-processing for simulation:
    #   writes files
    #   draws plots
    #   prints final temperature, pressure, slope of MSD, and self-diffusion coefficient
    
    start = time.time()

    t = make_x_axis(system.frames, system.dt, system.record_stride)

    # time-averaged values
    final_avg_kinetic = system.final_avg_K
    final_K_dim = final_avg_kinetic * epsilon * eV_per_J
    final_avg_potential = system.final_avg_U
    final_U_dim = final_avg_potential * epsilon * eV_per_J
    final_avg_temp = system.final_avg_temp
    final_avg_kelvin = dimensional_temperature(final_avg_temp, epsilon)
    final_avg_pressure = system.final_avg_pressure
    final_avg_p_dim = dimensional_pressure(final_avg_pressure, epsilon, sigma)

    # instantaneous values
    final_kinetic = system.K
    final_K_dim = final_kinetic * epsilon * eV_per_J
    final_potential = system.U
    final_U_dim = final_potential * epsilon * eV_per_J
    final_temp = system.temperature
    final_kelvin = dimensional_temperature(final_temp, epsilon)
    final_pressure = system.pressure
    final_p_dim = dimensional_pressure(final_pressure, epsilon, sigma)

    diff_coeff = system.diffusion_coeff
    dim_diff_coeff = dimensional_diffusion(diff_coeff, sigma, epsilon, mass)

    draw_plots(system, t, step_size, system.MSD_slope, system.MSD_intercept)
    write_files(system, species)

    end = time.time()
    post_processing = end - start

    print(f"Post processing complete. ({post_processing:.4f} s)")

    print(f"    > MSD slope: {system.MSD_slope:.4e}")
    print(f"    > Diffusion coefficient: {diff_coeff:.4e} ({dim_diff_coeff:.4e} mm²/s)")

    # calculate certain qualities iff thermostat was on during entire simulation
    if num_steps <= thermo_off_timestep:
        # cV
        pass

    # write time averages iff sim ran long enough to equilibrate and thermostat was off during equilibrium
    if num_steps > eqbm_timestep and eqbm_timestep >= thermo_off_timestep:
        print("Time-averaged equilibrium values")
        print(f"    > Average kinetic energy: {final_avg_kinetic:.4f} ({final_K_dim:.4e} eV)")
        print(f"    > Average potential energy: {final_avg_potential:.4f} ({final_U_dim:.4e} eV)")
        print(f"    > Average temperature: {final_avg_temp:.4f} ({final_avg_kelvin:.2f} K)")
        print(f"    > Average pressure: {final_avg_pressure:.4f} ({final_avg_p_dim:.4e} Pa)")
    else:
        print("    Time-averaged values not available; simulation was not run long enough to equilibrate.")
        print(f"    > Final instantaneous kinetic energy: {final_kinetic:.4f} ({final_K_dim:.4e} eV)")
        print(f"    > Final instantaneous pressure: {final_potential:.4f} ({final_U_dim:.4e} eV)")
        print(f"    > Final instantaneous temperature: {final_temp:.4f} ({final_kelvin:.2f} K)")
        print(f"    > Final instantaneous pressure: {final_pressure:.4f} ({final_p_dim:.4e} Pa)")


######################################################
# CLASS
######################################################

spec = [
    # scalars
    ('n', int64),               # number of particles
    ('d', int64),               # number of dimensions
    ('i', int64),               # current timestep
    ('record_stride', int64),   # stride between frames
    ('frame', int64),           # current frame
    ('frames', int64),          # number of frames
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
    ('MSD_slope', float64),     # slope of MSD fit line
    ('MSD_intercept', float64), # intercept of MSD fit line
    ('diffusion_coeff', float64), # self-diffusion coefficient
    ('heat_capacity', float64), # constant-volume heat capacity
    ('steps', int64),           # total number of steps
    ('dt', float64),            # timestep size
    ('tau_damp', float64),      # thermostat damping (off if 0)
    ('zeta', float64),          # dynamic friction variable for thermostat
    ('thermo_off', int64),      # when to turn off thermostat
    ('eqbm_step', int64),       # start data collection
    ('post_eqbm_frames', int64),# number of data collection frames

    # initial state
    ('x0', float64[:, :]),      # initial positions (n x d)
    ('v0', float64[:, :]),      # initial velocities (n x d)

    # current state
    ('xi', float64[:, :]),              # current positions (n x d)
    ('vi', float64[:, :]),              # current velocities (n x d)
    ('F', float64[:, :]),               # current forces (n x d)
    ('wrap_count', int64[:, :]),        # PBC wrap count (n x d)
    ('unrolled', float64[:, :]),        # unwrapped positions (n x d)
    ('momentum', float64[:]),           # system momentum (d)
    ('center_of_mass', float64[:]),     # center of mass (d)

    # time-averaged quantities
    ('final_avg_K', float64),
    ('time_average_K', float64[:]),             # average kinetic energy
    ('final_avg_U', float64),
    ('time_average_U', float64[:]),             # average potential energy
    ('final_avg_temp', float64),
    ('time_average_temperature', float64[:]),   # average temperature
    ('final_avg_pressure', float64),
    ('time_average_pressure', float64[:]),      # average pressure

    # cached vals that rely on large rij calculation
    ('_F_cache', float64[:,:]),
    ('_U_cache', float64),
    ('_virial_cache', float64),
    ('_cache_step', int64),

    # recorded arrays
    ('positions', float64[:, :, :]),        # positions over time (n x d x frames)
    ('velocities', float64[:, :, :]),       # velocities over time (frames x n x d)
    ('wrap_history', int64[:, :, :]),       # history of PBC wraps (n x d x frames)
    ('unrolled_array', float64[:, :, :]),   # positions w/o PBCs (n x d x frames)
    ('K_array', float64[:]),                # kinetic energy over time (frames)
    ('U_array', float64[:]),                # potential energy over time (frames)
    ('total_energy_array', float64[:]),     # total energy over time (frames)
    ('temperature_array', float64[:]),      # temperature over time (frames)
    ('pressure_array', float64[:]),         # pressure over time (frames)
    ('momentum_array', float64[:, :]),      # momentum over time (d x frames)
    ('MSD_array', float64[:]),              # MSD over time (frames)
    ('COM_array', float64[:, :]),           # center of mass over time (d x frames)
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
        self.zeta = self.tau_damp
        self.post_eqbm_frames = int(max(1, (self.steps - self.eqbm_step) // self.record_stride))

        # integer wrap counts and unrolled
        self.wrap_count = np.zeros((self.n, self.d), dtype=np.int64)
        self.unrolled = self.xi.copy()

        # recording arrays
        self.positions = np.empty((self.n, self.d, self.frames), dtype=np.float64)
        self.velocities = np.empty((self.frames, self.n, self.d), dtype=np.float64)
        self.wrap_history = np.empty((self.n, self.d, self.frames), dtype=np.int64)
        self.unrolled_array = np.empty((self.n, self.d, self.frames), dtype=np.float64)

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

        self.final_avg_K = 0.0
        self.final_avg_U = 0.0
        self.final_avg_temp = 0.0
        self.final_avg_pressure = 0.0

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
        self.MSD_slope = 0.0
        self.MSD_intercept = 0.0
        self.center_of_mass = np.zeros(self.d, dtype=np.float64)
        self.diffusion_coeff = 0.0
        self.heat_capacity = 0.0

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
            x, v, f, self.zeta = self.nh_step(self.vi, self.xi, self.F, self.dt)
        else:
            x, v, f = self.vv_step(self.vi, self.xi, self.F, self.dt)
        
        # update system
        self.per_step_update(x, v, f)

    def vv_step(self, v, x, f, dt):
        # first velocity half step
        v += 0.5 * dt * f
        # position full step
        x += dt * v
        x = self.PBC(x)
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
        z = self.zeta
        v += 0.5 * dt * (f - z * v) 
        # position full step 
        x += dt * v 
        x = self.PBC(x)
        # update force
        self.compute_forces_and_energy()
        f = self._F_cache
        # update temperature at t0 + dt with new KE 
        T_inst = self.get_system_temp()
        z += dt * self.dzeta(T_inst, self.temp_des, self.tau_damp) 
        # second velocity half step 
        v = (v + 0.5 * dt * f) / (1 + 0.5 * dt * z) 
        return x, v, f, z

    def per_step_update(self, x, v, f):
        self.xi = x
        self.vi = v
        self.F = f

    def periodic_system_update(self):
        self.momentum = self.get_system_momentum()
        self.temperature = self.get_system_temp()
        self.pressure = self.get_system_pressure()
        self.center_of_mass = self.get_center_of_mass()

    def record_data(self):
        self.positions[:, :, self.frame] = self.xi
        self.velocities[self.frame, :, :] = self.vi
        self.wrap_history[:, :, self.frame] = self.wrap_count
        self.unrolled = self.xi + self.wrap_count * self.L
        self.unrolled_array[:, :, self.frame] = self.unrolled

        self.compute_forces_and_energy()
        K = self.get_total_K()
        self.K_array[self.frame] = K
        self.U_array[self.frame] = self._U_cache
        self.total_energy_array[self.frame] = K + self._U_cache

        self.momentum_array[:, self.frame] = self.momentum
        self.temperature_array[self.frame] = self.temperature
        self.pressure_array[self.frame] = self.pressure
        self.COM_array[:, self.frame] = self.center_of_mass

    def time_averages(self, frame, interval):
        self.time_average_K[frame] = self.calculate_avg_K(interval)
        self.time_average_U[frame] = self.calculate_avg_U(interval)
        self.time_average_temperature[frame] = self.calculate_avg_temp(interval)
        self.time_average_pressure[frame] = self.calculate_avg_pressure(interval)

    def final_calculations(self):
        self.unrolled = self.xi + self.wrap_count * self.L
        self.MSD_array = self.get_MSD_array()
        self.MSD = float(self.MSD_array[self.frame - 1]) if self.frame > 0 else self.get_MSD()
        self.MSD_slope, self.MSD_intercept, self.diffusion_coeff = self.get_diffusion_coefficient()
        self.K = self.get_total_K()
        self.U = self._U_cache
        self.final_avg_K = self.calculate_avg_K(0)
        self.final_avg_U = self.calculate_avg_U(0)
        self.final_avg_temp = self.calculate_avg_temp(0)
        self.final_avg_pressure = self.calculate_avg_pressure(0)

    def compute_forces_and_energy(self):
        if self._cache_step != self.i:
            self.forces_and_energy(self.xi)
            self._cache_step = self.i

    def forces_and_energy(self, xi):
        # bundle and cache expensive RIJ calculations 
        rij = xi[:, np.newaxis, :] - xi[np.newaxis, :, :]
        # NIC
        rij -= self.L * np.round(rij / self.L)

        r2 = np.sum(rij**2, axis=2)
        r = np.sqrt(r2)
        for k in range(r.shape[0]):
            r[k, k] = np.inf

        # avoid division by 0
        mask = (r < self.r_cut) & (r > 0.0)

        inv_r = np.zeros_like(r)
        n = r.shape[0]
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    inv_r[i, j] = 1.0 / r[i, j]

        r6 = inv_r**6

        # compute pairwise potential and force magnitudes
        U_pairs = np.zeros_like(r)
        F_mag = np.zeros_like(r)
        U_raw = 4 * (r6**2 - r6)
        shift = self.U_r_cut - (r - self.r_cut) * self.dU_r_cut
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    U_pairs[i, j] = U_raw[i, j] - shift[i, j]
                    F_mag_raw = 24 * inv_r[i, j] * (2 * r6[i, j]**2 - r6[i, j])
                    F_mag[i, j] = F_mag_raw - self.dU_r_cut
                else:
                    U_pairs[i, j] = 0.0
                    F_mag[i, j] = 0.0

        self._U_cache = 0.5 * np.sum(U_pairs)

        # compute norms
        F_vec = np.empty((n, n, self.d), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if inv_r[i, j] != 0.0:
                    for k in range(self.d):
                        F_vec[i, j, k] = F_mag[i, j] * rij[i, j, k] * inv_r[i, j]
                else:
                    for k in range(self.d):
                        F_vec[i, j, k] = 0.0

        self._F_cache = np.sum(F_vec, axis=1)

        # virial pressure
        tmp = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(self.d):
                    s += rij[i, j, k] * F_vec[i, j, k]
                tmp[i, j] = s
        self._virial_cache = 0.5 * np.sum(tmp)

    def LJ_F(self):
        self.compute_forces_and_energy()
        return self._F_cache

    def LJ_U(self):
        self.compute_forces_and_energy()
        return self._U_cache

    @staticmethod
    def dzeta(T_inst, T_des, damping):
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
        if interval == 0:
            return np.mean(self.K_array)
        if self.dt <= 0.0:
            return 0.0
        last = max(1, int(interval / self.dt))
        end = self.frame
        start = max(0, end - last)
        return np.mean(self.K_array[start:end])

    def calculate_avg_U(self, interval):
        if interval == 0:
            return np.mean(self.U_array)
        if self.dt <= 0.0:
            return 0.0
        last = max(1, int(interval / self.dt))
        end = self.frame
        start = max(0, end - last)
        return np.mean(self.U_array[start:end])

    def calculate_avg_temp(self, interval):
        if interval == 0:
            return np.mean(self.temperature_array)
        if self.dt <= 0.0:
            return 0.0
        last = max(1, int(interval / self.dt))
        end = self.frame
        start = max(0, end - last)
        return np.mean(self.temperature_array[start:end])

    def calculate_avg_pressure(self, interval):
        if interval == 0:
            return np.mean(self.pressure_array)
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
    
    def get_MSD_array(self):
        n = self.n
        d = self.d
        frames = self.frames
        msd_t = np.empty(frames, dtype=np.float64)

        for t in range(frames):
            sum_over_particles = 0.0
            for i in range(n):
                sum_over_dims = 0.0
                for j in range(d):
                    dx = self.unrolled_array[i, j, t] - self.x0[i, j]
                    sum_over_dims += dx * dx
                sum_over_particles += sum_over_dims
            msd_t[t] = sum_over_particles / float(n)

        return msd_t

    def get_diffusion_coefficient(self):
        start_frame = (self.eqbm_step + self.record_stride - 1) // self.record_stride
        if start_frame < 0:
            start_frame = 0

        end_frame = self.frame
        n_points = end_frame - start_frame
        if n_points <= 1:
            return 0.0, 0.0, 0.0

        t = make_x_axis(self.frames, self.dt, self.record_stride)

        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        valid_count = 0.0

        for idx in range(start_frame, end_frame):
            y = self.MSD_array[idx]
            # skip NaNs or infinities in MSD data
            if not (y == y) or y == np.inf or y == -np.inf:
                continue
            x = t[idx]
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            valid_count += 1.0

        if valid_count <= 1.0:
            return 0.0, 0.0, 0.0

        denom = valid_count * sxx - sx * sx
        if denom == 0.0:
            slope = 0.0
            intercept = sy / valid_count
        else:
            slope = (valid_count * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / valid_count

        D = slope / (2.0 * float(self.d))
        return slope, intercept, D

    def get_heat_capacity(self):
        # TODO
        pass

    def PBC(self, x):
        wrap = np.floor(x / self.L).astype(np.int64)
        self.wrap_count += wrap
        x = x - wrap * self.L
        return x

######################################################
# MAIN: run simulation
######################################################

def main():
    start_setup = time.time()

    # particle properties
    species = "Ar"          # argon
    mass = 6.634 * 1e-26    # kg
    sigma = 3.4             # Å
    epsilon = 1.66 * 1e-21  # J

    # simulation properties
    tau_damp = 0.05
    r_cut = 2.5
    cell_side_length = 6.8

    # temperature
    des_kelvin = 100
    guess_kelvin = des_kelvin
    temp_guess = T_kelvin_to_dimensionless(guess_kelvin, epsilon)
    temp_des = T_kelvin_to_dimensionless(des_kelvin, epsilon)

    # time scale
    num_steps = 1000
    step_size = 0.002
    record_freq = 10
    time_units = num_steps * step_size
    eqbm_timestep = 25000
    thermo_off_timestep = eqbm_timestep
        # = 0:              thermo off (use VV the whole time)
        # = eqbm_timestep:  bring to desired temperature, then begin data collection (recommended)
        # = num_steps:      thermo on (use NH the whole time - not recommended for measuring most quantities)

    end_setup = time.time()
    thinking_time = end_setup - start_setup
    print(f"\nProgram setup took {thinking_time:.4f} s.")

    test_positions, test_velocities = initialize_particle_system("liquid256.txt", 1, temp_des)

    start_system_setup = time.time()
    system = ParticleSystem(test_positions, test_velocities, r_cut, 
                            temp_guess, temp_des, eqbm_timestep, thermo_off_timestep, 
                            cell_side_length, num_steps, step_size, 
                            record_freq, tau_damp)
    end_system_setup = time.time()
    system_setup = end_system_setup - start_system_setup

    print(f"    Particle system setup complete. ({system_setup:.4f} s)\n")
    print(f"Running simulation for {num_steps} steps ({time_units} units of time)...")
    
    start_sim = time.time()
    system.run_simulation()
    end_sim = time.time()
    sim_time = end_sim - start_sim
    avg_per_step = sim_time / num_steps

    print(f"    Simulation ran successfully. ({sim_time:.2f} s | {avg_per_step:.4e} per timestep.)\n")
    
    final_outputs(system, num_steps, step_size, thermo_off_timestep, eqbm_timestep, species, mass, sigma, epsilon)

if __name__ == "__main__":
    main()
