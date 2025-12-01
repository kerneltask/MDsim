# MDsim

MDsim contains three Python programs for molecular dynamics simulation of Lennard-Jones particles. It uses Velocity Verlet integration with an optional Nose-Hoover thermostat for temperature control.

This program was written for 12-623, Molecular Simulation of Materials at Carnegie Mellon University.

MD.py: benchmark program with O(N^2) pair calculations
MD_verlet_list.py: modification of MD.py that implements Verlet lists for O(N^3/2) pair calculations
MD_cell_list.py: modification of MD.py that implements cell lists for O(N) pair calculations

## Output

The simulation provides the following output for validation of expected physical properties:

XYZ format files for visualization in OVITO:

* positions.txt (applied periodic boundary conditions)
* unrolled.txt (no PBCs)

Plots:

* COM_plot
* energy_plot
* momentum_plot
* velocity_hist
* temp_pressure_plots
* MSD

CSV files:

* energies
* temp_pressure
* time_averages

Console output:

* Time-averaged energy, temperature, and pressure of the system
* Self-diffusion coefficient
* Slope of mean-squared displacement over time
* Simulation time

## Disclaimers

System parameters may be altered in main()

The program expects quantities in reduced Lennard-Jones units, except for temperature, which can be entered in Kelvin.

It has not been fully tested in dimensions other than 3D.
