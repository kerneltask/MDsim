MD.py

JIT-accelerated molecular dynamics simulator for Python

Uses Velocity Verlet to calculate particle positions with optional Nosé-Hoover thermostat to control temperature

Calculates system and material properties:

    Temperature
    Pressure
    Energies (kinetic and potential)
    Mean squared displacement (MSD)
    Self-diffusion coefficient

Produces an extended XYZ file for visualization in OVITO

Required packages:

    numpy
    numba
    matplotlib
    scipy
