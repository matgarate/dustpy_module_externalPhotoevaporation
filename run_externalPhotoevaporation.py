import numpy as np
import sys
import os


import dustpy
from dustpy import Simulation
from dustpy import constants as c



from setup_externalPhotoevaporation import setup_externalPhotoevaporation_FRIED
from setup_externalPhotoevaporation import setup_lostdust
from setup_externalPhotoevaporation import setup_gasonly

################################
# OPTIONAL PARAMETERS
################################

option_sqrt_grid = True         # Run the simulation with a grid linearly spaced in r^1/2
option_track_lostdust = False   # Track the time evolution of the mass lost by external photoevaporation
option_gasonly = True           # Deactivate the dust evolution (ideal for quick tests)


################################
# SIMULATION SETUP
################################

sim = Simulation()

# Star and disk parameters
sim.ini.star.M = 1.0 * c.M_sun              # Stellar mass
sim.ini.gas.Mdisk = 0.1 * sim.ini.star.M    # Initial disk mass
sim.ini.gas.SigmaRc = 60 * c.au             # Initial surface density characteristic radii
sim.ini.gas.gamma = 1.0                     # Adiabatic Index 1.0 for Isothermal gas


# Relevant Dust parameters
sim.ini.dust.d2gRatio = 0.01                # Initial dust-to-gas ratio
sim.ini.gas.alpha = 1.e-3                   # Alpha turbulence parameter
sim.ini.dust.vfrag = 1000.0                 # Dust fragmentation velocity


################################
# GRID SETUP
################################

# Radial Grid Parameters
sim.ini.grid.Nr = 250
sim.ini.grid.rmin = 5 * c.au
sim.ini.grid.rmax = 500 * c.au



# Mass Grid Parameters
sim.ini.grid.Nmbpd = 7
sim.ini.grid.mmin = 1.e-12
sim.ini.grid.mmax = 1.e5


# Reduce the size of the dust grid for a gas only simulation
if option_gasonly:
    sim.ini.grid.Nmbpd = 4
    sim.ini.grid.mmax = 1.e-9



# Option to set a customized radial grid to linearly spaced in r^1/2
if option_sqrt_grid:
    sim.grid.ri = np.square(np.linspace(np.sqrt(sim.ini.grid.rmin), np.sqrt(sim.ini.grid.rmax), num = sim.ini.grid.Nr +1))

sim.initialize()

################################
# EXTERNAL PHOTOEVAPORATION SETUP
################################

# Add the next line to your script after "initialize()"" to setup the external photoevaporation group and relevant updaters

setup_externalPhotoevaporation_FRIED(sim, fried_filename = "./friedgrid.dat", star_mass = 1., UV_flux = 1000., factor_SigmaFloor = 1.e-15)

# The user needs to input the location of the FRIED grid
# The user needs to define the stellar mass [Msun] UV flux [G0] to be used.

# The values stellar mass and UV flux need to available in the FRIED grid.
# Available stellar masses [Msun]:     [0.05 0.1  0.3  0.5  0.8  1.   1.3  1.6  1.9 ]
# Available UV fluxes [G0]:          [10.   100.  1000.  5000. 10000.]

# The gas surface density floor value is adjusted for performance



# Optional dust lost tracking
if option_track_lostdust:
    setup_lostdust(sim, using_FRIED = True)

# Optional gas only setup
if option_gasonly:
    setup_gasonly(sim)

################################
# RUN SIMULATION
################################
print("Running Simulation")

# Set the output directory
sim.writer.datadir = "./Simulation/"

# Set the snapshots
sim.t.snapshots = np.linspace(0.5, 5.0, 11) * 1.e5 * c.year



# Due to some error we cannot write the dumpfiles for these simulations :(
sim.writer.dumping = False
sim.writer.overwrite = True
sim.verbosity = 2

sim.run()
