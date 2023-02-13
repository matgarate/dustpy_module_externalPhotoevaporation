import numpy as np
from dustpy import constants as c


from functions_externalPhotoevaporation import get_MassLoss_ResampleGrid
from functions_externalPhotoevaporation import MassLoss_FRIED
from functions_externalPhotoevaporation import PhotoEntrainment_Size, PhotoEntrainment_Fraction
from functions_externalPhotoevaporation import SigmaDot_ExtPhoto, SigmaDot_ExtPhoto_Dust



################################################################################################
# Helper routine to add external photoevaporation to your Simulation object in one line.
################################################################################################

def setup_externalPhotoevaporation_FRIED(sim, fried_filename = "./friedgrid.dat", UV_Flux = 1000.,
                                            factor_SigmaFloor = 1.e-15, threshold_MassLoss = 1.05e-10 * c.M_sun/c.year):
    '''
    Add external photoevaporation using the FRIED grid (Haworth et al., 2018) and the Sellek et al.(2020) implementation.
    This setup routine also performs the interpolation in the stellar mass and UV flux parameters.

    Call the setup function after the initialization and then run, as follows:

    sim.initialize()
    setup_extphoto_FRIED(sim)
    sim.run()
    ----------------------------------------------

    fried_filename:             FRIED grid from Haworth+(2018), download from: http://www.friedgrid.com/Downloads/
    UV_target [G0]:             External UV Flux

    factor_SigmaFloor:          Re-adjust the floor value of the gas surface density to improve the simulation performance
    threshold_MassLoss[g/s]:    Further mass loss is prevented if the estimated mass loss rate drops below this treshold

    ----------------------------------------------
    '''


    ##################################
    # SET THE FRIED GRID
    ##################################

    # Obtain a resampled version of the FRIED grid for the simulation stellar mass and UV_flux.
    # Set the external photoevaporation Fields


    # Define a parameter space for the resampled radial and Sigma grids
    grid_radii = np.concatenate((np.array([1, 5]), np.linspace(10,400, num = 40)))
    grid_Sigma = np.concatenate((np.array([1e-8, 1e-6]), np.logspace(-5, 4, num = 100)))

    # Obtain the mass loss grid.
    # Also obtain the interpolator(M400, r) function to include in the FRIED class as a hidden function
    grid_MassLoss, grid_MassLoss_Interpolator = get_MassLoss_ResampleGrid(fried_filename= fried_filename,
                                                                            Mstar_target= sim.star.M[0]/c.M_sun, UV_target= UV_Flux,
                                                                            grid_radii= grid_radii, grid_Sigma= grid_Sigma)


    sim.addgroup('FRIED', description = "FRIED grid used to calculate mass loss rates due to external photoevaporation")
    sim.FRIED.addgroup('Table', description = "(Resampled) Table of the mass loss rates for a given radial-Sigma grid.")
    sim.FRIED.Table.addfield("radii", grid_radii, description ="Outer disk radius input to calculate FRIED mass loss rates [AU], (array, nr)")
    sim.FRIED.Table.addfield("Sigma", grid_Sigma, description = "Surface density grid to calculate FRIED mass loss rates [g/cm^2] (array, nSigma)")
    sim.FRIED.Table.addfield("MassLoss", grid_MassLoss, description = "FRIED Mass loss rates [log10 (M_sun/year)] (grid, nr*nSigma)")

    # We use this hidden _Interpolator function to avoid constructing the FRIED interpolator multiple times
    sim.FRIED._Interpolator = grid_MassLoss_Interpolator

    # Add the Mass Loss Rate field from the FRIED Grid
    sim.FRIED.addfield('MassLoss', np.zeros_like(sim.grid.r), description = 'Mass loss rate obtained by interpolating the FRIED Table at each grid cell [g/s]')

    sim.FRIED.MassLoss.updater =  MassLoss_FRIED
    sim.updater = ['star', 'grid', 'FRIED', 'gas', 'dust']
    sim.FRIED.updater = ['MassLoss']

    # Set a loss rate threshold
    # Prevents the simulation from getting stuck at very low disk masses
    sim.FRIED.addfield('Threshold_MassLoss', threshold_MassLoss, description = 'Further mass loss is prevented if the estimated value is below the threshold [g/s]')


    ###############################
    # DUST ENTRAINMENT
    ###############################
    # Add the entrainment size and the entrainment fraction for the dust loss rate.
    sim.dust.addgroup('Photo_Ent', description ="Dust entrainment fields")
    sim.dust.Photo_Ent.addfield('a_ent', sim.dust.a.T[-1], description = "Dust entrainment size [cm]")
    sim.dust.Photo_Ent.addfield('f_ent', np.ones_like(sim.dust.a), description = "Entrainment mass fraction")
    sim.dust.Photo_Ent.a_ent.updater = PhotoEntrainment_Size
    sim.dust.Photo_Ent.f_ent.updater = PhotoEntrainment_Fraction


    # The entrainment fraction needs to be updated before the dust source terms.
    sim.dust.Photo_Ent.updater = ['a_ent', 'f_ent']
    sim.dust.updater = ['delta', 'rhos', 'fill', 'a', 'St', 'H', 'rho', 'backreaction', 'v', 'D', 'eps', 'kernel', 'p', 'Photo_Ent','S']

    ###################################
    # ASSING GAS AND DUST LOSS RATES
    ###################################
    # Assign the External Photoevaporation Updater to the gas and dust
    sim.gas.S.ext.updater = SigmaDot_ExtPhoto
    sim.dust.S.ext.updater = SigmaDot_ExtPhoto_Dust



    ##################################
    # ADJUST THE GAS FLOOR VALUE
    ##################################
    # Setting higher floor value than the default avoids excessive mass loss rate calculations at the outer edge.
    # This speeds the code significantly, while still reproducing the results from Sellek et al.(2020)
    sim.gas.SigmaFloor = factor_SigmaFloor * sim.gas.Sigma
    sim.gas.SigmaFloor[-1] = 1.e-25

    sim.update()





################################################################################################
# Helper routine to add tracking of the dust lost through external photovaporation
################################################################################################
from simframe import Instruction
from simframe import schemes

def dSigma_lostdust(sim, x, Y):
    # Routing to evolve the dust surface density of the lost dust
    # This routine assumes that the dust is only lost through external photoevaporation
    return -sim.dust.S.ext

def M_lostdust(sim):
    # Computes the mass of the lost dust
    return (sim.grid.A * sim.lostdust.Sigma.sum(-1)).sum()


def setup_lostdust(sim, using_FRIED = True):
    '''
    Adds the group "lostdust" to the simulation object.
    This setup function is optional, and can used to track the total mass of dust removed by external sources
    using_FRIED: Set to true if the FRIED grid is implemented as well.
    '''


    # Creates the lost dust group and track the surface density and the total mass
    sim.addgroup("lostdust", description="Dust lost by photoevaporative entrainment")
    sim.lostdust.addfield("Sigma", np.zeros_like(sim.dust.Sigma), description="Lost dust surface density [g/cm²]")
    sim.lostdust.addfield("M", 0., description="Total mass of lost dust [photoevaporationg]")

    # Add the mass updater to the group and add the group to the simulation updater
    if using_FRIED:
        sim.updater = ["star", "grid", 'FRIED', "gas", "dust", "lostdust"]
    else:
        sim.updater = ["star", "grid", "gas", "dust", "lostdust"]


    sim.lostdust.updater = ["M"]

    # Assign the time derivative of the lost dust
    sim.lostdust.Sigma.differentiator = dSigma_lostdust
    # Assign the updater to track the total mass of lost dust
    sim.lostdust.M.updater = M_lostdust


    # Add the instruction to actually integrate the lost dust evolution in time
    inst_lostdust = Instruction(
        schemes.expl_1_euler,
        sim.lostdust.Sigma,
        description="Lost dust: explicit 1st-order Euler method")
    sim.integrator.instructions.append(inst_lostdust)

    sim.update()


################################################################################################
# Helper routine to remove dust evolution and run a gas-only simulation
################################################################################################

def setup_gasonly(sim):
    '''
    This routine deactivates all dust evolution source terms and integration instructions.
    Call it after setting up everything else in the simulation.
    '''
    sim.dust.S.coag[...] = 0.
    sim.dust.S.coag.updater = None
    sim.dust.S.ext[...] = 0.
    sim.dust.S.ext.updater = None
    sim.dust.S.hyd[...] = 0.
    sim.dust.S.hyd.updater = None
    sim.dust.S.tot[...] = 0.
    sim.dust.S.tot.updater = None
    sim.dust.S.updater = None
    sim.update()

    del(sim.integrator.instructions[0])
