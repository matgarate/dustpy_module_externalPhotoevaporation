import numpy as np
from dustpy import constants as c
from scipy.interpolate import interp1d, LinearNDInterpolator


from functions_externalPhotoevaporation import M400_FRIED, Set_FRIED_Interpolator
from functions_externalPhotoevaporation import MassLoss_FRIED
from functions_externalPhotoevaporation import PhotoEntrainment_Size, PhotoEntrainment_Fraction
from functions_externalPhotoevaporation import SigmaDot_ExtPhoto, SigmaDot_ExtPhoto_Dust

################################################################################################
# Helper routine to add external photoevaporation to your Simulation object in one line.
################################################################################################

def setup_externalPhotoevaporation_FRIED(sim, fried_filename = "./friedgrid.dat", star_mass = 1., UV_flux = 1000., factor_SigmaFloor = 1.e-15):
    '''
    Add external photoevaporation using the FRIED grid (Haworth et al., 2018) and the Sellek et al.(2020) implementation.
    Call the setup function after the initialization and then run, as follows:

    sim.initialize()
    setup_extphoto_FRIED(sim)
    sim.run()
    ----------------------------------------------

    fried_filename:     FRIED grid from Haworth+(2018), download from: http://www.friedgrid.com/Downloads/
    star_mass [M_sun]:          Stellar mass. It must be included in the FRIED grid: [0.05 0.1  0.3  0.5  0.8  1.   1.3  1.6  1.9 ]
    UV_flux [G0]:            External UV flux field. It must be included in the FRIED grid: [10.   100.  1000.  5000. 10000.]
    '''

    ##################################
    # LOAD FRIED GRID
    ##################################
    FRIED_Grid = np.loadtxt(fried_filename, unpack=True, skiprows=1)

    # Check that the stellar mass and the UV flux match those within the grid.
    if not star_mass in FRIED_Grid[0]:
        print("Star mass not available in FRIED")
    if not UV_flux in FRIED_Grid[1]:
        print("UV Flux not available in FRIED")

    FRIED_Grid = FRIED_Grid[1:, FRIED_Grid[0] == star_mass] # Remove the stellar mass dependency by picking one of the available masses
    FRIED_Grid = FRIED_Grid[1:, FRIED_Grid[0] == UV_flux] # Remove the UV_Field Dependency, which will be fixed, and pick one available brigthness


    # Set the external photoevaporation Fields
    sim.addgroup('FRIED', description = "FRIED table to calculate mass loss rates due to external photoevaporation")
    sim.FRIED.addgroup('Table', description = "Mass loss rate table")
    sim.FRIED.Table.addfield("Sigma", FRIED_Grid[1], description = "Surface density input to calculate FRIED mass loss rates [g/cm^2]")
    sim.FRIED.Table.addfield("r_out", FRIED_Grid[2], description ="Outer disk radius input to calculate FRIED mass loss rates [AU]")
    sim.FRIED.Table.addfield("Mass_loss", FRIED_Grid[3], description = "FRIED Mass loss rates [log10 (M_sun/year)]")


    ## Find the surface density limits.
    # Load the Sigma_out, r_out, and set the interpolator
    r_out = sim.FRIED.Table.r_out
    Sigma_out = sim.FRIED.Table.Sigma
    FRIED_Interpolator =  Set_FRIED_Interpolator(sim.FRIED.Table)

    # Give a buffer factor, since the FRIED interpolator cannot extrapolate outside the original domain
    buffer_max = 0.8 # buffer for the upper grid limit
    buffer_min = 1.2 # buffer for the lower grid limit

    shape_FRIED = (int(r_out.size/np.unique(r_out).size), np.unique(r_out).size)
    f_Sigma_FRIED_max = interp1d(r_out.reshape(shape_FRIED)[0], buffer_max * np.max(Sigma_out.reshape(shape_FRIED), axis= 0), kind='linear', fill_value = 'extrapolate' )
    f_Sigma_FRIED_min = interp1d(r_out.reshape(shape_FRIED)[0], buffer_min * np.min(Sigma_out.reshape(shape_FRIED), axis= 0), kind='linear',fill_value = 'extrapolate' )



    # Calculate the density limits and the corresponding mass loss rates
    r_AU = sim.grid.r / c.au
    Sigma_max = f_Sigma_FRIED_max(r_AU)
    Sigma_min = f_Sigma_FRIED_min(r_AU)
    Mass_loss_max = FRIED_Interpolator(M400_FRIED(Sigma_max, r_AU), r_AU) # Upper limit of the mass loss rate from the fried grid
    Mass_loss_min = FRIED_Interpolator(M400_FRIED(Sigma_min, r_AU), r_AU)  # Lower limit of the mass loss rate from the fried grid

    # Add the upper and lower limits of the FRIED grid as Fields
    sim.FRIED.addgroup('Limits', description = "Limits of the FRIED Grid in the surface density, and the corresponding mass loss rates")
    sim.FRIED.Limits.addfield('Sigma_min', Sigma_min, description = "Lower limit of the gas surface density [g/cm^2]")
    sim.FRIED.Limits.addfield('Sigma_max', Sigma_max, description = "Upper limit of the gas surface density [g/cm^2]")
    sim.FRIED.Limits.addfield('Mass_loss_min', Mass_loss_min, description = "Lower limit of mass loss rate [log10 (M_sun/year)]")
    sim.FRIED.Limits.addfield('Mass_loss_max', Mass_loss_max, description = "Upper limit of mass loss rate [log10 (M_sun/year)]")


    # Add the Mass Loss Rate field from the FRIED Grid
    sim.FRIED.addfield('MassLoss', np.zeros_like(sim.grid.r), description = 'Mass loss rate obtained by interpolating the FRIED Table at each grid cell [g/s]')
    sim.FRIED.MassLoss.updater =  MassLoss_FRIED
    sim.updater = ['star', 'grid', 'FRIED', 'gas', 'dust']
    sim.FRIED.updater = ['MassLoss']


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
