import numpy as np
from dustpy import constants as c
from scipy.interpolate import LinearNDInterpolator


#####################################
# FRIED GRID ROUTINES
#####################################


def M400_FRIED(Sigma_out, r_out):
    '''
    Transformed variable for the FRIED Grid.
    Receives r_out [au], and sigma_out  [g/cm^2].
    Returns a representative mass  [jupiter masses]
    '''
    return 2 * np.pi * Sigma_out * (r_out * c.au)**2 * (r_out/400)**(-1) / c.M_jup


def Set_FRIED_Interpolator(Table):
    '''
    Returns the interpolator function, constructed from the FRIED grid data
    The interpolator takes the (M400 [Jupiter mass], r_out[au]) variables
    The interpolator returns the external photoevaporation mass loss rate [log10 (M_sun/year)]
    '''

    r_out = Table.r_out
    Sigma_out = Table.Sigma
    Mass_loss = Table.Mass_loss

    # Following Sellek et al.(2020) implementation, the M400 converted variable is used to set the interpolator
    M400 = M400_FRIED(Sigma_out, r_out)
    return  LinearNDInterpolator(list(zip(M400, r_out)), Mass_loss)

def MassLoss_FRIED(sim):
    '''
    Calculates the instantaneous mass loss rate from the FRIED Grid (Haworth+, 2018) for each grid cell,
    using each r[i] as the input value r_out.
    '''

    r_AU = sim.grid.r / c.au
    Sigma_g = sim.gas.Sigma

    Sigma_max = sim.FRIED.Limits.Sigma_max
    Sigma_min = sim.FRIED.Limits.Sigma_min
    mass_loss_max = sim.FRIED.Limits.Mass_loss_max
    mass_loss_min = sim.FRIED.Limits.Mass_loss_min

    # Interpolation of the mass loss rate  with the the transformed variable M400 and the outer disk radius
    FRIED_Interpolator =  Set_FRIED_Interpolator(sim.FRIED.Table)

    mask_max= Sigma_g >= Sigma_max
    mask_min= Sigma_g <= Sigma_min

    # Calculate the mass loss rate for each grid cell according to the FRIED grid
    # Note that the mass loss rate is in logarithmic-10 space
    mass_loss_FRIED = FRIED_Interpolator(M400_FRIED(Sigma_g, r_AU), r_AU) # Mass loss rate from the FRIED grid
    mass_loss_FRIED[mask_max] = mass_loss_max[mask_max]
    mass_loss_FRIED[mask_min] = mass_loss_min[mask_min] + np.log10(Sigma_g / Sigma_min)[mask_min]
    mass_loss_FRIED[mass_loss_FRIED < -10] = -10

    # Clean the NANS that were still outside the grid
    mass_loss_FRIED[np.isnan(mass_loss_FRIED)] = -10


    # Convert the mass loss rate to cgs units in linear space
    mass_loss_FRIED = np.power(10, mass_loss_FRIED) * c.M_sun/c.year

    return mass_loss_FRIED

#####################################
# GAS LOSS RATE
#####################################

def SigmaDot_ExtPhoto(sim):

    '''
    Compute the Mass Loss Rate profile using Sellek+(2020) approach, using the mass loss rates from the FRIED grid of Haworth+(2018)
    '''


    # Find the photoevaporative radii.
    # See Sellek et al. (2020) Figure 2 for reference.
    ir_ext = np.argmax(sim.FRIED.MassLoss)


    # Obtain Mass at each radial ring and total mass outside the photoevaporative radius
    mass_profile = sim.grid.A * sim.gas.Sigma
    mass_ext = np.sum(mass_profile[ir_ext:])

    # Total mass loss rate.
    mass_loss_ext = np.sum((sim.FRIED.MassLoss * mass_profile)[ir_ext:] / mass_ext)

    # Obtain the surface density profile using the mass of each ring as a weight factor
    # Remember to add the (-) sign to the surface density mass loss rate
    SigmaDot = np.zeros_like(sim.grid.r)
    SigmaDot[ir_ext:] = -sim.gas.Sigma[ir_ext:] *  mass_loss_ext / mass_ext


    # If the surface density is within a factor of 10 near the floor, stop futhrer mass loss
    FloorThreshold = 10
    SigmaDot[sim.gas.Sigma < FloorThreshold * sim.gas.SigmaFloor] = 0

    # return the surface density loss rate [g/cm²/s]
    return SigmaDot


#####################################
# DUST ENTRAINMENT AND LOSS RATE
#####################################

def PhotoEntrainment_Size(sim):
    '''
    Returns a radial array of the dust entrainment size.
    See Eq. 11 from Sellek+(2020)
    '''
    v_th = np.sqrt(8/np.pi) * sim.gas.cs                    # Thermal speed
    F = sim.gas.Hp / np.sqrt(sim.gas.Hp**2 + sim.grid.r**2) # Geometric Solid Angle
    rhos = sim.dust.rhos[0,0]                               # Dust material density

    # Calculate the total mass loss rate (remember to add the (-) sign)
    M_loss = -np.sum(sim.grid.A * sim.gas.S.ext)

    a_ent = v_th / (c.G * sim.star.M) * M_loss /(4 * np.pi * F * rhos)
    return a_ent

def PhotoEntrainment_Fraction(sim):
    '''
    Returns fraction of dust grains that are entrained with the gas for each species at each location.
    * Must be multiplied by the dust-to-gas ratio to account for the mass fraction
    * Must be zero for grain sizes larger than the entrainment size
    * In Sellek+2020 the mass fraction is used to account for the dust distribution as well, but in dustpy that information comes for free in the sim.dust.Sigma array
    * Currently this factor must be either 1 (entrained) or 0 (not entrained)
    '''

    mask = sim.dust.a < sim.dust.Photo_Ent.a_ent[:, None] # Mask indicating which grains are entrained
    f_ent = np.where(mask, 1., 0.)
    return f_ent


def SigmaDot_ExtPhoto_Dust(sim):

    f_ent = sim.dust.Photo_Ent.f_ent                                # Factor to mask the entrained grains.
    d2g_ratio = sim.dust.Sigma / sim.gas.Sigma[:, None]             # Dust-to-gas ratio profile for each dust species
    SigmaDot_Dust = f_ent * d2g_ratio * sim.gas.S.ext[:, None]      # Dust loss rate [g/cm²/s]

    return SigmaDot_Dust
