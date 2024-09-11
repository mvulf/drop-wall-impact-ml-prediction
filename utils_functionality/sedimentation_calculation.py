import numpy as np
from . import velocity_calculation as vc


# TODO: Check!
def get_suspension_drag_coef(Re, volume_fraction):
    """Calculate drag coefficient for particle sedimenting in the monodispersed suspension based on DiFelice1994VoidageFunction [https://doi.org/10.1016/0301-9322(94)90011-6]
    Described in multiphase-flow-handbook [https://doi.org/10.1201/9781420040470]

    Args:
        Re: Reynolds number
        volume_fraction: Particle volume fraction (V_p/V)

    Raises:
        ValueError: if any Reynolds number is negative, raise value error

    Returns:
        numpy array of drag coefficients
    """
    
    if (Re < 0).any():
        raise ValueError('Negative Reynolds number!')
    
    C_D = np.inf*np.ones(Re.shape)
    pos_mask = Re > 0
    pos_Re = Re[pos_mask]
    
    C_D0 = vc.get_drag_coef(pos_Re)
    K = 3.7 - 0.65*np.exp(
        -0.5 * (1.5 - np.log10(pos_Re))**2
    )
    
    C_D[pos_mask] = C_D0 * (1 - volume_fraction)**(-K)
    
    return C_D