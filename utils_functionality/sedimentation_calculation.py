import numpy as np
import velocity_calculation as vc


# TODO: Check!
def get_suspension_drag_coef(Re, volume_fraction):
    
    if (Re < 0).any():
        raise ValueError('Negative Reynolds number!')
    
    C_D = np.inf*np.ones(Re.shape)
    pos_mask = Re > 0
    pos_Re = Re[pos_mask]
    
    C_D0 = vc.get_drag_coef(pos_Re)
    K = 3.7 - 0.65*np.exp(
        -0.5 * (1.5 - np.log10(pos_Re))**2
    )
    
    C_D[pos_mask] = (
        C_D0 * (1 - volume_fraction)**(-K)
    )
    
    return C_D