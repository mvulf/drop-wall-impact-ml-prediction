import numpy as np
from scipy.integrate import solve_ivp
# from numpy import inf

def get_impact_velocity(
    height,
    drop_diameter,
    drop_density,
    verbose=False,
):
    """
    Simulate droplet dynamics with drag force 
    and get droplet velocity before the impact

    Args:
        height: fall height (distance between droplet orifice and substrate) [m]
        drop_diameter: droplet diameter [m]
        drop_density: droplet density [kg/m^3]
        verbose: Display information or not. Defaults to False.

    Returns:
        Velocity before impact [m/s]
    """

    system = DropFallSystem(
        init_state=np.array([height, 0.]),
        system_parameters_init = {
            "drop_diameter": drop_diameter, # droplet diameter [m]
            "gas_density": 1.204, # air density [kg/m^3]
            "gas_viscosity": 1.825e-5,# air dynamic viscosity [Pa*s]
            "drop_density": drop_density, # droplet density [kg/m^3]
            "free_fall_acceleration": 9.81, # gravitational acceleration [m/s^2]
        }
    )
    
    numerical_results = solve_ivp(
        fun=system.compute_closed_loop_rhs,
        t_span=(0, 1e2), # s
        y0=system.init_state,
        method='RK45',
        events=_impact_event,
    )
    
    state = numerical_results.y.T[-1]
    if verbose:
        time = numerical_results.t[-1]
        print(f'Time of fall: {time:.3f}')
        print(f'Last state: {state}')
    
    return state[1]


def _impact_event(time, state):
    """Supplementary function for the solve_ivp to track impact
    """
    drop_position = state[0]
    return -drop_position
_impact_event.terminal = True


class DropFallSystem():
    _name = 'DropFallSystem'
    _system_type = 'diff_eqn'
    _dim_state = 2
    # _dim_inputs = 0
    # _dim_observation = 2
    _state_naming = [
        "droplet position [m]",
        "droplet velocity [m]"
    ]
    
    def __init__(
        self,
        init_state,
        *args,
        system_parameters_init=None,
        **kwargs,
    ):
        """Initialize droplet fall system with drag force 0

        Args:
            init_state: droplet position and velocity at the beginning
            system_parameters_init: parameters of the system. See description below.
            Defaults to None.
        """
        self.init_state = init_state
        
        if system_parameters_init is None:
            system_parameters_init = {
                # "height": 2.0, # droplet fall height [m]
                "drop_diameter": 3.0e-3, # droplet diameter [m]
                "gas_density": 1.204, # air density [kg/m^3]
                "gas_viscosity": 1.825e-5,# air dynamic viscosity [Pa*s]
                "drop_density": 998.2, # droplet density [kg/m^3]
                "free_fall_acceleration": 9.81, # gravitational acceleration [m/s^2]
            }
            print('Standard parameters are set')
            print(system_parameters_init)
        
        self._system_parameters_init = system_parameters_init
        self._parameters = system_parameters_init.copy()
        
        self._parameters["cross_sectional_area"] = (
            1/4 * np.pi * self._parameters["drop_diameter"]**2
        ) # [m^2]
        
        self._parameters["drop_mass"] = (
            1/6 * np.pi * self._parameters["drop_diameter"]**3
            * self._parameters["drop_density"]
        ) # [kg]
        
        # self._parameters["gravity_force"] = (
        #     drop_mass * self._parameters["free_fall_acceleration"]
        # ) # [N]
    
    
    def compute_closed_loop_rhs(self, time, state):
        """ Compute right-hand-side of the droplet dynamics

        Args:
            time: current time
            state: current state [drop_position, drop_velocity]

        Returns:
            Right-hand-side of the droplet dynamics with drag force
        """
        Dstate = np.zeros(self._dim_state)
        
        # Get current state parameters
        # drop_position, drop_velocity = [
        #     state[i] for i in range(self._dim_state)
        # ]
        drop_position = state[0]
        drop_velocity = state[1]
        
        m_drop, g = (
            self._parameters["drop_mass"],
            self._parameters["free_fall_acceleration"],
        )
        
        Dstate[0] = -drop_velocity # droplet position changes by its velocity
        
        if drop_velocity == 0:
            Dstate[1] = g # Free fall at the beginning
        else:
            F_D = self.get_drag_force(
                velocity=drop_velocity
            )
            Dstate[1] = g + F_D/m_drop # Fall with drag force acceleration
        
        return Dstate
    
    
    def get_drag_force(self, velocity):
        """ Get drag force acting on the droplet falling in the gas (air)

        Args:
            velocity: current droplet velocity

        Returns:
            Force acting on the droplet
        """
        
        rho_g, D, mu_g, A = (
            self._parameters["gas_density"],
            self._parameters["drop_diameter"],
            self._parameters["gas_viscosity"],
            self._parameters["cross_sectional_area"],
        )
        
        Re = get_Re(
            velocity=np.abs(np.array([velocity])), # Necessary to pass array
            diameter=D,
            density=rho_g,
            viscosity=mu_g
        )
        C_D = get_drag_coef(Re)[0] # Get scalar drag coefficient
        
        F_D = -1/2 * rho_g * velocity*np.abs(velocity) * C_D * A # Drag force orientation considered
        
        return F_D
        
        
        
def get_drag_coef(Re):
    """Calculate drag coefficient based on Clif and Gauvin fit 
    [https://doi.org/10.1002/cjce.5450490403].
    Mentioned by Eric Loth in 
    "Supersonic and Hypersonic Drag Coefficients for a Sphere" 
    [https://doi.org/10.2514/1.J060153 ]

    Args:
        Re: Reynolds number

    Raises:
        ValueError: if any Reynolds number is negative, raise value error

    Returns:
        numpy array of drag coefficients
    """
     
    if (Re < 0).any():
        raise ValueError('Negative Reynolds number!')
    
    # C_D = np.zeros(Re.shape)
    # C_D = (
    #     24/Re*(1 + 0.15*Re**0.687)
    #     + 0.42/(1 + 42_500/Re**1.16)
    # )
    
    C_D = np.inf*np.ones(Re.shape)
    pos_mask = Re > 0
    pos_Re = Re[pos_mask]
    C_D[pos_mask] = (
        24/pos_Re*(1 + 0.15*pos_Re**0.687)
        + 0.42/(1 + 42_500/pos_Re**1.16)
    )
    
    return C_D


def get_Re(velocity, diameter, density, viscosity):
    """Calculate Reynolds number

    Args:
        velocity: particle or droplet velicity [m/s]
        diameter: particle or droplet diameter [m]
        density: fluid (gas) density [kg/m^3]
        viscosity: fluid (gas) viscosity [Pa*s]

    Returns:
        Reynolds number
    """
    Re = velocity * diameter * density / viscosity
    
    return Re