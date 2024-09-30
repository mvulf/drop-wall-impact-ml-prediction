import numpy as np
from . import velocity_calculation as vc
from collections.abc import Iterable
from IPython.display import display

from scipy.optimize import fsolve


class IntegratedSedimentationSystem():
    _name = 'IntegratedSedimentationSystem'
    _system_type = 'diff_eqn'
    _dim_state = 2
    
    _state_naming = [
        "particle velocity [m/s]",
        "particle volume fraction in droplet",
        # "concentration change rate [1/s]"
    ]

    def __init__(
        self,
        init_state,
        *args,
        system_parameters_init=None,
        verbose=True,
        **kwargs,
    ):
        """Initialize Sedimentation system

        Args:
            init_state: 2D-vector. Particle velocity at the droplet interface and volume fraction of particles in the droplet
            system_parameters_init: parameters of the system. See description below.
            Defaults to None.
        """
        self.init_state = init_state.copy()
        
        if system_parameters_init is None:
            system_parameters_init = {
                "particle_size": 41.5e-6, # particle diameter [m]
                "droplet_size": 3e-3, # droplet diamter [m]
                "particle_liquid_density_ratio": 1200/1180, # epsilon_p = rho_p/rho_l
                "density_liquid": 1180, # [kg/m^3]
                "viscosity_liquid": 23.1e-3, # Dynamic viscosity [Pa*s]
                "free_fall_acceleration": 9.81, # gravitational acceleration [m/s^2]
                "diameter_exit": 1.6e-3, # exit diameter of the tip [m]
                "basic_volume_fraction": 0.10, # initial, basic, volume fraction for the system
                "weight_basic_volume_fraction": 0.5, # coefficient of the basic volume fraction in interface volume fraction estimation
            }
            print('Standard parameters are set')
            print(system_parameters_init)
        
        self._system_parameters_init = system_parameters_init
        self._parameters = system_parameters_init.copy()
        
        self._calc_state = 0 # init time calc state
        
        self._parameters['height_drop'] = get_height_drop(
            D_drop=self._parameters['droplet_size'],
            D_exit=self._parameters['diameter_exit']
        )
        
        self._parameters['buoyancy_acceleration'] = (
            (1 - 1/self._parameters['particle_liquid_density_ratio'])
            *self._parameters['free_fall_acceleration']
        )
        
        phi_0, alpha = (
            self._parameters["basic_volume_fraction"],
            self._parameters["weight_basic_volume_fraction"]
        )
        
        self._parameters["constant_of_syringe_volume_fraction"] = (
            alpha*phi_0
        )
        
        self._parameters["constant_of_drop_volume_fraction_change_rate"] = (
            1/self._parameters["height_drop"]
            * phi_0
        )
        self._parameters["weight_droplet_volume_fraction"] = 1 - alpha
        
        self._parameters["constant_of_Re"] = (
            self._parameters["particle_size"] 
            * self._parameters["density_liquid"] 
            / self._parameters["viscosity_liquid"]
        )
        
        if verbose:
            print('Init parameters')
            display(self._parameters)
        
    
    def compute_closed_loop_rhs(self, time, state, pbar, indicator_dt):
        """ Compute right-hand-side of the sedimentation dynamics

        Args:
            time: current time
            state: current state
            pbar (tqdm): progress bar for the calculation tracing

        Returns:
            Right-hand-side of the sedimentation dynamics
        """
        
        v_interface, phi_drop = state
        # Dstate
        Dstate = np.zeros_like(state)
        
        # Get system parameters
        B_Dphi_drop =\
            self._parameters["constant_of_drop_volume_fraction_change_rate"]

        Dv_interface = self.get_interface_acceleration(v_interface, phi_drop)
   
        # DROPLET VOLUME FRACTION CHANGE RATE
        Dphi_drop = B_Dphi_drop * v_interface
        
        Dstate[0] = Dv_interface
        Dstate[1] = Dphi_drop
        
        # Return tqdm-status
        # Based on https://gist.github.com/thomaslima/d8e795c908f334931354da95acb97e54
        # Time-step reaching indicator
        n = int((time - self._calc_state)/indicator_dt)
        # Progress bar increment: 
        # when indicator time step (self._indicator_dt) reached -> 1, else 0
        pbar.update(n)
        # Update current state (when n = 0, no real update)
        self._calc_state += indicator_dt*n
        
        return Dstate
    
    
    def get_interface_acceleration(self, v_interface, phi_drop):
        """Calculate acceleration of the particle at the droplet interface

        Args:
            v_interface: particle velocity at the interface
            phi_drop: particle volume fraction in the droplet

        Returns:
            Acceleration of the particle at the droplet interface
        """
        
        # Get system parameters
        d_p, eps_p, a_b, B_Re = (
            self._parameters["particle_size"],
            self._parameters["particle_liquid_density_ratio"],
            self._parameters["buoyancy_acceleration"],
            self._parameters["constant_of_Re"]
        )
        
        # Simplify, create special constants
        phi_interface = self.get_interface_volume_fraction(phi_drop)

        Re = self.get_Re(v_interface)
        C_D = get_suspension_drag_coef(
            Re=Re,
            volume_fraction=phi_interface,
            corrections=False,
        )[0]
        
        # ACCELERATION
        # Get free fall acceleration
        Dv_interface = a_b
        # Add drag force if velocity is not zero
        if v_interface != 0:
            Dv_interface -= (
                0.75*v_interface*abs(v_interface)*C_D
                /(eps_p*d_p)
            )
        
        return Dv_interface
    
    
    def get_terminal_velocity(
        self,
        v_interface:float=None,
        phi_drop:float=None,
        **kwargs,
    ):
        # Previous interface speed is used as initial approximation 
        if v_interface is None:
            v_interface = 0.
        
        if phi_drop is None:
            phi_drop = self._parameters['basic_volume_fraction']
        
        terminal_velocity = fsolve(
            self.get_interface_acceleration,
            x0=v_interface,
            args=(phi_drop,),
            **kwargs,
        )
        
        return terminal_velocity
    
    
    def get_Re(
        self,
        v_interface,
    ):
        """ Calculate Reynolds number
        """
        B_Re = self._parameters["constant_of_Re"]
        
        Re = np.array([abs(v_interface)]) * B_Re
        
        return Re


    def get_interface_volume_fraction(self, phi_drop):
        """ Get volume fraction at the droplet interface, which is used for the drag coefficient evaluation

        Args:
            phi_drop: particle volume fraction in the droplet

        Returns:
            Interface volume fraction
        """
        
        phi_0, B_phi_syringe, alpha_drop = (
            self._parameters["basic_volume_fraction"],
            self._parameters["constant_of_syringe_volume_fraction"],
            self._parameters["weight_droplet_volume_fraction"],
        )
        
        if phi_drop > phi_0:
            phi_interface = B_phi_syringe + alpha_drop*phi_drop
        # In case of positive buoyancy (eps_p < 1.0)
        else:
            phi_interface = phi_0
        
        return phi_interface



class SedimentationSystem():
    _name = 'SedimentationSystem'
    _system_type = 'diff_eqn'
    
    _state_naming = [
        "particle position [m]",
        "particle velocity [m/s]",
        "particle concentration",
        # "concentration change rate [1/s]"
    ]

    def __init__(
        self,
        init_state,
        *args,
        system_parameters_init=None,
        **kwargs,
    ):
        """Initialize Sedimentation system

        Args:
            init_state: particles position and their velocity;
                concentrations and their initial change rates at the beginning
            system_parameters_init: parameters of the system. See description below.
            Defaults to None.
        """
        self.init_state = init_state.copy()
        
        if system_parameters_init is None:
            system_parameters_init = {
                "particle_size": 41.5e-6, # particle diameter [m]
                "particle_liquid_density_ratio": 1200/1180, # epsilon_p = rho_p/rho_l
                "density_liquid": 1180, # [kg/m^3]
                "viscosity_liquid": 23.1e-3, # Dynamic viscosity [Pa*s]
                "free_fall_acceleration": 9.81, # gravitational acceleration [m/s^2]
                "height_exit": 10e-3, # effective height of the sedimentation [m]
                "n_lagrangian_particles": 31, # number of lagrangian particles
                "n_eulerian_nodes": 31, # number of eulerian nodes
                "n_bottom_nodes": 2, # number of nodes to make zero changing
            }
            print('Standard parameters are set')
            print(system_parameters_init)
        
        self._system_parameters_init = system_parameters_init
        self._parameters = system_parameters_init.copy()
        
        self._calc_state = 0 # init time calc state
        
    
    def compute_closed_loop_rhs(self, time, state, pbar, indicator_dt):
        """ Compute right-hand-side of the sedimentation dynamics

        Args:
            time: current time
            state: current state
            pbar (tqdm): progress bar for the calculation tracing

        Returns:
            Right-hand-side of the sedimentation dynamics
        """
        
        # Dstate: vstack of sub Dstates
        
        # Get system parameters
        d_p, eps_p, rho_l, mu_l, g = (
            self._parameters["particle_size"],
            self._parameters["particle_liquid_density_ratio"],
            self._parameters["density_liquid"],
            self._parameters["viscosity_liquid"],
            self._parameters["free_fall_acceleration"],
        )
        h_exit, N_L, N_E = (
            self._parameters["height_exit"],
            self._parameters["n_lagrangian_particles"],
            self._parameters["n_eulerian_nodes"],
        )
        
        z_nodes = np.linspace(0, h_exit, N_E)
        
        # z_p = state[:N_L]
        # v_p = state[N_L:2*N_L]
        # phi = state[2*N_L:2*N_L+N_E]
        # q_phi = state[2*N_L+N_E:2*N_L+2*N_E]
        
        z_p, v_p, phi = self.get_substates(state) # q_phi
        
        Re = vc.get_Re(
            velocity=np.abs(v_p), # Necessary to pass array
            diameter=d_p,
            density=rho_l,
            viscosity=mu_l,
        )
        # phi-interpolation for the Lagrangian particles
        phi_L = np.interp(
            x=z_p,
            xp=z_nodes,
            fp=phi, 
        )
        C_D = get_suspension_drag_coef(
            Re=Re,
            volume_fraction=phi_L,
            corrections=False,
        )
        
        # VELOCITY
        Dz_p = v_p
        # ACCELERATION
        # Get free fall acceleration
        Dv_p = (1 - 1/eps_p)*g*np.ones_like(v_p)
        # Dv_p[nonboundary_mask] = (1 - 1/eps_p)*g*np.ones_like(v_p[nonboundary_mask])
        # Add drag force if velocity is not zero
        drag_mask = (v_p != 0)
        # mask = drag_mask & nonboundary_mask
        if drag_mask.any():
            Dv_p[drag_mask] -= (
                0.75*v_p[drag_mask]*np.abs(v_p[drag_mask])*C_D[drag_mask]
                /(eps_p*d_p)
            )
        
        # velocity-interpolation for the Eulerian nodes
        v_p_E = np.interp(
            x=z_nodes,
            xp=z_p,
            fp=v_p,
        )
        # # Velocity at the bottom is equal to zero
        # v_p_E[-1] = 0
        
        # TODO: ADD VELOCITY GRADIENT AT THE BOTTOM
        # IDEA: Multiply velocity with the coefficient from 0 to 1!
        N_BN = self._parameters["n_bottom_nodes"]
        
        # Only last two nodes have zero-velocity
        coefs = 0
        v_p_E[-N_BN:] = v_p_E[-N_BN:]*coefs
        
        # Space differentiation for the concentration change rate
        Dphi = np.zeros_like(phi) # NOTE: First node concentration change rate is constant
        Dphi[1:] = - (
            (phi[1:]*v_p_E[1:] - phi[:-1]*v_p_E[:-1])
            /np.diff(z_nodes)
        )
        
        Dstate = np.hstack(
            (
                Dz_p,
                Dv_p,
                Dphi,
                # Dq_phi
            )
        )
        
        # Return tqdm-status
        # Based on https://gist.github.com/thomaslima/d8e795c908f334931354da95acb97e54
        # Time-step reaching indicator
        n = int((time - self._calc_state)/indicator_dt)
        # Progress bar increment: 
        # when indicator time step (self._indicator_dt) reached -> 1, else 0
        pbar.update(n)
        # Update current state (when n = 0, no real update)
        self._calc_state += indicator_dt*n
        
        return Dstate
    
    
    def get_substates(self, state, verbose=False, display_cnt=None):
        """Get sub states of the state: particles position, their velocity, particle concentration, particle concentration change rate

        Args:
            state: full state
            verbose: Print substates. Defaults to False.

        Returns:
            sub states of the state [z_p, v_p, phi, q_phi]
        """
        N_L, N_E = (
            self._parameters["n_lagrangian_particles"],
            self._parameters["n_eulerian_nodes"],
        )
        
        z_p = state[:N_L]
        v_p = state[N_L:2*N_L]
        phi = state[2*N_L:2*N_L+N_E]
        # q_phi = state[2*N_L+N_E:2*N_L+2*N_E]
        
        if verbose:
            if (display_cnt is None):   
                print(f'z_p [m] = {z_p}')
                print(f'v_p [m/s] = {v_p}')
                print(f'phi = {phi}')
                # print(f'q_phi [1/s] = {q_phi}')
            else:
                print(f'z_p [m] = {z_p[-display_cnt:]}')
                print(f'v_p [m/s] = {v_p[-display_cnt:]}')
                print(f'phi = {phi[-display_cnt:]}')
                # print(f'q_phi [1/s] = {q_phi[-display_cnt:]}')
        
        return z_p, v_p, phi # q_phi



def get_height_drop(
    D_drop:float, # [m]
    D_exit:float, # default exit diameter of the syringe tip [m]
):
    """Get height of the cylinder with the D_exit diamter and droplet volume 

    Args:
        D_drop: Droplet size for the droplet volume estimation
        D_exit: Exit diameter of the tip [m]

    Returns:
        height of the cylinder with the D_exit diamter and droplet volume
    """

    drop_height = 2/3 * D_drop**3/D_exit**2
    
    return drop_height


def get_suspension_drag_coef(Re, volume_fraction, corrections=True):
    """Calculate drag coefficient for particle sedimenting 
    in the monodispersed suspension based on 
    DiFelice1994VoidageFunction [https://doi.org/10.1016/0301-9322(94)90011-6]
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
    
    C_D0 = get_drag_coef(pos_Re, corrections=corrections)
    beta = 3.7*np.ones_like(pos_Re)
    
    # Apply beta-corrections only if necessary
    mask_beta_correction = pos_Re > 1e-2
    beta[mask_beta_correction] -= 0.65*np.exp(
        -0.5 * (1.5 - np.log10(pos_Re[mask_beta_correction]))**2
    )
    
    if isinstance(volume_fraction, np.ndarray):
        assert Re.shape == volume_fraction.shape, "Re_vec and phi have different shape"
        volume_fraction = volume_fraction[pos_mask]
    
    C_D[pos_mask] = C_D0 * (1 - volume_fraction)**(-beta)
    
    return C_D


def get_drag_coef(Re, corrections):
    """Calculate drag coefficient based on the Schiller and Nauman expression, recommended by Wen and Wu (1966)
    Wen, C.Y. and Wu, Y.H., Mechanics of fluidization, Chem. Eng. Prog. Symp. Ser., 62, 100–125, 1966
    
    This formula is part of the Clif and Gauvin fit 
    [https://doi.org/10.1002/cjce.5450490403]. See vc.get_drag_coef

    Args:
        Re: Reynolds number

    Raises:
        ValueError: if any Reynolds number is negative, raise value error

    Returns:
        numpy array of drag coefficients
    """
    
    C_D = (
        24/Re*(1 + 0.15*Re**0.687)
        # + 0.42/(1 + 42_500/pos_Re**1.16)
    )
    
    if corrections:
        mask_drag_correction = Re > 1e2
        C_D[mask_drag_correction] += (
            0.42/(1 + 42_500/Re[mask_drag_correction]**1.16)
        )
    
    return C_D


