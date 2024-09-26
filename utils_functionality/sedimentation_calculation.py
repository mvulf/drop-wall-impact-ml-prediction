import numpy as np
from . import velocity_calculation as vc
from collections.abc import Iterable


class SedimentationSystem():
    _name = 'SedimentationSystem'
    _system_type = 'diff_eqn'
    
    _state_naming = [
        "particle position [m]",
        "particle velocity [m/s]",
        "particle concentration",
        "concentration change rate [1/s]"
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
                "n_bottom_nodes": 6, # number of nodes to make linear velocity profile
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
        
        z_p, v_p, phi, q_phi = self.get_substates(state)
        
        # # Test only. TODO: Delete further
        # print(f'z_nodes = {z_nodes}')
        # print(f'z_p = {z_p}')
        # print(f'v_p = {v_p}')
        # print(f'phi = {phi}')
        # print(f'q_phi = {q_phi}')
        
        
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
        )
        # # Test only. TODO: Delete further
        # print(f'Re = {Re}')
        # print(f'C_D = {C_D}')
        
        # VELOCITY
        Dz_p = v_p
        # ACCELERATION
        # Get free fall acceleration
        Dv_p = (1 - 1/eps_p)*g*np.ones_like(v_p)
        # Add drag force if velocity is not zero
        mask = (v_p != 0)
        if mask.any():
            Dv_p[mask] -= (
                0.75*v_p[mask]*np.abs(v_p[mask])*C_D[mask]
                /(eps_p*d_p)
            )
        
        # concentration change rate
        Dphi = q_phi
        # "ACCELERATION" of Concentration change rate
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
        # coefs = np.linspace(1, 0, N_BN)
        
        # Check if only last two nodes have zero-velocity
        coefs = 0
        v_p_E[-N_BN:] = v_p_E[-N_BN:]*coefs
        
        # Space differentiation
        Dq_phi = np.zeros_like(q_phi) # NOTE: First node concentration change rate is constant
        Dq_phi[1:] = - (
            (phi[1:]*v_p_E[1:] - phi[:-1]*v_p_E[:-1])
            /np.diff(z_nodes)
        )
        # # TODO: DELETE
        # Dq_phi[0] = Dq_phi[1]
        
        # # TODO: Delete after debug
        # print(f'Dq_phi = {Dq_phi}')
        # print(f'np.diff(z_nodes) = {np.diff(z_nodes)}')
        # print(f'Dz_p shape = {Dz_p.shape}')
        # print(f'Dv_p shape = {Dv_p.shape}')
        # print(f'Dphi shape = {Dphi.shape}')
        # print(f'Dq_phi shape = {Dq_phi.shape}')
        
        
        Dstate = np.hstack(
            (
                Dz_p,
                Dv_p,
                Dphi,
                Dq_phi
            )
        )
        
        # # DELETE. Debug only
        # print(f'State shape = {state.shape}')
        # print(f'Dstate shape = {Dstate.shape}')
        
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
    
    
    def get_substates(self, state, verbose=False):
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
        q_phi = state[2*N_L+N_E:2*N_L+2*N_E]
        
        if verbose:
            print(f'z_p [m] = {z_p}')
            print(f'v_p [m/s] = {v_p}')
            print(f'phi = {phi}')
            print(f'q_phi [1/s] = {q_phi}')
        
        return z_p, v_p, phi, q_phi
        
        

def get_suspension_drag_coef(Re, volume_fraction):
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
    
    C_D0 = vc.get_drag_coef(pos_Re)
    beta = 3.7 - 0.65*np.exp(
        -0.5 * (1.5 - np.log10(pos_Re))**2
    )
    
    if isinstance(volume_fraction, np.ndarray):
        assert Re.shape == volume_fraction.shape, "Re_vec and phi have different shape"
        volume_fraction = volume_fraction[pos_mask]
    
    C_D[pos_mask] = C_D0 * (1 - volume_fraction)**(-beta)
    
    return C_D


