import numpy as np
from dynamical_systems_polished import dlqr
import matplotlib.pyplot as plt
import os
from config import (
    GAMMA,
    M_C,
    M_P,
    L,
    DT,
    C_CART_POLE,
    RHO_CART_POLE,
    U_BOUNDS,
    DEGREE,
)

from polynomial_features import FilteredPolynomialFeatures, StateOnlyPolynomialFeatures
class PolicyExtractor:
    """
    Policy extractor with better numerical stability and optimization
    """
    
    def __init__(self, M_c=None, M_p=None, l=None, dt=None, gamma=GAMMA,
                 degree=DEGREE, u_bounds=None, C=None, rho=None,
                 system_type="cartpole"):
        """
        Initialize policy extractor
        
        Args:
            M_c: cart mass (kg)
            M_p: pole mass (kg) 
            l: half-pole length (m)
            dt: sampling time (s)
            gamma: discount factor
            degree: polynomial feature degree
            u_bounds: control input bounds
            system_type: which dynamical system to assume for defaults
        """
        self.system_type = (system_type or "cartpole").lower()
        if self.system_type == "cartpole":
            default_dt = DT
            default_C = C_CART_POLE
            default_rho = RHO_CART_POLE
            default_u_bounds = U_BOUNDS
            default_state_names = ['Position (m)', 'Velocity (m/s)', 'Angle (rad)', 'Angular Velocity (rad/s)']
            default_state_units = ['m', 'm/s', 'rad', 'rad/s']
            control_label = 'Control Torque (N·m)'
            default_state_limits = [(-5.0, 5.0), (-5.0, 5.0), (-2.0*np.pi, 2.0*np.pi), (-10.0, 10.0)]
        elif self.system_type == "pointmass2d":
            default_dt = DT
            # Default C will be set dynamically based on actual state dimension when system is created
            # For now, use a placeholder that will be overridden by extractor_kwargs
            default_C = None  # Will be set from extractor_kwargs
            default_rho = 1.0
            default_u_bounds = (-8.0, 8.0)  # Force bounds
            # Default state names/units/limits for 4D (2D point mass) - will be generated dynamically for other dimensions
            default_state_names = ['Position x (m)', 'Position y (m)', 'Velocity x (m/s)', 'Velocity y (m/s)']
            default_state_units = ['m', 'm', 'm/s', 'm/s']
            control_label = 'Force (N)'
            # Use test bounds from config to match actual test state sampling range
            from config import P_BOUNDS_POINT_MASS_TEST, V_BOUNDS_POINT_MASS_TEST
            # Default limits for 4D - will be generated dynamically for other dimensions
            default_state_limits = [P_BOUNDS_POINT_MASS_TEST, P_BOUNDS_POINT_MASS_TEST, 
                                   V_BOUNDS_POINT_MASS_TEST, V_BOUNDS_POINT_MASS_TEST]
        else:
            raise ValueError(f"Unsupported system_type '{system_type}'. Supported: 'cartpole', 'pointmass2d'.")
        
        # Store physical parameters when available
        self.M_c = M_c if M_c is not None else (M_C if self.system_type == "cartpole" else None)
        self.M_p = M_p if M_p is not None else (M_P if self.system_type == "cartpole" else None)
        self.l = l if l is not None else (L if self.system_type == "cartpole" else None)
        self.dt = dt if dt is not None else default_dt
        
        # Common parameters
        self.gamma = gamma
        self.degree = degree
        self.u_bounds = u_bounds if u_bounds is not None else default_u_bounds
        
        # LQR parameters (with defaults)
        self.C = C if C is not None else default_C
        self.rho = rho if rho is not None else default_rho
        
        # Numerical stability parameters
        self.regularization = 1e-6
        self.optimization_tolerance = 1e-8
        self.max_optimization_iterations = 1000
        
        # Plotting parameters
        self.plots_base_dir = "figures"
        self.degree = degree
        self._default_state_names = default_state_names
        self._default_state_units = default_state_units
        self._default_state_limits = default_state_limits
        self._control_label = control_label

    def _get_state_metadata(self, n_states):
        """
        Return state names and units tailored to the current system type and dimensionality.
        """
        if self.system_type == "cartpole" and n_states == 4:
            names = self._default_state_names
            units = self._default_state_units
            limits = self._default_state_limits
        elif self.system_type == "pointmass2d":
            # For point mass: state is [p, v] where p,v are n-dimensional
            # So n_states = 2n, meaning n = n_states / 2
            if n_states % 2 != 0:
                # Fallback for invalid dimensions
                names = [f"State {i+1}" for i in range(n_states)]
                units = ['units'] * n_states
                limits = [None] * n_states
            else:
                n = n_states // 2
                # Generate names: Position x, Position y, ..., Velocity x, Velocity y, ...
                names = []
                units = []
                limits = []
                from config import P_BOUNDS_POINT_MASS_TEST, V_BOUNDS_POINT_MASS_TEST
                # Position components
                position_labels = ['x', 'y', 'z'] + [f'{i}' for i in range(4, n+1)]
                for i in range(n):
                    label = position_labels[i] if i < len(position_labels) else f'{i+1}'
                    names.append(f'Position {label} (m)')
                    units.append('m')
                    limits.append(P_BOUNDS_POINT_MASS_TEST)
                # Velocity components
                for i in range(n):
                    label = position_labels[i] if i < len(position_labels) else f'{i+1}'
                    names.append(f'Velocity {label} (m/s)')
                    units.append('m/s')
                    limits.append(V_BOUNDS_POINT_MASS_TEST)
        else:
            names = [f"State {i+1}" for i in range(n_states)]
            units = ['units'] * n_states
            limits = [None] * n_states
        return names, units, limits
        
    def extract_lqr_policy(self, system):
        """
        Extract LQR policy from system.
        
        Supports:
        - LQR systems (dlqr class) with optimal_solution method
        - Nonlinear systems with linearized_system method (e.g., cartpole)
        
        Returns:
            policy_func: function that takes state and returns control action
            K_lqr: LQR gain matrix
            Q_lqr: LQR Q-function matrix
        """
        # Auto-detect system type
        if hasattr(system, 'optimal_solution'):
            # This is an LQR system (dlqr class)
            system_type = 'lqr'
        elif hasattr(system, 'linearized_system'):
            # This is a nonlinear system with linearization capability
            # (e.g., cartpole)
            system_type = 'nonlinear'
        else:
            raise ValueError(f"Unknown system type: {type(system)}")
        
        if system_type == 'lqr':
            # For LQR systems, get the optimal solution directly
            P_lqr, K_lqr, q_lqr = system.optimal_solution()
        elif system_type == 'nonlinear':
            # For nonlinear systems, linearize and solve LQR
            A_d, B_d = system.linearized_system(use_backward_euler=False)
            lqr_system = dlqr(A_d, B_d, self.C, self.rho, self.gamma)
            P_lqr, K_lqr, q_lqr = lqr_system.optimal_solution()
        else:
            raise ValueError(f"Unsupported system type: {system_type}")
        
        # Create policy function
        def lqr_policy(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            u = -K_lqr @ x.T  # Standard LQR control law: u = -K @ x
            u_clipped = u.T  # Shape: (batch, N_u)
            
            # Clip to bounds (handle both scalar and tuple bounds)
            if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                # Scalar bounds: apply to all dimensions
                u_clipped = np.clip(u_clipped, self.u_bounds[0], self.u_bounds[1])
            else:
                # Assume bounds is a list/array of (min, max) for each dimension
                # For now, if it's a tuple, treat as scalar bounds
                u_clipped = np.clip(u_clipped, self.u_bounds[0], self.u_bounds[1])
            
            # Return format: scalar for single control, array for multi-dimensional
            if u_clipped.shape[1] == 1:
                if u_clipped.shape[0] == 1:
                    return float(u_clipped[0, 0])
                else:
                    return u_clipped.flatten()
            else:
                if u_clipped.shape[0] == 1:
                    return u_clipped[0, :]
                else:
                    return u_clipped
        
        return lqr_policy, K_lqr, P_lqr
    
    def _create_plot_directory(self, N_samples):
        """
        Create directory structure for plots based on degree and sample count.
        
        Args:
            N_samples: Number of samples used in the experiment
            
        Returns:
            str: Path to the created directory
        """
        # Create directory structure: figures/degree_{degree}/N_{N_samples}/
        degree_dir = os.path.join(self.plots_base_dir, f"degree_{self.degree}")
        sample_dir = os.path.join(degree_dir, f"N_{N_samples}")
        
        # Create directories if they don't exist
        os.makedirs(sample_dir, exist_ok=True)
        
        return sample_dir
    
    def _generate_plot_filename(self, N_samples, plot_type, suffix=""):
        """
        Generate standardized plot filename.
        
        Args:
            N_samples: Number of samples used
            plot_type: Type of plot (e.g., 'control_comparison', 'trajectory_comparison')
            suffix: Optional suffix for the filename
            
        Returns:
            str: Full path to the plot file
        """
        plot_dir = self._create_plot_directory(N_samples)
        
        # Create filename: {plot_type}_degree_{degree}_N_{N_samples}{suffix}.png
        filename = f"{plot_type}_degree_{self.degree}_N_{N_samples}{suffix}.png"
        
        return os.path.join(plot_dir, filename)

    def extract_policy_grid_search(self, system, Q_matrix, degree=None, dx=None, du=None, n_grid_points=100):
        """
        Extract policy using grid search - serves as baseline for verification
        
        Args:
            system: dynamical system
            Q_matrix: Q-matrix from moment matching
            degree: polynomial feature degree
            dx: state dimension
            du: control dimension
            n_grid_points: number of grid points for action search
            
        Returns:
            policy function that takes states and returns optimal actions
        """
        if degree is None:
            degree = self.degree
        if dx is None or du is None:
            # Infer from system or Q_matrix
            if hasattr(system, 'N_x') and hasattr(system, 'N_u'):
                dx = system.N_x
                du = system.N_u
            else:
                raise ValueError("dx and du must be provided or system must have N_x and N_u attributes")
        
        # Create polynomial feature transformer
        poly = StateOnlyPolynomialFeatures(degree=degree, include_bias=False, dx=dx, du=du)
        # Fit on dummy data to initialize (will be used for transform only)
        dummy_data = np.zeros((1, dx + du))
        poly.fit(dummy_data)
        
        def policy_grid_search(x):
            x = np.atleast_2d(x)
            n = x.shape[0]
            actions = np.empty((n, 1), dtype=float)

            for i in range(n):
                state = x[i]

                def q_function(u_val):
                    """Evaluate Q-function for given action"""
                    try:
                        # Create state-action pair
                        state_action = np.concatenate([state, [u_val]]).reshape(1, -1)
                        phi = poly.transform(state_action)
                        
                        # Evaluate Q-function
                        q_value = phi @ Q_matrix @ phi.T
                        return float(q_value[0, 0])
                    except:
                        return np.inf

                # Grid search over action space
                u_candidates = np.linspace(self.u_bounds[0], self.u_bounds[1], n_grid_points)
                q_values = []
                
                for u in u_candidates:
                    q_val = q_function(u)
                    q_values.append(q_val)
                
                # Find optimal action
                best_idx = np.argmin(q_values)
                actions[i, 0] = u_candidates[best_idx]

            # For single control systems, return scalars for single states, 1D arrays for multiple states
            if actions.shape[1] == 1:
                if len(actions) == 1:
                    # Single state - return scalar
                    return float(actions[0])
                else:
                    # Multiple states - return 1D array
                    return actions.flatten()
            else:
                return actions

        return policy_grid_search

    def extract_policy_scipy_optimization(self, system, Q_matrix, degree=None, dx=None, du=None):
        """
        Extract policy using scipy optimization - more efficient than grid search
        """
        from scipy.optimize import minimize_scalar, minimize
        
        if degree is None:
            degree = self.degree
        if dx is None or du is None:
            # Infer from system or Q_matrix
            if hasattr(system, 'N_x') and hasattr(system, 'N_u'):
                dx = system.N_x
                du = system.N_u
            else:
                raise ValueError("dx and du must be provided or system must have N_x and N_u attributes")
        
        # Create polynomial feature transformer
        poly = StateOnlyPolynomialFeatures(degree=degree, include_bias=False, dx=dx, du=du)
        # Fit on dummy data to initialize (will be used for transform only)
        dummy_data = np.zeros((1, dx + du))
        poly.fit(dummy_data)
        
        def policy_scipy_optimization(x):
            x = np.atleast_2d(x)
            n = x.shape[0]
            actions = np.empty((n, 1), dtype=float)

            for i in range(n):
                state = x[i]

                def q_function(u_val):
                    """Evaluate Q-function for given action"""
                    try:
                        # Create state-action pair
                        state_action = np.concatenate([state, [u_val]]).reshape(1, -1)
                        phi = poly.transform(state_action)
                        
                        # Evaluate Q-function
                        q_value = phi @ Q_matrix @ phi.T
                        return float(q_value[0, 0])
                    except:
                        return np.inf

                # Use scipy optimization
                try:
                    result = minimize_scalar(
                        q_function, 
                        bounds=(self.u_bounds[0], self.u_bounds[1]),
                        method='bounded',
                    )
                    
                    if result.success:
                        actions[i, 0] = result.x
                    else:
                        # Fallback to grid search
                        print("Fallback to grid search")
                        u_candidates = np.linspace(self.u_bounds[0], self.u_bounds[1], 50)
                        q_values = [q_function(u) for u in u_candidates]
                        best_idx = np.argmin(q_values)
                        actions[i, 0] = u_candidates[best_idx]
                        
                except:
                    # Fallback to grid search
                    print("Fallback to grid search")
                    u_candidates = np.linspace(self.u_bounds[0], self.u_bounds[1], 50)
                    q_values = [q_function(u) for u in u_candidates]
                    best_idx = np.argmin(q_values)
                    actions[i, 0] = u_candidates[best_idx]

            # For single control systems, return scalars for single states, 1D arrays for multiple states
            if actions.shape[1] == 1:
                if len(actions) == 1:
                    # Single state - return scalar
                    return float(actions[0])
                else:
                    # Multiple states - return 1D array
                    return actions.flatten()
            else:
                return actions

        return policy_scipy_optimization
    
    def extract_moment_matching_policy_analytical(self, system, Q_matrix, degree=None, dx=None, du=None, debug=False, debug_max_print=10):
        """
        Analytical policy extraction using gradient-based approach
        
        This method computes the optimal action analytically when possible,
        falling back to numerical optimization when needed.
        
        Args:
            system: dynamical system
            Q_matrix: Q-matrix from moment matching
            degree: polynomial feature degree
            dx: state dimension
            du: control dimension
        """
        if degree is None:
            degree = self.degree
        if dx is None or du is None:
            # Infer from system
            if hasattr(system, 'N_x') and hasattr(system, 'N_u'):
                dx = system.N_x
                du = system.N_u
            else:
                raise ValueError("dx and du must be provided or system must have N_x and N_u attributes")
        
        # Create polynomial feature transformer (always excludes u^2 terms)
        poly = StateOnlyPolynomialFeatures(degree=degree, include_bias=False, dx=dx, du=du)
        # Fit on dummy data to initialize (will be used for transform only)
        dummy_data = np.zeros((1, dx + du))
        poly.fit(dummy_data)
        
        # Check if we can use the analytical solution for StateOnlyPolynomialFeatures
        # This works for any degree and any du, as long as features are [poly(x), u]
        use_stateonly_analytical = isinstance(poly, StateOnlyPolynomialFeatures)
        
        n_phi = poly.n_output_features_
        assert Q_matrix.shape == (n_phi, n_phi), (
            f"Q_matrix shape {Q_matrix.shape} != ({n_phi},{n_phi})"
        )
        
        # Diagnostics state (used only if debug=True)
        debug_ctx = {
            "enabled": bool(debug),
            "max_print": int(debug_max_print) if debug_max_print is not None else 0,
            "printed": 0,
            "n_calls": 0,
            "n_clipped": 0,
        }
        
        def mm_policy_analytical(x):
            x = np.atleast_2d(x)
            n = x.shape[0]
            actions = np.empty((n, du), dtype=float)
            debug_ctx["n_calls"] += n

            for i in range(n):
                state = x[i]

                try:
                    # For polynomial features, we can compute the gradient analytically
                    # The objective is: min_u phi(x,u)^T Q phi(x,u)
                    # where phi(x,u) = [x, u, x^2, xu, u^2, ...]
                    
                    # Try analytical solution first
                    if degree == 1:
                        # For linear features with StateOnlyPolynomialFeatures: phi = [x1, x2, ..., x_dx, u1, u2, ..., u_du]
                        # For FilteredPolynomialFeatures: same structure
                        # Both can use the same analytical solution
                        if use_stateonly_analytical:
                            # Use the general StateOnlyPolynomialFeatures analytical solver
                            u_opt = self._solve_stateonly_polynomial_policy_analytical(
                                state, Q_matrix, poly, dx, du, debug=debug_ctx["enabled"], debug_ctx=debug_ctx
                            )
                            if u_opt is not None and np.isfinite(u_opt).all():
                                actions[i, :] = u_opt
                            else:
                                # Fallback to direct computation
                                n_poly_features = poly.poly_x.n_output_features_
                                Q_xu = Q_matrix[:n_poly_features, n_poly_features:]
                                Q_uu = Q_matrix[n_poly_features:, n_poly_features:]
                                phi_x = poly.poly_x.transform(state.reshape(1, -1)).flatten()
                                try:
                                    u_opt = -np.linalg.solve(Q_uu, Q_xu.T @ phi_x)
                                    u_opt = np.clip(u_opt, self.u_bounds[0], self.u_bounds[1])
                                    actions[i, :] = u_opt
                                except:
                                    actions[i, :] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                        else:
                            # For FilteredPolynomialFeatures: features are [x1, x2, ..., x_dx, u1, u2, ..., u_du]
                            # So Q_xx is [:dx, :dx], Q_xu is [:dx, dx:dx+du], Q_uu is [dx:dx+du, dx:dx+du]
                            Q_xx = Q_matrix[:dx, :dx]
                            Q_xu = Q_matrix[:dx, dx:dx+du]
                            Q_uu = Q_matrix[dx:dx+du, dx:dx+du]

                            # Optimal u: ∂/∂u = 0 → 2Q_xu^T x + 2Q_uu u = 0 → u = -Q_uu^{-1} Q_xu^T x
                            try:
                                u_opt = -np.linalg.solve(Q_uu, Q_xu.T @ state)
                                u_opt = np.asarray(u_opt).reshape(-1)
                            except np.linalg.LinAlgError:
                                # Use regularized solve for ill-conditioned matrices
                                cond_Q_uu = np.linalg.cond(Q_uu)
                                print(f"Warning: Q_uu condition number {cond_Q_uu:.2e}, using regularized solve")
                                reg = 1e-6 * np.trace(Q_uu) / Q_uu.shape[0]
                                u_opt = -np.linalg.solve(Q_uu + reg * np.eye(Q_uu.shape[0]), Q_xu.T @ state)
                                u_opt = np.asarray(u_opt).reshape(-1)
                            
                            # Clip to bounds (handle both scalar and tuple bounds)
                            if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                                # Scalar bounds: apply to all dimensions
                                u_opt = np.clip(u_opt, self.u_bounds[0], self.u_bounds[1])
                            else:
                                # Assume bounds is a list/array of (min, max) for each dimension
                                # For now, if it's a tuple, treat as scalar bounds
                                u_opt = np.clip(u_opt, self.u_bounds[0], self.u_bounds[1])
                            
                            actions[i, :] = u_opt
                        continue
                    
                    elif degree == 2:
                        # For degree 2 polynomial features: phi = [x, u, x^2, xu, u^2, ...]
                        # For multi-dimensional control, we need to optimize over vector u
                        if use_stateonly_analytical:
                            # Use analytical solution for StateOnlyPolynomialFeatures (works for any du)
                            u_opt = self._solve_stateonly_polynomial_policy_analytical(
                                state, Q_matrix, poly, dx, du, debug=debug_ctx["enabled"], debug_ctx=debug_ctx
                            )
                            if u_opt is not None and np.isfinite(u_opt).all():
                                actions[i, :] = u_opt
                            else:
                                # Fallback to numerical optimization
                                u_opt = self._solve_multidimensional_policy(state, Q_matrix, poly, dx, du)
                                if u_opt is not None and np.isfinite(u_opt).all():
                                    actions[i, :] = u_opt
                                else:
                                    # Fallback to center of action bounds
                                    if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                                        actions[i, :] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                                    else:
                                        actions[i, :] = 0.0
                        elif du == 1:
                            # Scalar control: use analytical solution for FilteredPolynomialFeatures
                            u_opt = self._solve_quadratic_policy_analytical(state, Q_matrix, poly, dx, du)
                            if u_opt is not None and np.isfinite(u_opt):
                                actions[i, 0] = u_opt
                            else:
                                # Fallback to center of action bounds
                                print(f"Fallback to center of action bounds")
                                actions[i, 0] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                        else:
                            # Multi-dimensional control: use numerical optimization
                            u_opt = self._solve_multidimensional_policy(state, Q_matrix, poly, dx, du)
                            if u_opt is not None and np.isfinite(u_opt).all():
                                actions[i, :] = u_opt
                            else:
                                # Fallback to center of action bounds
                                print(f"Fallback to center of action bounds")
                                if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                                    actions[i, :] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                                else:
                                    actions[i, :] = 0.0
                        continue

                    elif degree > 2:
                        # For higher degrees, try analytical solution if using StateOnlyPolynomialFeatures
                        if use_stateonly_analytical:
                            u_opt = self._solve_stateonly_polynomial_policy_analytical(
                                state, Q_matrix, poly, dx, du, debug=debug_ctx["enabled"], debug_ctx=debug_ctx
                            )
                            if u_opt is not None and np.isfinite(u_opt).all():
                                actions[i, :] = u_opt
                            else:
                                # Fallback to numerical optimization
                                u_opt = self._solve_multidimensional_policy(state, Q_matrix, poly, dx, du)
                                if u_opt is not None and np.isfinite(u_opt).all():
                                    actions[i, :] = u_opt
                                else:
                                    # Fallback to center of action bounds
                                    if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                                        actions[i, :] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                                    else:
                                        actions[i, :] = 0.0
                        else:
                            # For higher degrees, use numerical optimization
                            u_opt = self._solve_multidimensional_policy(state, Q_matrix, poly, dx, du)
                            if u_opt is not None and np.isfinite(u_opt).all():
                                actions[i, :] = u_opt
                            else:
                                # Fallback to center of action bounds
                                print(f"Fallback to center of action bounds")
                                if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                                    actions[i, :] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                                else:
                                    actions[i, :] = 0.0
                        continue
                    
                    print(f"Analytical optimization failed for state {i}, using fallback")
                    if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                        actions[i, :] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                    else:
                        actions[i, :] = 0.0
                        
                except Exception as e:
                    print(f"Exception in analytical optimization for state {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    actions[i, 0] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])

            # Return format: scalar for single state + scalar control, array otherwise
            if du == 1:
                if len(actions) == 1:
                    # Single state, scalar control - return scalar
                    return float(actions[0, 0])
                else:
                    # Multiple states, scalar control - return 1D array
                    return actions.flatten()
            else:
                # Multi-dimensional control
                if len(actions) == 1:
                    # Single state, vector control - return 1D array
                    return actions[0, :]
                else:
                    # Multiple states, vector control - return 2D array
                    return actions

        return mm_policy_analytical
    
    
    def _solve_multidimensional_policy(self, state, Q_matrix, poly, dx, du):
        """
        Solve for optimal multi-dimensional control u using numerical optimization.
        
        Args:
            state: state vector
            Q_matrix: Q-matrix from moment matching
            poly: FilteredPolynomialFeatures transformer
            dx: state dimension
            du: control dimension
            
        Returns:
            optimal u vector or None if optimization fails
        """
        from scipy.optimize import minimize
        
        # Get bounds for optimization
        if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
            # Scalar bounds: apply to all dimensions
            bounds = [(self.u_bounds[0], self.u_bounds[1])] * du
        else:
            # Assume bounds is a list/array of (min, max) for each dimension
            # For now, if it's a tuple, treat as scalar bounds
            bounds = [(self.u_bounds[0], self.u_bounds[1])] * du
        
        def q_function(u_vec):
            """Evaluate Q-function for given control vector"""
            try:
                u_vec = np.asarray(u_vec).reshape(-1)
                if u_vec.size != du:
                    return np.inf
                # Create state-action pair
                state_action = np.concatenate([state, u_vec]).reshape(1, -1)
                phi = poly.transform(state_action)
                
                # Evaluate Q-function
                q_value = phi @ Q_matrix @ phi.T
                return float(q_value[0, 0])
            except:
                return np.inf
        
        # Initial guess: center of bounds
        u0 = np.array([0.5 * (b[0] + b[1]) for b in bounds])
        
        try:
            result = minimize(
                q_function,
                u0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                return np.asarray(result.x).reshape(-1)
            else:
                # Try with different initial guess or method
                result = minimize(
                    q_function,
                    u0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-8}
                )
                if result.success:
                    return np.asarray(result.x).reshape(-1)
        except:
            pass
        
        # Fallback: grid search over a coarse grid
        try:
            n_grid = 10
            if du == 2:
                # 2D grid search
                u1_candidates = np.linspace(bounds[0][0], bounds[0][1], n_grid)
                u2_candidates = np.linspace(bounds[1][0], bounds[1][1], n_grid)
                best_val = np.inf
                best_u = None
                for u1 in u1_candidates:
                    for u2 in u2_candidates:
                        val = q_function([u1, u2])
                        if val < best_val:
                            best_val = val
                            best_u = np.array([u1, u2])
                return best_u
            else:
                # For higher dimensions, use random search
                best_val = np.inf
                best_u = None
                for _ in range(100):
                    u_candidate = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
                    val = q_function(u_candidate)
                    if val < best_val:
                        best_val = val
                        best_u = u_candidate
                return best_u
        except:
            return None
    
    def _solve_stateonly_polynomial_policy_analytical(self, state, Q_matrix, poly, dx, du, debug=False, debug_ctx=None):
        """
        Solve for optimal multi-dimensional control u analytically for StateOnlyPolynomialFeatures.
        
        This method exploits the structure where φ(x,u) = [φ_x(x), u], where φ_x(x) are
        polynomial features of x only, and u is appended unchanged.
        
        The Q-function is: Q(x,u) = φ(x,u)ᵀ Q φ(x,u)
        
        Partitioning Q into blocks:
        - Q_xx: state features × state features
        - Q_xu: state features × control features  
        - Q_ux: control features × state features (Q_xuᵀ)
        - Q_uu: control features × control features
        
        Then: Q(x,u) = φ_x(x)ᵀ Q_xx φ_x(x) + 2 φ_x(x)ᵀ Q_xu u + uᵀ Q_uu u
        
        Taking gradient w.r.t. u and setting to zero:
        ∇_u Q(x,u) = 2 Q_xuᵀ φ_x(x) + 2 Q_uu u = 0
        → u* = -Q_uu^{-1} Q_xuᵀ φ_x(x)
        
        Args:
            state: state vector of shape (dx,)
            Q_matrix: Q-matrix from moment matching, shape (n_poly_features + du, n_poly_features + du)
            poly: StateOnlyPolynomialFeatures transformer
            dx: state dimension
            du: control dimension
            
        Returns:
            optimal u vector of shape (du,) or None if solution fails
        """
        x = np.asarray(state).reshape(-1)
        
        # Get number of polynomial features for states
        n_poly_features = poly.poly_x.n_output_features_
        
        # Partition Q-matrix into blocks
        # Q = [[Q_xx, Q_xu], [Q_ux, Q_uu]]
        Q_xx = Q_matrix[:n_poly_features, :n_poly_features]
        Q_xu = Q_matrix[:n_poly_features, n_poly_features:]
        Q_ux = Q_matrix[n_poly_features:, :n_poly_features]  # Should be Q_xu.T
        Q_uu = Q_matrix[n_poly_features:, n_poly_features:]
        
        # Compute polynomial features of state only
        phi_x = poly.poly_x.transform(x.reshape(1, -1))  # Shape: (1, n_poly_features)
        phi_x = phi_x.flatten()  # Shape: (n_poly_features,)
        
        # Analytical solution: u* = -Q_uu^{-1} Q_xu^T φ_x(x)
        try:
            # Compute Q_xu^T φ_x(x)
            Q_xu_T_phi_x = Q_xu.T @ phi_x  # Shape: (du,)
            
            # Solve Q_uu u = -Q_xu^T φ_x(x)
            # Check condition number for numerical stability
            cond_Q_uu = np.linalg.cond(Q_uu)
            used_reg = False
            if cond_Q_uu > 1e12:
                # Use regularized solve for ill-conditioned matrices
                reg = 1e-6 * np.trace(Q_uu) / Q_uu.shape[0]
                used_reg = True
                u_opt = -np.linalg.solve(Q_uu + reg * np.eye(Q_uu.shape[0]), Q_xu_T_phi_x)
            else:
                u_opt = -np.linalg.solve(Q_uu, Q_xu_T_phi_x)
            
            u_free = np.asarray(u_opt).reshape(-1)
            
            # Clip to bounds
            if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                u_clipped = np.clip(u_free, self.u_bounds[0], self.u_bounds[1])
            else:
                # Assume bounds is a list/array of (min, max) for each dimension
                u_clipped = np.clip(u_free, self.u_bounds[0], self.u_bounds[1])
            
            # Diagnostics (print only first few times to avoid spam)
            if debug and debug_ctx is not None:
                debug_ctx["n_clipped"] += int(not np.allclose(u_free, u_clipped, atol=1e-10, rtol=1e-10))
                if debug_ctx["printed"] < debug_ctx["max_print"]:
                    try:
                        eigs = np.linalg.eigvalsh(Q_uu)
                        eig_min = float(np.min(eigs))
                        eig_max = float(np.max(eigs))
                    except Exception:
                        eig_min, eig_max = float("nan"), float("nan")
                    print(
                        "[MM debug][StateOnly analytic] "
                        f"state={np.asarray(state).reshape(-1)} | "
                        f"||phi_x||={np.linalg.norm(phi_x):.3e} | "
                        f"rhs=Q_xu^T phi_x={Q_xu_T_phi_x} | "
                        f"cond(Q_uu)={cond_Q_uu:.3e} | eig(Q_uu)∈[{eig_min:.3e},{eig_max:.3e}] | "
                        f"used_reg={used_reg} | "
                        f"u_free={u_free} -> u_clipped={u_clipped} | "
                        f"clipped={not np.allclose(u_free, u_clipped, atol=1e-10, rtol=1e-10)} | "
                        f"clip_rate_so_far={debug_ctx['n_clipped']}/{max(1, debug_ctx['n_calls'])}"
                    )
                    debug_ctx["printed"] += 1
            
            return u_clipped
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # If solve fails, return None to trigger fallback
            return None
    
    def _solve_quadratic_policy_analytical(self, state, Q_matrix, poly, dx, du):
        """
        Solve for optimal u analytically for degree 2 polynomial features.
        
        For degree 2 features φ(x,u) = [x, u, x², xu, u², ...], the objective
        φ(x,u)ᵀ Q φ(x,u) can be expressed as a polynomial in u.
        
        The analytical approach:
        1. Express φ(x,u) = φ₀ + φ₁*u + φ₂*u² where:
           - φ₀: features independent of u (x, x², cross terms)
           - φ₁: features linear in u (xu terms)  
           - φ₂: features quadratic in u (u² term)
        2. The objective becomes: (φ₀ + φ₁*u + φ₂*u²)ᵀ Q (φ₀ + φ₁*u + φ₂*u²)
        3. This expands to: a₄*u⁴ + a₃*u³ + a₂*u² + a₁*u + a₀
        4. Find critical points by solving: 4*a₄*u³ + 3*a₃*u² + 2*a₂*u + a₁ = 0
        
        Args:
            state: state vector
            Q_matrix: Q-matrix from moment matching
            poly: FilteredPolynomialFeatures transformer
            dx: state dimension
            du: control dimension
            
        Returns:
            optimal u or None if analytical solution fails
        """
        x = np.asarray(state).reshape(-1)
        umin, umax = self.u_bounds

        powers = poly.powers_  # shape (D, n_inputs): exponents for [x1,...,xn,u1,...,um]
        D = powers.shape[0]
        
        assert Q_matrix.shape == (D, D)

        # For each feature j: phi_j(x,u) = c_j(x) * u^{e_j}
        # c_j depends only on x (product of x_i^power), e_j is the power on u.
        c = np.ones(D, dtype=float)
        # For multi-dimensional control, we need to handle the control exponents differently
        # For now, assume we're optimizing over the first control variable
        if du == 1:
            e = powers[:, dx].astype(int)  # exponents of u: 0,1,2 for degree 2
        else:
            # For multi-dimensional control, use the first control variable's exponent
            e = powers[:, dx].astype(int)  # exponents of u1: 0,1,2 for degree 2
            
        for j in range(D):
            # multiply x_i^{p_ji} for i=0..dx-1
            for i in range(dx):
                p = powers[j, i]
                if p:
                    c[j] *= x[i] ** p

        # Assemble quartic coefficients a_m = sum_{j,k: e_j+e_k=m} Q_jk c_j c_k
        # m in {0,1,2,3,4}
        a = np.zeros(5, dtype=float)
        for j in range(D):
            for k in range(D):
                m = e[j] + e[k]
                a[m] += Q_matrix[j, k] * c[j] * c[k]
        a0, a1, a2, a3, a4 = a
        # print(f"a0: {a0}, a1: {a1}, a2: {a2}, a3: {a3}, a4: {a4}")

        # Derivative cubic: 4 a4 u^3 + 3 a3 u^2 + 2 a2 u + a1 = 0
        # Handle degenerate lower-degree cases robustly
        coeffs = np.array([4*a4, 3*a3, 2*a2, a1], dtype=float)
        # print(f"coeffs: {coeffs}")

        # Candidate set = stationary points + the two bounds
        candidates = set([umin, umax])  # Use set to avoid duplicates

        # If the polynomial is actually lower degree, trim leading zeros
        # Find first non-zero coefficient
        nonzero_mask = np.abs(coeffs) > 1e-12
        if np.any(nonzero_mask):
            first_nz = np.argmax(nonzero_mask)
            try:
                # Only solve if we have at least 2 coefficients (degree >= 1)
                if len(coeffs[first_nz:]) >= 2:
                    roots = np.roots(coeffs[first_nz:])  # closed-form cubic/quadratic/linear solver
                    for r in roots:
                        # Handle numerical precision: check if root is effectively real
                        # (imaginary part is negligible compared to real part)
                        if np.iscomplexobj(r):
                            # Check if imaginary part is negligible
                            if abs(np.imag(r)) < 1e-10 * abs(np.real(r)) or abs(np.imag(r)) < 1e-10:
                                u = float(np.real(r))
                            else:
                                continue  # Skip complex roots
                        elif np.isreal(r) and not np.isnan(np.real(r)):
                            u = float(np.real(r))
                        else:
                            continue  # Skip invalid roots
                        
                        # Check if root is within bounds (with small tolerance for numerical precision)
                        tolerance = 1e-8 * (umax - umin) if umax > umin else 1e-8
                        if (umin - tolerance) <= u <= (umax + tolerance):
                            # Clip to exact bounds if slightly outside due to numerical precision
                            u_clipped = np.clip(u, umin, umax)
                            candidates.add(u_clipped)
            except (np.linalg.LinAlgError, ValueError):
                # If root finding fails, just use bounds
                pass

        # Convert set to sorted list for consistent evaluation
        candidates = sorted(list(candidates))

        # Evaluate objective J(u) = a4 u^4 + a3 u^3 + a2 u^2 + a1 u + a0
        def J(u):
            try:
                return (((a4*u + a3)*u + a2)*u + a1)*u + a0  # Correct Horner's method for quartic
            except (OverflowError, ValueError):
                return np.inf
        
        # Second derivative: J''(u) = 12*a4*u^2 + 6*a3*u + 2*a2
        # Used to check if a stationary point is a minimum (J'' > 0) or maximum (J'' < 0)
        def J_second_derivative(u):
            try:
                return (12*a4*u + 6*a3)*u + 2*a2
            except (OverflowError, ValueError):
                return np.nan

        # Evaluate all candidates and find minimum
        vals = []
        valid_candidates = []
        candidate_types = []  # Track if candidate is a minimum, maximum, or boundary
        
        for u in candidates:
            try:
                val = J(u)
                if not np.isfinite(val):
                    continue  # Skip invalid values
                
                # Determine candidate type
                if u == umin or u == umax:
                    candidate_type = 'boundary'
                else:
                    # Check second derivative at stationary point
                    Jpp = J_second_derivative(u)
                    if np.isfinite(Jpp):
                        if Jpp > 1e-10:  # Positive second derivative -> minimum
                            candidate_type = 'minimum'
                        elif Jpp < -1e-10:  # Negative second derivative -> maximum
                            candidate_type = 'maximum'
                        else:  # Near zero -> saddle point or inflection
                            candidate_type = 'saddle'
                    else:
                        candidate_type = 'unknown'
                
                vals.append(val)
                valid_candidates.append(u)
                candidate_types.append(candidate_type)
            except:
                continue  # Skip candidates that cause errors

        if not valid_candidates:
            return None
            
        # Find minimum among all candidates
        # Prefer stationary points that are minima over boundaries, but always pick global minimum
        min_idx = int(np.argmin(vals))
        u_opt_scaled = valid_candidates[min_idx]
        
        # Optional: If the minimum is at a boundary and there's a nearby minimum, check if it's better
        # This handles edge cases where numerical precision might miss a slightly better minimum
        if candidate_types[min_idx] == 'boundary':
            # Check if there are any minima nearby that might be better
            for i, (u, val, ctype) in enumerate(zip(valid_candidates, vals, candidate_types)):
                if ctype == 'minimum' and abs(u - u_opt_scaled) < 0.01 * (umax - umin):
                    # If there's a minimum very close to the boundary, prefer it if it's better
                    if val < vals[min_idx] + 1e-10:
                        min_idx = i
                        u_opt_scaled = u
                        break
        
        # No scaling, so u_opt is already in original space
        u_opt = u_opt_scaled
        return u_opt
    
    
    def _simulate_policy(self, system, policy, initial_state, horizon=10000, wrap_angles=False):
        """
        Simulate a policy and compute total cost with error handling
        
        Args:
            system: dynamical system
            policy: policy function
            initial_state: initial state
            horizon: simulation horizon
            wrap_angles: if True, wrap angles to [-pi, pi) for pendulum systems
            
        Returns:
            trajectory: state trajectory
            total_cost: total discounted cost
            success: whether simulation was stable
        """
        x_current = initial_state.copy()
        trajectory = [x_current.copy()]
        inputs = []
        total_cost = 0.0
        success = True
        
        for step in range(horizon):
            # Get control action with error handling
            try:
                u_current = policy(x_current)
                if u_current is None or not np.isfinite(u_current).all():
                    print(f"u_current is None or not np.isfinite(u_current).all()")
                    success = False
                    break
            except Exception as e:
                print(f"Policy evaluation failed at step {step}: {e}")
                success = False
                break
            
            # Convert u_current to numpy array and ensure correct shape
            u_current = np.asarray(u_current).reshape(-1)
            N_u = system.N_u if hasattr(system, 'N_u') else 1
            
            # Handle scalar vs vector control
            if N_u == 1:
                # For scalar control, extract scalar value
                if u_current.size > 0:
                    u_current_scalar = float(u_current[0])
                elif hasattr(u_current, 'item'):
                    u_current_scalar = u_current.item()
                else:
                    u_current_scalar = float(u_current)
                inputs.append(u_current_scalar)
                u_for_cost = np.array([[u_current_scalar]])
                u_for_step = u_current_scalar
            else:
                # For multi-dimensional control, keep as array
                if u_current.size != N_u:
                    # If wrong size, pad or truncate
                    if u_current.size > N_u:
                        u_current = u_current[:N_u]
                    else:
                        u_padded = np.zeros(N_u)
                        u_padded[:u_current.size] = u_current
                        u_current = u_padded
                inputs.append(u_current.copy())
                u_for_cost = u_current.reshape(1, -1)
                u_for_step = u_current
            
            # Check for invalid control values
            if not np.isfinite(u_current).all():
                print(f"u_current is not finite: {u_current}")
                success = False
                break
            
            # Compute cost
            try:
                cost = system.cost(x_current.reshape(1, -1), u_for_cost)
                # if step % 500 == 0:
                #     print(f"Step {step} - Cost: {cost}, total cost: {total_cost}")
                #     # print(f"x_current: {x_current}, u_current: {u_current}")
                if not np.isfinite(cost):
                    print(f"cost is not finite: {cost}")
                    success = False
                    break
                total_cost += (self.gamma ** step) * cost
            except Exception:
                print(f"Exception in cost computation")
                success = False
                break
            
            # Update state
            try:
                # Pass wrap_angles parameter to step() function
                # wrap_angles=True for MM/ID, wrap_angles=False for LQR
                x_current = system.step(x_current, u_for_step, wrap_angles=wrap_angles)
                trajectory.append(x_current.copy())
            except Exception:
                print(f"Exception in state update")
                success = False
                break
            
            # General divergence check (optional, can be enabled if needed)
            # if np.any(np.abs(x_current) > 50):
            #     success = False
            #     total_cost = np.inf
            #     break
        # Check stability based on last 100 samples if we have enough data
        if success and len(trajectory) >= 100:
            # Get the last 100 states
            last_100_states = np.array(trajectory[-100:])
            
            # Check if all last 100 states are sufficiently close to equilibrium (origin)
            equilibrium_tolerance = 0.5  # Adjust this threshold as needed
            distances_to_equilibrium = np.linalg.norm(last_100_states, axis=1)
            
            # If any of the last 100 states are too far from equilibrium, mark as unstable
            if not np.all(distances_to_equilibrium <= equilibrium_tolerance):
                success = False
                total_cost = np.inf
        
        # print("Total cost: ", total_cost)
        
        return np.array(trajectory), total_cost, success, inputs
    
    def plot_policy_comparison(self, lqr_inputs, mm_inputs, N_samples, save_path=None):
        """
        Plot control input comparison and save to structured directory
        
        Args:
            lqr_inputs: LQR control inputs - list or array, can be (T,) for scalar or list of arrays for multi-dimensional
            mm_inputs: Moment matching control inputs - list or array, can be (T,) for scalar or list of arrays for multi-dimensional
            N_samples: Number of samples used in the experiment
            save_path: Optional custom path to save the plot (overrides default structure)
        """
        # Convert inputs to arrays and determine control dimension
        lqr_inputs_arr = np.asarray(lqr_inputs)
        mm_inputs_arr = np.asarray(mm_inputs)
        
        # Handle list of arrays (multi-dimensional control)
        if isinstance(lqr_inputs, list) and len(lqr_inputs) > 0:
            if isinstance(lqr_inputs[0], np.ndarray) and lqr_inputs[0].ndim == 1:
                # List of 1D arrays - convert to 2D array
                lqr_inputs_arr = np.array([np.asarray(u).reshape(-1) for u in lqr_inputs])
            elif not isinstance(lqr_inputs[0], np.ndarray):
                # List of scalars - convert to 1D array
                lqr_inputs_arr = np.asarray(lqr_inputs)
        
        if isinstance(mm_inputs, list) and len(mm_inputs) > 0:
            if isinstance(mm_inputs[0], np.ndarray) and mm_inputs[0].ndim == 1:
                # List of 1D arrays - convert to 2D array
                mm_inputs_arr = np.array([np.asarray(u).reshape(-1) for u in mm_inputs])
            elif not isinstance(mm_inputs[0], np.ndarray):
                # List of scalars - convert to 1D array
                mm_inputs_arr = np.asarray(mm_inputs)
        
        # Determine control dimension
        if lqr_inputs_arr.ndim == 1:
            du = 1
        else:
            du = lqr_inputs_arr.shape[1] if lqr_inputs_arr.shape[1] > 1 else 1
        
        # Create subplots: one for each control dimension
        if du == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        else:
            n_cols = min(2, du)
            n_rows = int(np.ceil(du / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = np.atleast_1d(axes).flatten()
        
        # Convert time steps to actual time using DT
        time_lqr = np.arange(len(lqr_inputs_arr)) * self.dt
        time_mm = np.arange(len(mm_inputs_arr)) * self.dt
        
        # Plot each control dimension
        for dim in range(du):
            ax = axes[dim] if du > 1 else axes[0]
            
            # Extract this dimension's inputs
            if du == 1:
                lqr_dim = lqr_inputs_arr if lqr_inputs_arr.ndim == 1 else lqr_inputs_arr[:, 0]
                mm_dim = mm_inputs_arr if mm_inputs_arr.ndim == 1 else mm_inputs_arr[:, 0]
            else:
                lqr_dim = lqr_inputs_arr[:, dim]
                mm_dim = mm_inputs_arr[:, dim]
            
            # Plot control inputs
            ax.plot(time_lqr, lqr_dim, 'b-', label='LQR', linewidth=2, alpha=0.8)
            ax.plot(time_mm, mm_dim, 'r-', label='Moment Matching', linewidth=2, alpha=0.8)
            
            # Formatting with proper units
            ax.set_xlabel('Time (s)')
            if du == 1:
                ax.set_ylabel(self._control_label)
                ax.set_title('Control Input Comparison: LQR vs Moment Matching')
            else:
                ax.set_ylabel(f'{self._control_label} (dim {dim+1})')
                ax.set_title(f'Control Input {dim+1}: LQR vs Moment Matching')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add action bounds as horizontal lines
            if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                ax.axhline(y=self.u_bounds[0], color='gray', linestyle='--', alpha=0.5, 
                          label=f'Lower Bound ({self.u_bounds[0]})')
                ax.axhline(y=self.u_bounds[1], color='gray', linestyle='--', alpha=0.5, 
                          label=f'Upper Bound ({self.u_bounds[1]})')
            
            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide unused subplots
        if du > 1:
            for dim in range(du, len(axes)):
                axes[dim].set_visible(False)
        
        if du > 1:
            fig.suptitle('Control Input Comparison: LQR vs Moment Matching', fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Generate save path if not provided
        if save_path is None:
            save_path = self._generate_plot_filename(N_samples, "control_comparison")
        
        # Save the plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Control input comparison plot saved to {save_path}")
        
        plt.close(fig)  # Close the figure to free memory
        
        return fig
    
    def plot_trajectory_comparison(self, lqr_traj, mm_traj, N_samples, save_path=None):
        """
        Plot trajectory comparison with 4 separate subplots for each state dimension
        
        Args:
            lqr_traj: LQR trajectory (T, 4) array
            mm_traj: Moment matching trajectory (T, 4) array
            N_samples: Number of samples used in the experiment
            save_path: Optional custom path to save the plot (overrides default structure)
        """
        if lqr_traj.ndim != 2 or mm_traj.ndim != 2:
            raise ValueError("Trajectories must be 2D arrays of shape (T, n_states)")
        
        n_states = lqr_traj.shape[1]
        state_names, state_units, state_limits = self._get_state_metadata(n_states)
        
        n_cols = min(2, n_states)
        n_rows = int(np.ceil(n_states / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        
        time_lqr = np.arange(len(lqr_traj)) * self.dt
        time_mm = np.arange(len(mm_traj)) * self.dt
        
        for i in range(n_states):
            ax = axes[i]
            
            ax.plot(time_lqr, lqr_traj[:, i], 'b-', label='LQR', linewidth=2, alpha=0.8)
            ax.plot(time_mm, mm_traj[:, i], 'r-', label='Moment Matching', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'State Value ({state_units[i]})')
            ax.set_title(state_names[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if state_limits[i] is not None:
                ax.set_ylim(*state_limits[i])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for j in range(n_states, len(axes)):
            axes[j].set_visible(False)
        
        fig.suptitle('Trajectory Comparison: LQR vs Moment Matching', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Generate save path if not provided
        if save_path is None:
            save_path = self._generate_plot_filename(N_samples, "trajectory_comparison")
        
        # Save the plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to {save_path}")
        
        plt.close(fig)  # Close the figure to free memory
        
        return fig
    
    def plot_comprehensive_comparison(self, lqr_trajs, mm_trajs, lqr_inputs_list, mm_inputs_list, N_samples, 
                                      save_path=None, extra_trajs=None, extra_inputs_list=None, extra_label="extra"):
        """
        Plot comprehensive comparison with both trajectories and control inputs
        
        Args:
            lqr_trajs: List of LQR trajectories, each (T, n_states) array
            mm_trajs: List of moment matching trajectories, each (T, n_states) array
            lqr_inputs_list: List of LQR control inputs, each (T,) array
            mm_inputs_list: List of moment matching control inputs, each (T,) array
            N_samples: Number of samples used in the experiment
            save_path: Optional custom path to save the plot (overrides default structure)
            extra_trajs: Optional list of extra policy trajectories (e.g., identity policy)
            extra_inputs_list: Optional list of extra policy control inputs
            extra_label: Label for the extra policy (default: "extra")
        """
        # Determine the actual state dimension from the trajectory data
        if len(lqr_trajs) > 0 and len(lqr_trajs[0]) > 0:
            n_states = lqr_trajs[0].shape[1]
        elif len(mm_trajs) > 0 and len(mm_trajs[0]) > 0:
            n_states = mm_trajs[0].shape[1]
        elif extra_trajs is not None and len(extra_trajs) > 0 and len(extra_trajs[0]) > 0:
            n_states = extra_trajs[0].shape[1]
        else:
            raise ValueError("Cannot determine state dimension from empty trajectories")
        
        # State dimension names with proper units - dynamically generate based on n_states
        state_names, state_units, state_limits = self._get_state_metadata(n_states)
        
        # Determine control dimension from inputs first
        du = 1  # Default to scalar control
        if len(lqr_inputs_list) > 0 and len(lqr_inputs_list[0]) > 0:
            first_input = lqr_inputs_list[0]
            if isinstance(first_input, list):
                if len(first_input) > 0 and isinstance(first_input[0], np.ndarray):
                    du = len(first_input[0])
                else:
                    du = 1
            elif isinstance(first_input, np.ndarray):
                if first_input.ndim == 1:
                    # Check first element to see if it's an array (multi-dimensional)
                    if len(first_input) > 0:
                        elem = first_input[0]
                        if isinstance(elem, np.ndarray) and elem.ndim == 1:
                            du = len(elem)
                        else:
                            du = 1
                else:
                    du = first_input.shape[1] if first_input.shape[1] > 1 else 1
        
        # Convert inputs to arrays for easier handling
        def convert_inputs_to_array(input_list):
            """Convert list of inputs to array format"""
            if len(input_list) == 0:
                return None
            result = []
            for traj in input_list:
                # Handle empty arrays/lists
                if isinstance(traj, np.ndarray) and traj.size == 0:
                    result.append(np.array([]))
                    continue
                if isinstance(traj, list) and len(traj) == 0:
                    result.append(np.array([]))
                    continue
                
                if isinstance(traj, list):
                    # List of arrays or scalars
                    if len(traj) > 0 and isinstance(traj[0], np.ndarray):
                        # List of arrays - convert to 2D array
                        result.append(np.array([np.asarray(u).reshape(-1) for u in traj]))
                    else:
                        # List of scalars - convert to 1D array
                        result.append(np.asarray(traj))
                elif isinstance(traj, np.ndarray):
                    if traj.ndim == 1:
                        # Check if elements are arrays (multi-dimensional) or scalars
                        if len(traj) > 0:
                            elem = traj[0]
                            if isinstance(elem, np.ndarray) and elem.ndim == 1:
                                # Multi-dimensional: convert list of arrays to 2D array
                                result.append(np.array([np.asarray(u).reshape(-1) for u in traj]))
                            else:
                                # Scalar: already 1D array
                                result.append(traj)
                        else:
                            result.append(traj)
                    else:
                        # Already 2D array
                        result.append(traj)
                else:
                    # List of scalars
                    result.append(np.asarray(traj))
            return result
        
        lqr_inputs_arr_list = convert_inputs_to_array(lqr_inputs_list)
        mm_inputs_arr_list = convert_inputs_to_array(mm_inputs_list)
        extra_inputs_arr_list = convert_inputs_to_array(extra_inputs_list) if extra_inputs_list is not None else None
        
        # Create figure with appropriate number of subplots (n_states + du control dimensions)
        n_plots = n_states + du
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each state dimension
        for i in range(n_states):
            ax = axes[i]
            
            # Plot trajectories for each test case
            for j in range(len(lqr_trajs)):
                if len(lqr_trajs[j]) > 0 and len(mm_trajs[j]) > 0:
                    # Convert time steps to actual time using DT
                    time_lqr = np.arange(len(lqr_trajs[j])) * self.dt
                    time_mm = np.arange(len(mm_trajs[j])) * self.dt
                    
                    # Plot LQR trajectory
                    ax.plot(time_lqr, lqr_trajs[j][:, i], 'b-', linewidth=2, alpha=0.8)
                    
                    # Plot MM trajectory
                    ax.plot(time_mm, mm_trajs[j][:, i], 'r-', linewidth=2, alpha=0.8)
                    
                    # Plot extra policy trajectory if available
                    if extra_trajs is not None and j < len(extra_trajs) and len(extra_trajs[j]) > 0:
                        time_extra = np.arange(len(extra_trajs[j])) * self.dt
                        ax.plot(time_extra, extra_trajs[j][:, i], 'g-', linewidth=2, alpha=0.8)
            
            if i == 0:
                ax.plot([], [], 'b-', label='LQR', linewidth=2)
                ax.plot([], [], 'r-', label='Moment Matching', linewidth=2)
                if extra_trajs is not None:
                    ax.plot([], [], 'g-', label=extra_label.capitalize(), linewidth=2)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'State Value ({state_units[i]})')
            ax.set_title(state_names[i])
            ax.grid(True, alpha=0.3)
            
            if state_limits[i] is not None:
                ax.set_ylim(*state_limits[i])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Plot control inputs - one subplot per dimension
        for dim in range(du):
            ax = axes[n_states + dim]
            
            for j in range(len(lqr_inputs_arr_list)):
                if (j < len(mm_inputs_arr_list) and 
                    lqr_inputs_arr_list[j] is not None and mm_inputs_arr_list[j] is not None and
                    len(lqr_inputs_arr_list[j]) > 0 and len(mm_inputs_arr_list[j]) > 0):
                    # Convert time steps to actual time using DT
                    time_lqr = np.arange(len(lqr_inputs_arr_list[j])) * self.dt
                    time_mm = np.arange(len(mm_inputs_arr_list[j])) * self.dt
                    
                    # Extract this dimension's inputs
                    if du == 1:
                        lqr_dim = lqr_inputs_arr_list[j] if lqr_inputs_arr_list[j].ndim == 1 else lqr_inputs_arr_list[j][:, 0]
                        mm_dim = mm_inputs_arr_list[j] if mm_inputs_arr_list[j].ndim == 1 else mm_inputs_arr_list[j][:, 0]
                    else:
                        lqr_dim = lqr_inputs_arr_list[j][:, dim]
                        mm_dim = mm_inputs_arr_list[j][:, dim]
                    
                    # Plot LQR inputs
                    ax.plot(time_lqr, lqr_dim, 'b-', linewidth=2, alpha=0.8)
                    
                    # Plot MM inputs
                    ax.plot(time_mm, mm_dim, 'r-', linewidth=2, alpha=0.8)
                    
                    # Plot extra policy inputs if available
                    if extra_inputs_arr_list is not None and j < len(extra_inputs_arr_list) and extra_inputs_arr_list[j] is not None:
                        if len(extra_inputs_arr_list[j]) > 0:
                            time_extra = np.arange(len(extra_inputs_arr_list[j])) * self.dt
                            if du == 1:
                                extra_dim = extra_inputs_arr_list[j] if extra_inputs_arr_list[j].ndim == 1 else extra_inputs_arr_list[j][:, 0]
                            else:
                                extra_dim = extra_inputs_arr_list[j][:, dim]
                            ax.plot(time_extra, extra_dim, 'g-', linewidth=2, alpha=0.4)
            
            # Add labels for control inputs
            if dim == 0:
                ax.plot([], [], 'b-', label='LQR', linewidth=2)
                ax.plot([], [], 'r-', label='Moment Matching', linewidth=2)
                if extra_inputs_arr_list is not None:
                    ax.plot([], [], 'g-', label=extra_label.capitalize(), linewidth=2)
            
            # Add action bounds
            if isinstance(self.u_bounds, tuple) and len(self.u_bounds) == 2:
                ax.axhline(y=self.u_bounds[0], color='gray', linestyle='--', alpha=0.5, 
                          label=f'Lower Bound ({self.u_bounds[0]})' if dim == 0 else '')
                ax.axhline(y=self.u_bounds[1], color='gray', linestyle='--', alpha=0.5, 
                          label=f'Upper Bound ({self.u_bounds[1]})' if dim == 0 else '')
            
            ax.set_xlabel('Time (s)')
            if du == 1:
                ax.set_ylabel(self._control_label)
                ax.set_title('Control Input Comparison')
            else:
                ax.set_ylabel(f'{self._control_label} (dim {dim+1})')
                ax.set_title(f'Control Input {dim+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide any unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # Overall title
        if extra_trajs is not None:
            fig.suptitle(f'Comprehensive Policy Comparison: LQR vs Moment Matching vs {extra_label.capitalize()}', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Comprehensive Policy Comparison: LQR vs Moment Matching', fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Generate save path if not provided
        if save_path is None:
            save_path = self._generate_plot_filename(N_samples, "policy_comparison", "")
        
        # Save the plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive comparison plot saved to {save_path}")
        
        plt.close(fig)  # Close the figure to free memory
        
        return fig
    
    def compare_policies(self, system, lqr_policy, mm_policy, test_states, N_samples, horizon=1000, 
                        analyze_divergence=False, divergence_threshold=5.0, divergence_save_path=None,
                        extra_policy=None, extra_label="extra"):
        """
        Policy comparison with better error handling and statistics.
        Only evaluates costs for starting points where both methods converge.
        
        Args:
            system: dynamical system
            lqr_policy: LQR policy function
            mm_policy: moment matching policy function
            test_states: array of test states
            N_samples: Number of samples used in the experiment
            horizon: simulation horizon
            analyze_divergence: If True, analyze divergence cases and create plots
            divergence_threshold: State difference threshold for divergence detection
            divergence_save_path: Path to save divergence analysis plots
            
        Returns:
            dict: comparison results with statistics (and optionally divergence analysis)
        """
        n_tests = test_states.shape[0]
        lqr_costs = []
        mm_costs = []
        extra_costs = []
        lqr_success = []
        mm_success = []
        extra_success = []
        all_three_converge_flags = []  # Track when all three methods converge
        convergent_costs_lqr = []  # Costs only for convergent cases (LQR and MM both converge)
        convergent_costs_mm = []   # Costs only for convergent cases (LQR and MM both converge)
        convergent_costs_extra = []  # Costs for optional extra policy vs LQR (LQR and extra both converge)
        convergent_costs_all = []    # Costs where all provided policies converge
        # Costs for each method when all three converge
        all_convergent_lqr_costs = []
        all_convergent_mm_costs = []
        all_convergent_extra_costs = []
        # Costs for each method individually when that method converged (regardless of others)
        lqr_costs_when_lqr_converged = []
        mm_costs_when_mm_converged = []
        extra_costs_when_extra_converged = []
        lqr_trajs = []
        mm_trajs = []
        extra_trajs = []
        lqr_inputs_list = []
        mm_inputs_list = []
        extra_inputs_list = []
        
        for i in range(n_tests):
            initial_state = test_states[i]
            
            # Simulate LQR policy (no angle wrapping)
            try:
                # print(f"Simulating LQR policy with initial state: {initial_state}")
                lqr_traj, lqr_cost, lqr_succ, lqr_inputs_single = self._simulate_policy(
                    system, lqr_policy, initial_state, horizon, wrap_angles=False
                )
                lqr_costs.append(lqr_cost)
                lqr_success.append(lqr_succ)
                lqr_inputs_list.append(lqr_inputs_single)
                lqr_trajs.append(lqr_traj)
            except Exception as e:
                print(f"LQR simulation failed for state {i}: {e}")
                lqr_costs.append(np.inf)
                lqr_success.append(False)
                lqr_inputs_list.append(np.array([]))
                lqr_trajs.append(np.array([]))
            
            # Simulate moment matching policy (with angle wrapping)
            try:
                # print(f"Simulating MM policy with initial state: {initial_state}")
                mm_traj, mm_cost, mm_succ, mm_inputs_single = self._simulate_policy(
                    system, mm_policy, initial_state, horizon, wrap_angles=True
                )
                mm_costs.append(mm_cost)
                mm_success.append(mm_succ)
                mm_inputs_list.append(mm_inputs_single)
                mm_trajs.append(mm_traj)
            except Exception as e:
                print(f"MM simulation failed for state {i}: {e}")
                mm_costs.append(np.inf)
                mm_success.append(False)
                mm_inputs_list.append(np.array([]))
                mm_trajs.append(np.array([]))

            # Simulate optional extra policy (e.g., identity) - use wrapping like MM
            if extra_policy is not None:
                try:
                    extra_traj, extra_cost, extra_succ, extra_inputs_single = self._simulate_policy(
                        system, extra_policy, initial_state, horizon, wrap_angles=True
                    )
                    extra_costs.append(extra_cost)
                    extra_success.append(extra_succ)
                    extra_inputs_list.append(extra_inputs_single)
                    extra_trajs.append(extra_traj)
                except Exception as e:
                    print(f"{extra_label} simulation failed for state {i}: {e}")
                    extra_costs.append(np.inf)
                    extra_success.append(False)
                    extra_inputs_list.append(np.array([]))
                    extra_trajs.append(np.array([]))
            
            # Track costs for each method when that method individually converged
            if lqr_success[-1]:
                lqr_costs_when_lqr_converged.append(lqr_cost)
            if mm_success[-1]:
                mm_costs_when_mm_converged.append(mm_cost)
            if extra_policy is not None and extra_success[-1]:
                extra_costs_when_extra_converged.append(extra_costs[-1])
            
            # Only include costs if both methods converged for this starting point
            if lqr_success[-1] and mm_success[-1]:
                convergent_costs_lqr.append(lqr_cost)
                convergent_costs_mm.append(mm_cost)
                if extra_policy is None:
                    convergent_costs_all.append((lqr_cost, mm_cost))
            # For optional extra policy, track convergence intersections
            if extra_policy is not None and lqr_success[-1] and extra_success[-1]:
                convergent_costs_extra.append(extra_costs[-1])
                if mm_success[-1]:
                    convergent_costs_all.append((lqr_cost, mm_cost, extra_costs[-1]))
            
            # Track cases where all three methods converge (for statistics)
            if extra_policy is not None:
                all_three_converge = lqr_success[-1] and mm_success[-1] and extra_success[-1]
                all_three_converge_flags.append(all_three_converge)
                if all_three_converge:
                    all_convergent_lqr_costs.append(lqr_cost)
                    all_convergent_mm_costs.append(mm_cost)
                    all_convergent_extra_costs.append(extra_costs[-1])
            else:
                all_three_converge_flags.append(None)
        
        # Compute robust statistics
        def safe_mean(data, default=np.inf):
            return np.mean(data) if len(data) > 0 else default
        def safe_std(data, default=0.0):
            return np.std(data) if len(data) > 0 else default
        def safe_median(data, default=0.0):
            return np.median(data) if len(data) > 0 else default
        
        # Calculate statistics for convergent cases only
        n_convergent = len(convergent_costs_lqr)
        convergent_lqr_mean = safe_mean(convergent_costs_lqr)
        convergent_mm_mean = safe_mean(convergent_costs_mm)
        convergent_extra_mean = safe_mean(convergent_costs_extra) if extra_policy is not None else None
        convergent_all_mean = safe_mean([c[-1] for c in convergent_costs_all]) if len(convergent_costs_all) > 0 else None
        
        # Calculate statistics for each method individually (when that method converged)
        lqr_costs_all_converged_mean = safe_mean(lqr_costs_when_lqr_converged)
        mm_costs_all_converged_mean = safe_mean(mm_costs_when_mm_converged)
        extra_costs_all_converged_mean = safe_mean(extra_costs_when_extra_converged) if extra_policy is not None else None
        
        # Calculate total costs (sum) for each method when that method converged
        def safe_sum(data, default=0.0):
            return np.sum(data) if len(data) > 0 else default
        
        lqr_costs_all_converged_total = safe_sum(lqr_costs_when_lqr_converged)
        mm_costs_all_converged_total = safe_sum(mm_costs_when_mm_converged)
        extra_costs_all_converged_total = safe_sum(extra_costs_when_extra_converged) if extra_policy is not None else None
        
        # Calculate statistics when all three methods converge (if extra_policy is provided)
        if extra_policy is not None:
            n_all_convergent = len(all_convergent_lqr_costs)
            all_convergent_lqr_mean = safe_mean(all_convergent_lqr_costs)
            all_convergent_mm_mean = safe_mean(all_convergent_mm_costs)
            all_convergent_extra_mean = safe_mean(all_convergent_extra_costs)
            all_convergent_success_rate = np.mean(all_three_converge_flags) if len(all_three_converge_flags) > 0 else 0.0
            # Calculate total costs when all three methods converge
            all_convergent_lqr_total = safe_sum(all_convergent_lqr_costs)
            all_convergent_mm_total = safe_sum(all_convergent_mm_costs)
            all_convergent_extra_total = safe_sum(all_convergent_extra_costs)
        else:
            n_all_convergent = None
            all_convergent_lqr_mean = None
            all_convergent_mm_mean = None
            all_convergent_extra_mean = None
            all_convergent_success_rate = None
            all_convergent_lqr_total = None
            all_convergent_mm_total = None
            all_convergent_extra_total = None
        
        print(f"Policy comparison: {n_convergent}/{n_tests} starting points had both LQR and MM convergent")
        
        # Print which test points failed to converge for each policy
        lqr_failed_indices = [i for i in range(n_tests) if not lqr_success[i]]
        mm_failed_indices = [i for i in range(n_tests) if not mm_success[i]]
        
        if lqr_failed_indices:
            print(f"\n--- LQR failed to converge for {len(lqr_failed_indices)} test point(s) ---")
            for idx in lqr_failed_indices:
                print(f"  Test point {idx}: {test_states[idx]}")
        else:
            print(f"\n--- LQR converged for all {n_tests} test points ---")
        
        if mm_failed_indices:
            print(f"\n--- MM failed to converge for {len(mm_failed_indices)} test point(s) ---")
            for idx in mm_failed_indices:
                print(f"  Test point {idx}: {test_states[idx]}")
        else:
            print(f"\n--- MM converged for all {n_tests} test points ---")
        
        if extra_policy is not None:
            extra_failed_indices = [i for i in range(n_tests) if not extra_success[i]]
            if extra_failed_indices:
                print(f"\n--- {extra_label.upper()} failed to converge for {len(extra_failed_indices)} test point(s) ---")
                for idx in extra_failed_indices:
                    print(f"  Test point {idx}: {test_states[idx]}")
            else:
                print(f"\n--- {extra_label.upper()} converged for all {n_tests} test points ---")
        
        print(f"\n--- Costs when both LQR and MM converge ---")
        print(f"LQR mean cost: {convergent_lqr_mean:.4f}")
        print(f"MM mean cost: {convergent_mm_mean:.4f}")
        
        print(f"\n--- Costs for each method individually (when that method converged) ---")
        print(f"LQR mean cost (all LQR converged): {lqr_costs_all_converged_mean:.4f} (n={len(lqr_costs_when_lqr_converged)})")
        print(f"MM mean cost (all MM converged): {mm_costs_all_converged_mean:.4f} (n={len(mm_costs_when_mm_converged)})")
        if extra_policy is not None:
            if extra_costs_all_converged_mean is not None and np.isfinite(extra_costs_all_converged_mean):
                print(f"{extra_label.upper()} mean cost (all {extra_label} converged): {extra_costs_all_converged_mean:.4f} (n={len(extra_costs_when_extra_converged)})")
            else:
                print(f"{extra_label.upper()} mean cost (all {extra_label} converged): N/A")
        
        if extra_policy is not None:
            if convergent_extra_mean is not None and np.isfinite(convergent_extra_mean):
                print(f"\n--- Costs when LQR and {extra_label.upper()} both converge ---")
                print(f"{extra_label.upper()} mean cost: {convergent_extra_mean:.4f}")
            if convergent_all_mean is not None and np.isfinite(convergent_all_mean):
                print(f"\n--- Costs when all policies converge ---")
                print(f"Mean {extra_label.upper()} cost: {convergent_all_mean:.4f}")
            if n_all_convergent is not None:
                print(f"\n--- All three methods convergent: {n_all_convergent}/{n_tests} starting points ({all_convergent_success_rate:.2%}) ---")
                print(f"Mean costs when all three converge:")
                print(f"  LQR: {all_convergent_lqr_mean:.4f}")
                print(f"  MM: {all_convergent_mm_mean:.4f}")
                print(f"  {extra_label.upper()}: {all_convergent_extra_mean:.4f}")

        # Convert lists to arrays for plotting
        if len(lqr_trajs) > 0 and len(mm_trajs) > 0:
            # Filter out empty trajectories
            valid_lqr_trajs = [traj for traj in lqr_trajs if len(traj) > 0]
            valid_mm_trajs = [traj for traj in mm_trajs if len(traj) > 0]
            valid_lqr_inputs = [inp for inp in lqr_inputs_list if len(inp) > 0]
            valid_mm_inputs = [inp for inp in mm_inputs_list if len(inp) > 0]
            
            # Filter extra policy trajectories and inputs if available
            valid_extra_trajs = None
            valid_extra_inputs = None
            if extra_policy is not None and len(extra_trajs) > 0:
                valid_extra_trajs = [traj for traj in extra_trajs if len(traj) > 0]
                valid_extra_inputs = [inp for inp in extra_inputs_list if len(inp) > 0]
                if len(valid_extra_trajs) == 0:
                    valid_extra_trajs = None
                    valid_extra_inputs = None
            
            if len(valid_lqr_trajs) > 0 and len(valid_mm_trajs) > 0:
                self.plot_comprehensive_comparison(
                    valid_lqr_trajs, valid_mm_trajs, valid_lqr_inputs, valid_mm_inputs, N_samples,
                    extra_trajs=valid_extra_trajs, extra_inputs_list=valid_extra_inputs, extra_label=extra_label
                )
        
        result = {
            # Number of convergent cases
            "n_convergent": n_convergent,
            "n_total": n_tests,
            "convergent_ratio": n_convergent / n_tests if n_tests > 0 else 0.0,
            
            # Success rates: computed separately for each method (all test cases)
            "lqr_success": np.mean(lqr_success),
            "mm_success": np.mean(mm_success),
            
            # Cost averages: computed only when both LQR and MM converge
            "lqr_costs": convergent_lqr_mean,
            "mm_costs": convergent_mm_mean,
            
            # Cost averages: computed for each method individually when that method converged
            "lqr_costs_all_converged": lqr_costs_all_converged_mean,
            "mm_costs_all_converged": mm_costs_all_converged_mean,
            "lqr_costs_all_converged_std": safe_std(lqr_costs_when_lqr_converged),
            "mm_costs_all_converged_std": safe_std(mm_costs_when_mm_converged),
            "lqr_costs_all_converged_median": safe_median(lqr_costs_when_lqr_converged),
            "mm_costs_all_converged_median": safe_median(mm_costs_when_mm_converged),
            # Total costs: computed for each method individually when that method converged
            "lqr_costs_all_converged_total": lqr_costs_all_converged_total,
            "mm_costs_all_converged_total": mm_costs_all_converged_total,
        }
        # Add extra policy statistics if provided
        if extra_policy is not None:
            # Success rate: computed separately (all test cases)
            result[f"{extra_label}_success"] = np.mean(extra_success)
            # Cost average: computed only when LQR and extra converge
            result[f"{extra_label}_costs"] = convergent_extra_mean
            result[f"{extra_label}_costs_std"] = safe_std(convergent_costs_extra)
            result[f"{extra_label}_costs_median"] = safe_median(convergent_costs_extra)
            # Cost average: computed when extra method individually converged
            result[f"{extra_label}_costs_all_converged"] = extra_costs_all_converged_mean
            result[f"{extra_label}_costs_all_converged_std"] = safe_std(extra_costs_when_extra_converged)
            result[f"{extra_label}_costs_all_converged_median"] = safe_median(extra_costs_when_extra_converged)
            # Total cost: computed when extra method individually converged
            result[f"{extra_label}_costs_all_converged_total"] = extra_costs_all_converged_total
            if convergent_all_mean is not None:
                result[f"{extra_label}_costs_all_convergent"] = convergent_all_mean
            
            # Statistics when ALL THREE methods converge
            if n_all_convergent is not None:
                result["n_all_convergent"] = n_all_convergent
                result["all_convergent_success_rate"] = all_convergent_success_rate
                result["all_convergent_lqr_costs"] = all_convergent_lqr_mean
                result["all_convergent_mm_costs"] = all_convergent_mm_mean
                result[f"all_convergent_{extra_label}_costs"] = all_convergent_extra_mean
                result["all_convergent_lqr_costs_std"] = safe_std(all_convergent_lqr_costs)
                result["all_convergent_mm_costs_std"] = safe_std(all_convergent_mm_costs)
                result[f"all_convergent_{extra_label}_costs_std"] = safe_std(all_convergent_extra_costs)
                # Total costs when all three methods converge
                result["all_convergent_lqr_costs_total"] = all_convergent_lqr_total
                result["all_convergent_mm_costs_total"] = all_convergent_mm_total
                result[f"all_convergent_{extra_label}_costs_total"] = all_convergent_extra_total
        
        # Analyze divergence if requested (reusing trajectories we already computed)
        if analyze_divergence:
            divergence_results = self._analyze_divergence_from_trajectories(
                system, lqr_policy, mm_policy, test_states,
                lqr_trajs, mm_trajs, lqr_inputs_list, mm_inputs_list,
                lqr_costs, mm_costs, lqr_success, mm_success,
                divergence_threshold=divergence_threshold,
                save_path=divergence_save_path
            )
            result["divergence_analysis"] = divergence_results
        
        return result
    
    def _analyze_divergence_from_trajectories(self, system, lqr_policy, mm_policy, test_states,
                                              lqr_trajs, mm_trajs, lqr_inputs_list, mm_inputs_list,
                                              lqr_costs, mm_costs, lqr_success, mm_success,
                                              divergence_threshold=5.0, save_path=None):
        """
        Internal method to analyze divergence using pre-computed trajectories.
        Called from compare_policies to avoid re-simulation.
        """
        n_tests = test_states.shape[0]
        divergence_cases = []
        
        print(f"\n=== Policy Divergence Analysis ===")
        print(f"Analyzing {n_tests} test states using pre-computed trajectories...")
        
        for i in range(n_tests):
            initial_state = test_states[i]
            
            # Use pre-computed trajectories
            lqr_traj = lqr_trajs[i]
            mm_traj = mm_trajs[i]
            lqr_inputs = lqr_inputs_list[i]
            mm_inputs = mm_inputs_list[i]
            lqr_cost = lqr_costs[i]
            mm_cost = mm_costs[i]
            lqr_succ = lqr_success[i]
            mm_succ = mm_success[i]
            
            # Only analyze cases where LQR succeeds but MM fails
            if lqr_succ and not mm_succ:
                print(f"\n--- Divergence Case {len(divergence_cases) + 1}: Test State {i} ---")
                print(f"Initial state: {initial_state}")
                
                # Find where trajectories diverge
                min_len = min(len(lqr_traj), len(mm_traj))
                state_diffs = []
                action_diffs = []
                state_norms = []
                
                divergence_step = None
                divergence_reason = None
                
                for step in range(min_len):
                    # Compute state difference
                    state_diff = np.linalg.norm(lqr_traj[step] - mm_traj[step])
                    state_diffs.append(state_diff)
                    state_norms.append((np.linalg.norm(lqr_traj[step]), 
                                       np.linalg.norm(mm_traj[step])))
                    
                    # Compute action difference
                    if step < len(lqr_inputs) and step < len(mm_inputs):
                        lqr_u = np.asarray(lqr_inputs[step]).reshape(-1)
                        mm_u = np.asarray(mm_inputs[step]).reshape(-1)
                        # Compute L2 norm of difference for multi-dimensional control
                        action_diff = np.linalg.norm(lqr_u - mm_u)
                        action_diffs.append(action_diff)
                    else:
                        action_diffs.append(np.nan)
                    
                    # Check for divergence
                    if state_diff > divergence_threshold:
                        divergence_step = step
                        divergence_reason = f"State difference exceeded threshold ({state_diff:.4f} > {divergence_threshold})"
                        break
                    
                    # Check if MM trajectory has diverged (large state values)
                    if np.any(np.abs(mm_traj[step]) > 20):
                        divergence_step = step
                        divergence_reason = f"MM state exceeded bounds: {mm_traj[step]}"
                        break
                
                # If no clear divergence point found, use the last step
                if divergence_step is None:
                    divergence_step = min_len - 1
                    divergence_reason = "Trajectories diverged gradually or MM failed early"
                
                # Analyze the divergence point
                div_state_lqr = lqr_traj[divergence_step] if divergence_step < len(lqr_traj) else None
                div_state_mm = mm_traj[divergence_step] if divergence_step < len(mm_traj) else None
                div_action_lqr = lqr_inputs[divergence_step] if divergence_step < len(lqr_inputs) else None
                div_action_mm = mm_inputs[divergence_step] if divergence_step < len(mm_inputs) else None
                
                # Compute policy actions at divergence state
                if div_state_lqr is not None:
                    try:
                        lqr_action_at_div = lqr_policy(div_state_lqr)
                    except:
                        lqr_action_at_div = None
                else:
                    lqr_action_at_div = None
                
                if div_state_mm is not None:
                    try:
                        mm_action_at_div = mm_policy(div_state_mm)
                    except:
                        mm_action_at_div = None
                else:
                    mm_action_at_div = None
                
                # Compute policy actions at LQR state (to see what MM would do)
                if div_state_lqr is not None:
                    try:
                        mm_action_at_lqr_state = mm_policy(div_state_lqr)
                    except Exception as e:
                        mm_action_at_lqr_state = None
                        print(f"  Warning: MM policy failed at LQR divergence state: {e}")
                else:
                    mm_action_at_lqr_state = None
                
                # Compute policy actions at MM state (to see what LQR would do)
                if div_state_mm is not None:
                    try:
                        lqr_action_at_mm_state = lqr_policy(div_state_mm)
                    except:
                        lqr_action_at_mm_state = None
                else:
                    lqr_action_at_mm_state = None
                
                divergence_info = {
                    "test_state_idx": i,
                    "initial_state": initial_state.copy(),
                    "divergence_step": divergence_step,
                    "divergence_reason": divergence_reason,
                    "lqr_trajectory": lqr_traj[:divergence_step+1],
                    "mm_trajectory": mm_traj[:divergence_step+1],
                    "lqr_inputs": lqr_inputs[:divergence_step+1] if len(lqr_inputs) > divergence_step else lqr_inputs,
                    "mm_inputs": mm_inputs[:divergence_step+1] if len(mm_inputs) > divergence_step else mm_inputs,
                    "state_differences": np.array(state_diffs),
                    "action_differences": np.array(action_diffs),
                    "state_norms": np.array(state_norms),
                    "divergence_state_lqr": div_state_lqr,
                    "divergence_state_mm": div_state_mm,
                    "divergence_action_lqr": div_action_lqr,
                    "divergence_action_mm": div_action_mm,
                    "lqr_action_at_div_state": lqr_action_at_div,
                    "mm_action_at_div_state": mm_action_at_div,
                    "mm_action_at_lqr_state": mm_action_at_lqr_state,
                    "lqr_action_at_mm_state": lqr_action_at_mm_state,
                    "lqr_success": lqr_succ,
                    "mm_success": mm_succ,
                    "lqr_cost": lqr_cost,
                    "mm_cost": mm_cost,
                }
                
                divergence_cases.append(divergence_info)
                
                # Print summary
                print(f"  Divergence at step {divergence_step}: {divergence_reason}")
                if div_state_lqr is not None:
                    print(f"  LQR state at divergence: {div_state_lqr}")
                if div_state_mm is not None:
                    print(f"  MM state at divergence: {div_state_mm}")
                if div_action_lqr is not None and div_action_mm is not None:
                    print(f"  Actions at divergence - LQR: {div_action_lqr:.4f}, MM: {div_action_mm:.4f}, Diff: {abs(div_action_lqr - div_action_mm):.4f}")
                if mm_action_at_lqr_state is not None:
                    print(f"  MM action at LQR divergence state: {mm_action_at_lqr_state:.4f}")
                if lqr_action_at_mm_state is not None:
                    print(f"  LQR action at MM divergence state: {lqr_action_at_mm_state:.4f}")
        
        print(f"\nFound {len(divergence_cases)} divergence cases (LQR succeeds, MM fails)")
        
        # Create visualization if we have divergence cases
        if len(divergence_cases) > 0 and save_path is not None:
            self._plot_divergence_analysis(divergence_cases, save_path)
        
        return {
            "n_divergence_cases": len(divergence_cases),
            "divergence_cases": divergence_cases,
            "summary": self._summarize_divergence(divergence_cases)
        }
    
    def analyze_policy_divergence(self, system, lqr_policy, mm_policy, test_states, 
                                  horizon=10000, divergence_threshold=5.0, 
                                  max_early_steps=1000, save_path=None):
        """
        Standalone divergence analysis (simulates trajectories if not provided).
        For use when compare_policies is not called first.
        
        Args:
            system: dynamical system
            lqr_policy: LQR policy function
            mm_policy: Moment matching policy function
            test_states: array of test states to analyze
            horizon: simulation horizon
            divergence_threshold: State difference threshold to consider as divergence
            max_early_steps: Maximum steps to track before giving up
            save_path: Optional path to save divergence analysis plots
            
        Returns:
            dict: Detailed divergence analysis results
        """
        n_tests = test_states.shape[0]
        lqr_trajs = []
        mm_trajs = []
        lqr_inputs_list = []
        mm_inputs_list = []
        lqr_costs = []
        mm_costs = []
        lqr_success = []
        mm_success = []
        
        print(f"\n=== Policy Divergence Analysis ===")
        print(f"Simulating {n_tests} test states...")
        
        for i in range(n_tests):
            initial_state = test_states[i]
            
            # Simulate both policies step by step
            lqr_traj, lqr_cost, lqr_succ, lqr_inputs = self._simulate_policy(
                system, lqr_policy, initial_state, horizon, wrap_angles=False
            )
            mm_traj, mm_cost, mm_succ, mm_inputs = self._simulate_policy(
                system, mm_policy, initial_state, horizon, wrap_angles=True
            )
            
            lqr_trajs.append(lqr_traj)
            mm_trajs.append(mm_traj)
            lqr_inputs_list.append(lqr_inputs)
            mm_inputs_list.append(mm_inputs)
            lqr_costs.append(lqr_cost)
            mm_costs.append(mm_cost)
            lqr_success.append(lqr_succ)
            mm_success.append(mm_succ)
        
        # Use internal method to analyze divergence
        return self._analyze_divergence_from_trajectories(
            system, lqr_policy, mm_policy, test_states,
            lqr_trajs, mm_trajs, lqr_inputs_list, mm_inputs_list,
            lqr_costs, mm_costs, lqr_success, mm_success,
            divergence_threshold=divergence_threshold,
            save_path=save_path
        )
    
    def _plot_divergence_analysis(self, divergence_cases, save_path):
        """Plot detailed divergence analysis for all cases."""
        n_cases = len(divergence_cases)
        if n_cases == 0:
            return
        
        # Determine control dimension from first case
        du = 1
        if len(divergence_cases) > 0:
            first_case = divergence_cases[0]
            lqr_inputs = first_case["lqr_inputs"]
            if len(lqr_inputs) > 0:
                first_input = lqr_inputs[0]
                if isinstance(first_input, np.ndarray) and first_input.ndim == 1:
                    du = len(first_input)
                elif isinstance(first_input, list):
                    if len(first_input) > 0 and isinstance(first_input[0], np.ndarray):
                        du = len(first_input[0])
        
        # Convert inputs to arrays for easier handling
        def convert_inputs(input_list):
            """Convert inputs to 2D array if multi-dimensional"""
            if len(input_list) == 0:
                return None
            result = []
            for u in input_list:
                u_arr = np.asarray(u).reshape(-1)
                result.append(u_arr)
            return np.array(result) if len(result) > 0 else None
        
        # Number of columns: 2 (state trajectories, state diff) + du (control dimensions)
        n_cols = 2 + du
        fig = plt.figure(figsize=(5*n_cols, 5 * n_cases))
        gs = fig.add_gridspec(n_cases, n_cols, hspace=0.3, wspace=0.3)
        
        for case_idx, case in enumerate(divergence_cases):
            lqr_traj = case["lqr_trajectory"]
            mm_traj = case["mm_trajectory"]
            steps = np.arange(len(lqr_traj))
            
            # Determine state dimension
            n_states = lqr_traj.shape[1] if lqr_traj.ndim == 2 else 4
            state_names, state_units, _ = self._get_state_metadata(n_states)
            
            # Plot 1: State trajectories comparison
            ax1 = fig.add_subplot(gs[case_idx, 0])
            for i in range(n_states):
                ax1.plot(steps, lqr_traj[:, i], 'b-', alpha=0.7, linewidth=2, 
                        label='LQR' if i == 0 else '')
                ax1.plot(steps[:len(mm_traj)], mm_traj[:len(mm_traj), i], 'r--', 
                        alpha=0.7, linewidth=2, label='MM' if i == 0 else '')
            
            ax1.axvline(x=case["divergence_step"], color='black', linestyle=':', 
                       linewidth=2, label='Divergence')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('State Value')
            ax1.set_title(f'Case {case_idx+1}: State Trajectories\n'
                         f'Initial: {case["initial_state"]}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            # Fix y-axis limits for state plots
            ax1.set_ylim(-20, 20)
            
            # Plot 2: State difference over time
            ax2 = fig.add_subplot(gs[case_idx, 1])
            state_diffs = case["state_differences"]
            ax2.plot(state_diffs, 'g-', linewidth=2)
            ax2.axvline(x=case["divergence_step"], color='black', linestyle=':', 
                       linewidth=2)
            ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, 
                       label='Threshold')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('State Difference (L2 norm)')
            ax2.set_title(f'State Difference Over Time\n'
                         f'Divergence at step {case["divergence_step"]}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Convert inputs to arrays
            lqr_inputs_arr = convert_inputs(case["lqr_inputs"])
            mm_inputs_arr = convert_inputs(case["mm_inputs"])
            
            # Plot control actions - one subplot per dimension
            for dim in range(du):
                ax3 = fig.add_subplot(gs[case_idx, 2 + dim])
                
                if lqr_inputs_arr is not None and mm_inputs_arr is not None:
                    min_len = min(len(lqr_inputs_arr), len(mm_inputs_arr))
                    steps_inputs = np.arange(min_len)
                    
                    # Extract this dimension's inputs
                    if du == 1:
                        lqr_dim = lqr_inputs_arr[:min_len] if lqr_inputs_arr.ndim == 1 else lqr_inputs_arr[:min_len, 0]
                        mm_dim = mm_inputs_arr[:min_len] if mm_inputs_arr.ndim == 1 else mm_inputs_arr[:min_len, 0]
                    else:
                        lqr_dim = lqr_inputs_arr[:min_len, dim]
                        mm_dim = mm_inputs_arr[:min_len, dim]
                    
                    ax3.plot(steps_inputs, lqr_dim, 'b-', alpha=0.7, 
                            linewidth=2, label='LQR')
                    ax3.plot(steps_inputs, mm_dim, 'r--', alpha=0.7, 
                            linewidth=2, label='MM')
                    ax3.axvline(x=case["divergence_step"], color='black', linestyle=':', 
                               linewidth=2)
                    
                    # Compute action difference at divergence
                    div_action_lqr = case["divergence_action_lqr"]
                    div_action_mm = case["divergence_action_mm"]
                    if div_action_lqr is not None and div_action_mm is not None:
                        lqr_u_div = np.asarray(div_action_lqr).reshape(-1)
                        mm_u_div = np.asarray(div_action_mm).reshape(-1)
                        if du == 1:
                            action_diff = abs(lqr_u_div[0] - mm_u_div[0])
                        else:
                            action_diff = np.linalg.norm(lqr_u_div - mm_u_div)
                        title_suffix = f'\nAction diff: {action_diff:.4f}'
                    else:
                        title_suffix = ''
                    
                    ax3.set_xlabel('Time Step')
                    if du == 1:
                        ax3.set_ylabel('Control Action')
                        ax3.set_title(f'Control Actions{title_suffix}')
                    else:
                        ax3.set_ylabel(f'Control Action (dim {dim+1})')
                        ax3.set_title(f'Control Actions {dim+1}{title_suffix}')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Policy Divergence Analysis: {n_cases} Cases Where LQR Succeeds but MM Fails',
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved divergence analysis plot to {save_path}")
        plt.close(fig)
    
    def _summarize_divergence(self, divergence_cases):
        """Summarize common patterns in divergence cases."""
        if len(divergence_cases) == 0:
            return {}
        
        divergence_steps = [case["divergence_step"] for case in divergence_cases]
        action_diffs_at_div = []
        state_diffs_at_div = []
        
        for case in divergence_cases:
            if case["divergence_action_lqr"] is not None and case["divergence_action_mm"] is not None:
                action_diffs_at_div.append(abs(case["divergence_action_lqr"] - case["divergence_action_mm"]))
            if len(case["state_differences"]) > 0:
                state_diffs_at_div.append(case["state_differences"][-1])
        
        summary = {
            "mean_divergence_step": np.mean(divergence_steps),
            "median_divergence_step": np.median(divergence_steps),
            "min_divergence_step": np.min(divergence_steps),
            "max_divergence_step": np.max(divergence_steps),
            "mean_action_diff_at_divergence": np.mean(action_diffs_at_div) if len(action_diffs_at_div) > 0 else None,
            "mean_state_diff_at_divergence": np.mean(state_diffs_at_div) if len(state_diffs_at_div) > 0 else None,
        }
        
        return summary
    
    def check_value_function_positivity(self, Q_matrix, poly, scaler=None, 
                                      n_samples=1000, state_bounds=None, 
                                      u_bounds=None, verbose=True):
        """
        Check if the value function φ(x,u)ᵀQφ(x,u) is positive for sampled state-input pairs.
        
        Args:
            Q_matrix: Learned Q matrix from moment matching (d, d)
            poly: Polynomial feature transformer
            scaler: Optional feature scaler
            n_samples: Number of state-input pairs to sample
            state_bounds: Bounds for state sampling [(min, max), ...] for each state dimension
            u_bounds: Bounds for control input sampling (min, max)
            verbose: Whether to print detailed results
            
        Returns:
            dict: Results containing positivity statistics and sample values
        """
        if Q_matrix is None:
            return {
                "all_positive": False,
                "positive_ratio": 0.0,
                "min_value": np.inf,
                "max_value": -np.inf,
                "mean_value": np.nan,
                "n_samples": 0,
                "error": "Q_matrix is None"
            }
        
        # Use default bounds if not provided
        if state_bounds is None:
            # Default bounds for cart-pole system: [x, x_dot, theta, theta_dot]
            state_bounds = [(-5.0, 5.0), (-10.0, 10.0), (-np.pi, np.pi), (-10.0, 10.0)]
        
        if u_bounds is None:
            u_bounds = self.u_bounds
        
        # Sample state-input pairs
        np.random.seed(42)  # For reproducibility
        states = np.random.uniform(
            low=[bounds[0] for bounds in state_bounds],
            high=[bounds[1] for bounds in state_bounds],
            size=(n_samples, len(state_bounds))
        )
        
        actions = np.random.uniform(
            low=u_bounds[0],
            high=u_bounds[1],
            size=(n_samples, 1)
        )
        
        # Combine states and actions
        state_action_pairs = np.concatenate([states, actions], axis=1)
        
        # Apply scaling if available
        if scaler is not None:
            # Check if this is a robust polynomial scaler (has both poly and scaler attributes)
            if hasattr(scaler, 'poly') and hasattr(scaler, 'scaler'):
                # This is a robust polynomial scaler - it handles both scaling and polynomial transformation
                phi_features = scaler.transform(state_action_pairs)
            else:
                # This is a regular scaler - apply scaling then polynomial transformation
                # But first check if the scaler expects polynomial features (RobustScaler)
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ > state_action_pairs.shape[1]:
                    # This scaler expects polynomial features, so transform first
                    state_action_pairs_poly = poly.transform(state_action_pairs)
                    phi_features = scaler.transform(state_action_pairs_poly)
                else:
                    # This scaler expects original features
                    state_action_pairs_scaled = scaler.transform(state_action_pairs)
                    phi_features = poly.transform(state_action_pairs_scaled)
        else:
            # No scaling - just polynomial transformation
            phi_features = poly.transform(state_action_pairs)
        
        # Compute value function values: φ(x,u)ᵀQφ(x,u)
        value_function_values = np.array([
            phi_features[i] @ Q_matrix @ phi_features[i].T 
            for i in range(n_samples)
        ]).flatten()
        
        # Check positivity
        positive_mask = value_function_values > 0
        all_positive = np.all(positive_mask)
        positive_ratio = np.mean(positive_mask)
        
        # Compute statistics
        min_value = np.min(value_function_values)
        max_value = np.max(value_function_values)
        mean_value = np.mean(value_function_values)
        
        if verbose:
            print(f"\n=== Value Function Positivity Check ===")
            print(f"Number of samples: {n_samples}")
            print(f"All values positive: {all_positive}")
            print(f"Positive ratio: {positive_ratio:.4f}")
            print(f"Min value: {min_value:.6f}")
            print(f"Max value: {max_value:.6f}")
            print(f"Mean value: {mean_value:.6f}")
            
            # Check Q matrix properties
            eigenvals = np.linalg.eigvals(Q_matrix)
            print(f"Q matrix eigenvalues - min: {np.min(eigenvals):.6f}, max: {np.max(eigenvals):.6f}")
            print(f"Q matrix condition number: {np.linalg.cond(Q_matrix):.2e}")
            print(f"Q matrix is positive definite: {np.all(eigenvals > 0)}")
            
            if not all_positive:
                negative_count = np.sum(~positive_mask)
                negative_values = value_function_values[~positive_mask]
                print(f"Negative values count: {negative_count}")
                print(f"Most negative value: {np.min(negative_values):.6f}")
                print(f"Negative values range: [{np.min(negative_values):.6f}, {np.max(negative_values):.6f}]")
        
        return {
            "all_positive": all_positive,
            "positive_ratio": positive_ratio,
            "min_value": min_value,
            "max_value": max_value,
            "mean_value": mean_value,
            "n_samples": n_samples,
            "value_function_values": value_function_values,
            "positive_mask": positive_mask,
            "error": None
        }
    
    def compare_policy_methods(self, system, Q_matrix, poly, scaler=None, 
                              test_states=None, n_grid_points=100, verbose=True):
        """
        Compare grid search policy with analytical policy extraction for verification
        
        Args:
            system: dynamical system
            Q_matrix: Q-matrix from moment matching
            poly: polynomial feature transformer
            scaler: optional feature scaler
            test_states: test states for comparison (if None, generates random states)
            n_grid_points: number of grid points for grid search
            verbose: whether to print detailed comparison results
            
        Returns:
            dict: comparison results including action differences and statistics
        """
        if test_states is None:
            # Generate random test states
            np.random.seed(42)
            n_test = 20
            test_states = np.random.uniform(
                low=[-2.0, -5.0, -0.5, -5.0],  # [x, x_dot, theta, theta_dot]
                high=[2.0, 5.0, 0.5, 5.0],
                size=(n_test, 4)
            )
        
        # Extract policies using both methods
        grid_policy = self.extract_policy_grid_search(system, Q_matrix, poly, scaler, n_grid_points)
        analytical_policy = self.extract_moment_matching_policy_analytical(system, Q_matrix, poly, scaler)
        
        # Compare actions for test states
        grid_actions = grid_policy(test_states)
        analytical_actions = analytical_policy(test_states)
        
        # Compute differences
        action_differences = np.abs(grid_actions - analytical_actions)
        max_difference = np.max(action_differences)
        mean_difference = np.mean(action_differences)
        std_difference = np.std(action_differences)
        
        # Compute Q-function values for both policies
        grid_q_values = []
        analytical_q_values = []
        
        for i in range(len(test_states)):
            state = test_states[i]
            grid_action = grid_actions[i, 0]
            analytical_action = analytical_actions[i, 0]
            
            # Apply scaling if available
            if scaler is not None:
                if hasattr(scaler, 'scale_x_'):
                    state_scaled = state.reshape(1, -1) / scaler.scale_x_.reshape(1, -1)
                    state_scaled = state_scaled[0]
                else:
                    dummy_action = np.array([[0.5 * (self.u_bounds[0] + self.u_bounds[1])]])
                    state_action_temp = np.concatenate([state.reshape(1, -1), dummy_action], axis=1)
                    state_action_scaled = scaler.transform(state_action_temp)
                    state_scaled = state_action_scaled[0, :4]
            else:
                state_scaled = state
            
            # Evaluate Q-function for grid policy
            try:
                if scaler is not None:
                    if hasattr(scaler, 'scale_u_'):
                        scale_u = float(scaler.scale_u_.reshape(-1)[0])
                    else:
                        scale_u = float(scaler.scale_[-1])
                    u_scaled = float(grid_action) / scale_u
                    state_action_scaled = np.concatenate([state_scaled, [u_scaled]])
                else:
                    state_action_scaled = np.concatenate([state_scaled, [grid_action]])
                
                phi = poly.transform(state_action_scaled.reshape(1, -1))
                grid_q_val = phi @ Q_matrix @ phi.T
                grid_q_values.append(float(grid_q_val[0, 0]))
            except:
                grid_q_values.append(np.inf)
            
            # Evaluate Q-function for analytical policy
            try:
                if scaler is not None:
                    if hasattr(scaler, 'scale_u_'):
                        scale_u = float(scaler.scale_u_.reshape(-1)[0])
                    else:
                        scale_u = float(scaler.scale_[-1])
                    u_scaled = float(analytical_action) / scale_u
                    state_action_scaled = np.concatenate([state_scaled, [u_scaled]])
                else:
                    state_action_scaled = np.concatenate([state_scaled, [analytical_action]])
                
                phi = poly.transform(state_action_scaled.reshape(1, -1))
                analytical_q_val = phi @ Q_matrix @ phi.T
                analytical_q_values.append(float(analytical_q_val[0, 0]))
            except:
                analytical_q_values.append(np.inf)
        
        grid_q_values = np.array(grid_q_values)
        analytical_q_values = np.array(analytical_q_values)
        
        # Compute Q-value differences
        q_value_differences = np.abs(grid_q_values - analytical_q_values)
        max_q_difference = np.max(q_value_differences)
        mean_q_difference = np.mean(q_value_differences)
        
        # Check which policy is better (lower Q-value is better)
        grid_better = np.sum(grid_q_values < analytical_q_values)
        analytical_better = np.sum(analytical_q_values < grid_q_values)
        ties = len(test_states) - grid_better - analytical_better
        
        if verbose:
            print(f"\n=== Policy Method Comparison ===")
            print(f"Number of test states: {len(test_states)}")
            print(f"Grid search points: {n_grid_points}")
            print(f"\nAction Differences:")
            print(f"  Max difference: {max_difference:.6f}")
            print(f"  Mean difference: {mean_difference:.6f}")
            print(f"  Std difference: {std_difference:.6f}")
            print(f"\nQ-value Differences:")
            print(f"  Max Q-value difference: {max_q_difference:.6f}")
            print(f"  Mean Q-value difference: {mean_q_difference:.6f}")
            print(f"\nPolicy Performance:")
            print(f"  Grid search better: {grid_better}")
            print(f"  Analytical better: {analytical_better}")
            print(f"  Ties: {ties}")
            print(f"  Grid search Q-value - mean: {np.mean(grid_q_values):.6f}, std: {np.std(grid_q_values):.6f}")
            print(f"  Analytical Q-value - mean: {np.mean(analytical_q_values):.6f}, std: {np.std(analytical_q_values):.6f}")
            
            # Show some example comparisons
            print(f"\nExample comparisons (first 5 states):")
            for i in range(min(5, len(test_states))):
                print(f"  State {i}: grid_action={grid_actions[i,0]:.4f}, analytical_action={analytical_actions[i,0]:.4f}, "
                      f"diff={action_differences[i,0]:.6f}")
                print(f"    Grid Q-value: {grid_q_values[i]:.6f}, Analytical Q-value: {analytical_q_values[i]:.6f}")
        
        return {
            "action_differences": action_differences,
            "max_action_difference": max_difference,
            "mean_action_difference": mean_difference,
            "std_action_difference": std_difference,
            "q_value_differences": q_value_differences,
            "max_q_difference": max_q_difference,
            "mean_q_difference": mean_q_difference,
            "grid_q_values": grid_q_values,
            "analytical_q_values": analytical_q_values,
            "grid_better_count": grid_better,
            "analytical_better_count": analytical_better,
            "ties_count": ties,
            "test_states": test_states,
            "grid_actions": grid_actions,
            "analytical_actions": analytical_actions
        }