import numpy as np
from dynamical_systems import dlqr
import matplotlib.pyplot as plt
import os
from config import GAMMA, M_C, M_P, L, DT, C_CART_POLE, RHO_CART_POLE, U_BOUNDS, DEGREE

class PolicyExtractor:
    """
    Policy extractor with better numerical stability and optimization
    """
    
    def __init__(self, M_c=None, M_p=None, l=None, dt=None, gamma=GAMMA, 
                 degree=DEGREE, u_bounds=None, C=None, rho=None):
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
        """
        # Cartpole parameters (with defaults)
        self.M_c = M_c if M_c is not None else M_C
        self.M_p = M_p if M_p is not None else M_P
        self.l = l if l is not None else L
        self.dt = dt if dt is not None else DT
        
        # Common parameters
        self.gamma = gamma
        self.degree = degree
        self.u_bounds = u_bounds if u_bounds is not None else U_BOUNDS
        
        # LQR parameters (with defaults)
        self.C = C if C is not None else C_CART_POLE
        self.rho = rho if rho is not None else RHO_CART_POLE
        
        # Numerical stability parameters
        self.regularization = 1e-6
        self.optimization_tolerance = 1e-8
        self.max_optimization_iterations = 1000
        
        # Plotting parameters
        self.plots_base_dir = "figures"
        self.degree = degree
        
    def extract_lqr_policy(self, system):
        """
        Extract LQR policy from system (supports both cartpole and LQR systems)
        
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
            # This is a cartpole system
            system_type = 'cartpole'
        else:
            raise ValueError(f"Unknown system type: {type(system)}")
        
        if system_type == 'lqr':
            # For LQR systems, get the optimal solution directly
            P_lqr, K_lqr, q_lqr = system.optimal_solution()
        elif system_type == 'cartpole':
            # For cartpole systems, linearize and solve LQR
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
            u_clipped = np.clip(u.T, self.u_bounds[0], self.u_bounds[1])
            # Ensure we return a scalar for single control systems
            if u_clipped.size == 1:
                return float(u_clipped.item())
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

    def extract_policy_grid_search(self, system, Q_matrix, poly_scaler=None, n_grid_points=100):
        """
        Extract policy using grid search - serves as baseline for verification
        
        Args:
            system: dynamical system
            Q_matrix: Q-matrix from moment matching
            poly: polynomial feature transformer
            scaler: optional feature scaler
            n_grid_points: number of grid points for action search
            
        Returns:
            policy function that takes states and returns optimal actions
        """
        def policy_grid_search(x):
            x = np.atleast_2d(x)
            n = x.shape[0]
            actions = np.empty((n, 1), dtype=float)

            for i in range(n):
                state = x[i]

                def q_function(u_val):
                    """Evaluate Q-function for given action"""
                    try:
                        # Create state-action pair and let poly_scaler handle all transformations
                        state_action = np.concatenate([state, [u_val]])
                        phi = poly_scaler.transform(state_action.reshape(1, -1))
                        
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

    def extract_policy_scipy_optimization(self, system, Q_matrix, poly_scaler=None):
        """
        Extract policy using scipy optimization - more efficient than grid search
        """
        from scipy.optimize import minimize_scalar, minimize
        
        def policy_scipy_optimization(x):
            x = np.atleast_2d(x)
            n = x.shape[0]
            actions = np.empty((n, 1), dtype=float)

            for i in range(n):
                state = x[i]

                def q_function(u_val):
                    """Evaluate Q-function for given action"""
                    try:
                        # Create state-action pair and let poly_scaler handle all transformations
                        state_action = np.concatenate([state, [u_val]])
                        phi = poly_scaler.transform(state_action.reshape(1, -1))
                        
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
    
    def extract_moment_matching_policy_analytical(self, system, Q_matrix, poly_scaler):
        """
        Analytical policy extraction using gradient-based approach
        
        This method computes the optimal action analytically when possible,
        falling back to numerical optimization when needed.
        """
        n_phi = poly_scaler.poly.n_output_features_
        assert Q_matrix.shape == (n_phi, n_phi), (
            f"Q_matrix shape {Q_matrix.shape} != ({n_phi},{n_phi})"
        )
        
        def mm_policy_analytical(x):
            x = np.atleast_2d(x)
            n = x.shape[0]
            actions = np.empty((n, 1), dtype=float)

            for i in range(n):
                state = x[i]
                
                # Apply scaling if available
                if poly_scaler is not None:
                    # Use new PolynomialFeatureScaler approach
                    if hasattr(poly_scaler, 'scaler_x') and poly_scaler.scaler_x is not None:
                        state_scaled = poly_scaler.scaler_x.transform(state.reshape(1, -1))[0]
                    else:
                        # No separate scaling, use state as is
                        state_scaled = state
                else:
                    state_scaled = state

                try:
                    # For polynomial features, we can compute the gradient analytically
                    # The objective is: min_u phi(x,u)^T Q phi(x,u)
                    # where phi(x,u) = [x, u, x^2, xu, u^2, ...]
                    
                    # Try analytical solution first
                    if self.degree == 1:
                        # print(f"Solving linear policy analytically for state {i}")
                        # For linear features: phi = [x1, x2, x3, x4, u]
                        # Objective: [x, u]^T Q [x, u] = x^T Q_xx x + 2 x^T Q_xu u + u^T Q_uu u
                        # Optimal u = -Q_uu^{-1} Q_xu^T x
                        
                        # For degree 1, features are [x1, x2, x3, x4, u], so Q is (5, 5)
                        n_states = len(state_scaled)  # Should be 4
                        Q_xx = Q_matrix[:n_states, :n_states]
                        Q_ux = Q_matrix[n_states:, :n_states]
                        Q_xu = Q_matrix[:n_states, n_states:]
                        Q_uu = Q_matrix[n_states:, n_states:]

                        # For linear features: phi = [x1, x2, x3, x4, u]
                        # Q-function: phi^T Q phi = x^T Q_xx x + 2x^T Q_xu u + u^T Q_uu u
                        # Optimal u: ∂/∂u = 0 → 2Q_xu^T x + 2Q_uu u = 0 → u = -Q_uu^{-1} Q_xu^T x
                        
                        # Use scaled state with Q-matrix (which was learned on scaled features)
                        try:
                            u_opt_scaled = -np.linalg.solve(Q_uu, Q_xu.T @ state_scaled)
                            u_opt_scaled = float(u_opt_scaled.item())
                        except np.linalg.LinAlgError:
                            # Use regularized solve for ill-conditioned matrices
                            cond_Q_uu = np.linalg.cond(Q_uu)
                            print(f"Warning: Q_uu condition number {cond_Q_uu:.2e}, using regularized solve")
                            reg = 1e-6 * np.trace(Q_uu) / Q_uu.shape[0]
                            u_opt_scaled = -np.linalg.solve(Q_uu + reg * np.eye(Q_uu.shape[0]), Q_xu.T @ state_scaled)
                            u_opt_scaled = float(u_opt_scaled.item())
                        
                        # Transform the scaled action back to original space if needed
                        if hasattr(poly_scaler, 'scaler_u') and poly_scaler.scaler_u is not None:
                            u_opt = poly_scaler.scaler_u.inverse_transform(np.array([u_opt_scaled]).reshape(1, -1))[0, 0]
                        else:
                            u_opt = u_opt_scaled
                        
                        u_opt = float(np.clip(u_opt, self.u_bounds[0], self.u_bounds[1]))
                        actions[i, 0] = u_opt
                        continue
                    
                    elif self.degree == 2:
                        # print(f"Solving quadratic policy analytically for state {i}")
                        # For degree 2 polynomial features: phi = [x, u, x^2, xu, u^2, ...]
                        # The objective is quartic in u: a4*u^4 + a3*u^3 + a2*u^2 + a1*u + a0
                        # Find critical points by solving the derivative
                        
                        u_opt = self._solve_quadratic_policy_analytical(state_scaled, Q_matrix, poly_scaler)
                        # print(f"u_opt: {u_opt}")
                        if u_opt is not None and np.isfinite(u_opt):
                            actions[i, 0] = u_opt
                        else:
                            # Fallback to center of action bounds if analytical solution fails
                            print(f"Fallback to center of action bounds")
                            actions[i, 0] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                        continue

                    elif self.degree > 2:
                        u_opt = self.extract_policy_scipy_optimization(system, Q_matrix, poly_scaler=poly_scaler)(state_scaled)
                        if u_opt is not None and np.isfinite(u_opt):
                            # u_clipped = float(np.clip(u_opt, self.u_bounds[0], self.u_bounds[1]))
                            actions[i, 0] = u_opt
                            # if abs(u_opt - u_clipped) > 1e-6:
                            #     print(f"Warning: Control action clipped from {u_opt:.4f} to {u_clipped:.4f} for state {i}")
                        else:
                            # Fallback to center of action bounds if analytical solution fails
                            print(f"Fallback to center of action bounds")
                            actions[i, 0] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                        continue
                    
                    print(f"Analytical optimization failed for state {i}, using fallback")
                    actions[i, 0] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                        
                except Exception as e:
                    print(f"Exception in analytical optimization for state {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    actions[i, 0] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])

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

        return mm_policy_analytical
    
    def _solve_quadratic_policy_analytical(self, state, Q_matrix, poly_scaler=None):
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
            state: state vector (scaled if applicable)
            Q_matrix: Q-matrix from moment matching
            poly: polynomial feature transformer
            
        Returns:
            optimal u or None if analytical solution fails
        """
        x = np.asarray(state).reshape(-1)
        # print(f"x: {x}")
        
        # Handle action bounds in scaled space if scaling is used
        if poly_scaler is not None and hasattr(poly_scaler, 'scaler_u') and poly_scaler.scaler_u is not None:
            # For multi-dimensional control systems, we need to create full control vectors
            # with the same bounds for all control dimensions
            du = poly_scaler.du if hasattr(poly_scaler, 'du') else 1
            umin_vec = np.full((1, du), self.u_bounds[0])
            umax_vec = np.full((1, du), self.u_bounds[1])
            
            umin_scaled = poly_scaler.scaler_u.transform(umin_vec)
            umax_scaled = poly_scaler.scaler_u.transform(umax_vec)
            
            # For now, use the first control dimension's bounds (assuming all controls have same bounds)
            umin = float(umin_scaled[0, 0])
            umax = float(umax_scaled[0, 0])
            # print(f"umin {umin}, umax {umax}")
        else:
            umin, umax = self.u_bounds

        powers = poly_scaler.poly.powers_            # shape (D, n_inputs): exponents for [x1,...,xn,u1,...,um]
        D = powers.shape[0]
        n_inputs = powers.shape[1]
        
        # Determine state and control dimensions
        # Try to get dx from poly_scaler, but handle None case
        if hasattr(poly_scaler, 'dx') and poly_scaler.dx is not None:
            dx = int(poly_scaler.dx)
        else:
            # Infer from state vector length or assume n_inputs - 1 (last dimension is control)
            if len(x) > 0:
                dx = len(x)
            else:
                dx = n_inputs - 1  # Assume last dimension is control
        
        # Try to get du from poly_scaler, but handle None case
        if hasattr(poly_scaler, 'du') and poly_scaler.du is not None:
            du = int(poly_scaler.du)
        else:
            du = n_inputs - dx  # Infer from difference, default to 1
            if du <= 0:
                du = 1  # Fallback to 1 if inference fails
        
        assert Q_matrix.shape == (D, D)

        # For each feature j: phi_j(x,u) = c_j(x) * u^{e_j}
        # c_j depends only on x (product of x_i^power), e_j is the power on u.
        c = np.ones(D, dtype=float)
        # For multi-dimensional control, we need to handle the control exponents differently
        # For now, assume we're optimizing over the first control variable
        if du == 1:
            e = powers[:, dx].astype(int)     # exponents of u: 0,1,2 for degree 2
        else:
            # For multi-dimensional control, use the first control variable's exponent
            e = powers[:, dx].astype(int)     # exponents of u1: 0,1,2 for degree 2
            
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
        
        # Transform back to original action space if scaling was used
        if poly_scaler is not None and hasattr(poly_scaler, 'scaler_u') and poly_scaler.scaler_u is not None:
            # For multi-dimensional control systems, we need to create a full control vector
            du = poly_scaler.du if hasattr(poly_scaler, 'du') else 1
            if du == 1:
                u_opt = poly_scaler.scaler_u.inverse_transform(np.array([u_opt_scaled]).reshape(1,-1))[0]
            else:
                # For multi-dimensional control, create a vector with the optimized value for the first control
                # and zeros for the other controls (assuming we're optimizing over the first control)
                u_opt_vec = np.zeros(du)
                u_opt_vec[0] = u_opt_scaled
                u_opt_full = poly_scaler.scaler_u.inverse_transform(u_opt_vec.reshape(1,-1))[0]
                u_opt = u_opt_full[0]  # Return only the first control value
        else:
            u_opt = u_opt_scaled
            
        return u_opt
    
    
    def _simulate_policy(self, system, policy, initial_state, horizon=10000):
        """
        Simulate a policy and compute total cost with error handling
        
        Args:
            system: dynamical system
            policy: policy function
            initial_state: initial state
            horizon: simulation horizon
            
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
            
            # Ensure u_current is a scalar
            if hasattr(u_current, '__len__') and len(u_current) > 0:
                u_current = float(u_current[0])
            elif hasattr(u_current, 'item'):
                u_current = u_current.item()
            else:
                u_current = float(u_current)
            
            inputs.append(u_current)
            # Check for invalid control values
            if not np.isfinite(u_current):
                print(f"u_current is not finite: {u_current}")
                success = False
                break
            
            # Compute cost
            try:
                cost = system.cost(x_current.reshape(1, -1), 
                                  np.array([[u_current]]))
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
                x_current = system.step(x_current, u_current)
                trajectory.append(x_current.copy())
            except Exception:
                print(f"Exception in state update")
                success = False
                break
            
            # Check for divergence during simulation
            # if np.any(np.abs(x_current) > 100):
            #     success = False
            #     total_cost = np.inf
            #     break
        # Check stability based on last 100 samples if we have enough data
        # if success and len(trajectory) >= 100:
        #     # Get the last 100 states
        #     last_100_states = np.array(trajectory[-100:])
            
        #     # Check if all last 100 states are sufficiently close to equilibrium (origin)
        #     equilibrium_tolerance = 10.0  # Adjust this threshold as needed
        #     distances_to_equilibrium = np.linalg.norm(last_100_states, axis=1)
            
        #     # If any of the last 100 states are too far from equilibrium, mark as unstable
        #     if not np.all(distances_to_equilibrium <= equilibrium_tolerance):
        #         success = False
        #         total_cost = np.inf
        
        # print("Total cost: ", total_cost)
        
        return np.array(trajectory), total_cost, success, inputs
    
    def plot_policy_comparison(self, lqr_inputs, mm_inputs, N_samples, save_path=None):
        """
        Plot control input comparison and save to structured directory
        
        Args:
            lqr_inputs: LQR control inputs (T,) array
            mm_inputs: Moment matching control inputs (T,) array
            N_samples: Number of samples used in the experiment
            save_path: Optional custom path to save the plot (overrides default structure)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Convert time steps to actual time using DT
        time_lqr = np.arange(len(lqr_inputs)) * self.dt
        time_mm = np.arange(len(mm_inputs)) * self.dt
        
        # Plot control inputs
        ax.plot(time_lqr, lqr_inputs, 'b-', label='LQR', linewidth=2, alpha=0.8)
        ax.plot(time_mm, mm_inputs, 'r-', label='Moment Matching', linewidth=2, alpha=0.8)
        
        # Formatting with proper units
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control Force (N)')
        ax.set_title('Control Input Comparison: LQR vs Moment Matching')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add action bounds as horizontal lines
        ax.axhline(y=self.u_bounds[0], color='gray', linestyle='--', alpha=0.5, label=f'Lower Bound ({self.u_bounds[0]} N)')
        ax.axhline(y=self.u_bounds[1], color='gray', linestyle='--', alpha=0.5, label=f'Upper Bound ({self.u_bounds[1]} N)')
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
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
        # State dimension names with proper units
        state_names = ['Position (m)', 'Velocity (m/s)', 'Angle (rad)', 'Angular Velocity (rad/s)']
        state_units = ['m', 'm/s', 'rad', 'rad/s']
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Convert time steps to actual time using DT
        time_lqr = np.arange(len(lqr_traj)) * self.dt
        time_mm = np.arange(len(mm_traj)) * self.dt
        
        # Plot each state dimension
        for i in range(4):
            ax = axes[i]
            
            # Plot LQR trajectory
            ax.plot(time_lqr, lqr_traj[:, i], 'b-', label='LQR', linewidth=2, alpha=0.8)
            
            # Plot MM trajectory
            ax.plot(time_mm, mm_traj[:, i], 'r-', label='Moment Matching', linewidth=2, alpha=0.8)
            
            # Formatting with proper units
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'State Value ({state_units[i]})')
            ax.set_title(state_names[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add some styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Overall title
        fig.suptitle('Trajectory Comparison: LQR vs Moment Matching', fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Generate save path if not provided
        if save_path is None:
            save_path = self._generate_plot_filename(N_samples, "trajectory_comparison")
        
        # Save the plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to {save_path}")
        
        plt.close(fig)  # Close the figure to free memory
        
        return fig
    
    def plot_comprehensive_comparison(self, lqr_trajs, mm_trajs, lqr_inputs_list, mm_inputs_list, N_samples, save_path=None):
        """
        Plot comprehensive comparison with both trajectories and control inputs
        
        Args:
            lqr_trajs: List of LQR trajectories, each (T, n_states) array
            mm_trajs: List of moment matching trajectories, each (T, n_states) array
            lqr_inputs_list: List of LQR control inputs, each (T,) array
            mm_inputs_list: List of moment matching control inputs, each (T,) array
            N_samples: Number of samples used in the experiment
            save_path: Optional custom path to save the plot (overrides default structure)
        """
        # Determine the actual state dimension from the trajectory data
        if len(lqr_trajs) > 0 and len(lqr_trajs[0]) > 0:
            n_states = lqr_trajs[0].shape[1]
        elif len(mm_trajs) > 0 and len(mm_trajs[0]) > 0:
            n_states = mm_trajs[0].shape[1]
        else:
            raise ValueError("Cannot determine state dimension from empty trajectories")
        
        # State dimension names with proper units - dynamically generate based on n_states
        if n_states == 4:
            # Cartpole system
            state_names = ['Position (m)', 'Velocity (m/s)', 'Angle (rad)', 'Angular Velocity (rad/s)']
            state_units = ['m', 'm/s', 'rad', 'rad/s']
        elif n_states == 2:
            # 2D LQR system
            state_names = ['State 1', 'State 2']
            state_units = ['units', 'units']
        else:
            # Generic n-dimensional system
            state_names = [f'State {i+1}' for i in range(n_states)]
            state_units = ['units'] * n_states
        
        # Create figure with appropriate number of subplots (n_states + 1 control)
        n_plots = n_states + 1
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
            
            # Add labels only once
            if i == 0:
                ax.plot([], [], 'b-', label='LQR', linewidth=2)
                ax.plot([], [], 'r-', label='Moment Matching', linewidth=2)
            
            # Formatting with proper units
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'State Value ({state_units[i]})')
            ax.set_title(state_names[i])
            # ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add some styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Plot control inputs in the last subplot
        ax = axes[n_states]
        for j in range(len(lqr_inputs_list)):
            if (j < len(mm_inputs_list) and 
                len(lqr_inputs_list[j]) > 0 and len(mm_inputs_list[j]) > 0):
                # Convert time steps to actual time using DT
                time_lqr = np.arange(len(lqr_inputs_list[j])) * self.dt
                time_mm = np.arange(len(mm_inputs_list[j])) * self.dt
                
                # Plot LQR inputs
                ax.plot(time_lqr, lqr_inputs_list[j], 'b-', linewidth=2, alpha=0.8)
                
                # Plot MM inputs
                ax.plot(time_mm, mm_inputs_list[j], 'r-', linewidth=2, alpha=0.8)
        
        # Add labels for control inputs
        ax.plot([], [], 'b-', label='LQR', linewidth=2)
        ax.plot([], [], 'r-', label='Moment Matching', linewidth=2)
        
        # Add action bounds
        ax.axhline(y=self.u_bounds[0], color='gray', linestyle='--', alpha=0.5, label=f'Lower Bound ({self.u_bounds[0]} N)')
        ax.axhline(y=self.u_bounds[1], color='gray', linestyle='--', alpha=0.5, label=f'Upper Bound ({self.u_bounds[1]} N)')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control Force (N)')
        ax.set_title('Control Input Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Hide any unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # Overall title
        fig.suptitle('Comprehensive Policy Comparison: LQR vs Moment Matching', fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Generate save path if not provided
        if save_path is None:
            save_path = self._generate_plot_filename(N_samples, "comprehensive_comparison", "smoothened_noscaling")
        
        # Save the plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive comparison plot saved to {save_path}")
        
        plt.close(fig)  # Close the figure to free memory
        
        return fig
    
    def compare_policies(self, system, lqr_policy, mm_policy, test_states, N_samples, horizon=1000):
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
            
        Returns:
            dict: comparison results with statistics
        """
        n_tests = test_states.shape[0]
        lqr_costs = []
        mm_costs = []
        lqr_success = []
        mm_success = []
        convergent_costs_lqr = []  # Costs only for convergent cases
        convergent_costs_mm = []   # Costs only for convergent cases
        lqr_trajs = []
        mm_trajs = []
        lqr_inputs_list = []
        mm_inputs_list = []
        
        for i in range(n_tests):
            initial_state = test_states[i]
            
            # Simulate LQR policy
            try:
                # print(f"Simulating LQR policy with initial state: {initial_state}")
                lqr_traj, lqr_cost, lqr_succ, lqr_inputs_single = self._simulate_policy(
                    system, lqr_policy, initial_state, horizon
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
            
            # Simulate moment matching policy
            try:
                # print(f"Simulating MM policy with initial state: {initial_state}")
                mm_traj, mm_cost, mm_succ, mm_inputs_single = self._simulate_policy(
                    system, mm_policy, initial_state, horizon
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
            
            # Only include costs if both methods converged for this starting point
            if lqr_success[-1] and mm_success[-1]:
                convergent_costs_lqr.append(lqr_cost)
                convergent_costs_mm.append(mm_cost)
        
        # Compute robust statistics
        def safe_mean(data, default=np.inf):
            return np.mean(data) if len(data) > 0 else default
        
        # Calculate statistics for convergent cases only
        n_convergent = len(convergent_costs_lqr)
        convergent_lqr_mean = safe_mean(convergent_costs_lqr)
        convergent_mm_mean = safe_mean(convergent_costs_mm)
        
        print(f"Policy comparison: {n_convergent}/{n_tests} starting points had both methods convergent")
        print(f"Convergent LQR mean cost: {convergent_lqr_mean:.4f}")
        print(f"Convergent MM mean cost: {convergent_mm_mean:.4f}")

        # Convert lists to arrays for plotting
        if len(lqr_trajs) > 0 and len(mm_trajs) > 0:
            # Filter out empty trajectories
            valid_lqr_trajs = [traj for traj in lqr_trajs if len(traj) > 0]
            valid_mm_trajs = [traj for traj in mm_trajs if len(traj) > 0]
            valid_lqr_inputs = [inp for inp in lqr_inputs_list if len(inp) > 0]
            valid_mm_inputs = [inp for inp in mm_inputs_list if len(inp) > 0]
            
            if len(valid_lqr_trajs) > 0 and len(valid_mm_trajs) > 0:
                self.plot_comprehensive_comparison(valid_lqr_trajs, valid_mm_trajs, valid_lqr_inputs, valid_mm_inputs, N_samples)
        
        return {
            # Number of convergent cases
            "n_convergent": n_convergent,
            "n_total": n_tests,
            "convergent_ratio": n_convergent / n_tests if n_tests > 0 else 0.0,
            
            # Backward compatibility (using convergent costs)
            "lqr_costs": convergent_lqr_mean,
            "mm_costs": convergent_mm_mean,
            "lqr_success": np.mean(lqr_success),
            "mm_success": np.mean(mm_success),
        }
    
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