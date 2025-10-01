import numpy as np
from dynamical_systems import dlqr
import matplotlib.pyplot as plt
from config import GAMMA, M_C, M_P, L, DT, C_CART_POLE, RHO_CART_POLE, U_BOUNDS

class PolicyExtractor:
    """
    Policy extractor with better numerical stability and optimization
    """
    
    def __init__(self, M_c=M_C, M_p=M_P, l=L, dt=DT, gamma=GAMMA, 
                 degree=2, u_bounds=U_BOUNDS):
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
        self.M_c = M_c
        self.M_p = M_p
        self.l = l
        self.dt = dt
        self.gamma = gamma
        self.degree = degree
        self.u_bounds = u_bounds
        
        # Cost matrices for LQR
        self.C = C_CART_POLE
        self.rho = RHO_CART_POLE
        
        # Numerical stability parameters
        self.regularization = 1e-6
        self.optimization_tolerance = 1e-8
        self.max_optimization_iterations = 1000
        
    def extract_lqr_policy(self, system):
        """
        Extract LQR policy from linearized system
        
        Returns:
            policy_func: function that takes state and returns control action
            K_lqr: LQR gain matrix
            Q_lqr: LQR Q-function matrix
        """
        # Get linearized dynamics
        A_d, B_d = system.linearized_system()
        
        # Solve LQR
        lqr_system = dlqr(A_d, B_d, self.C, self.rho, self.gamma)
        P_lqr, K_lqr, q_lqr = lqr_system.optimal_solution()
        
        # Create policy function
        def lqr_policy(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            u = -K_lqr @ x.T
            return np.clip(u.T, self.u_bounds[0], self.u_bounds[1])
            # return u.T
        
        return lqr_policy, K_lqr, P_lqr

    def extract_policy_scipy_optimization(self, system, Q_matrix, poly, scaler=None):
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
                
                # Apply scaling if available
                if scaler is not None:
                    dummy_action = np.array([[0.5 * (self.u_bounds[0] + self.u_bounds[1])]])
                    state_action_temp = np.concatenate([state.reshape(1, -1), dummy_action], axis=1)
                    state_action_scaled = scaler.transform(state_action_temp)
                    state_scaled = state_action_scaled[0, :4]
                else:
                    state_scaled = state

                def q_function(u_val):
                    """Evaluate Q-function for given action"""
                    try:
                        # Create state-action pair
                        if scaler is not None:
                            u_scaled_temp = np.concatenate([state_scaled.reshape(1, -1), [[u_val]]], axis=1)
                            u_scaled = scaler.transform(u_scaled_temp)[0, 4]
                            state_action_scaled = np.concatenate([state_scaled, [u_scaled]])
                        else:
                            state_action_scaled = np.concatenate([state_scaled, [u_val]])
                        
                        # Transform to polynomial features
                        phi = poly.transform(state_action_scaled.reshape(1, -1))
                        
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

            return actions

        return policy_scipy_optimization
    
    def extract_moment_matching_policy_analytical(self, system, Q_matrix, poly, scaler=None):
        """
        Analytical policy extraction using gradient-based approach
        
        This method computes the optimal action analytically when possible,
        falling back to numerical optimization when needed.
        """
        n_phi = poly.n_output_features_
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
                if scaler is not None:
                    # Use global scaler - need to create dummy action for scaling
                    # Use middle of action bounds as dummy for consistent scaling
                    dummy_action = np.array([[0.0]])
                    state_action_temp = np.concatenate([state.reshape(1, -1), dummy_action], axis=1)
                    state_action_scaled = scaler.transform(state_action_temp)
                    state_scaled = state_action_scaled[0, :4]  # Extract scaled state part
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
                        
                        if scaler is not None:
                            # Use scaled state with Q-matrix (which was learned on scaled features)
                            try:
                                u_opt_scaled = -np.linalg.solve(Q_uu, Q_xu.T @ state_scaled)
                                u_opt_scaled = float(u_opt_scaled[0]) if hasattr(u_opt_scaled, '__len__') else float(u_opt_scaled)
                            except np.linalg.LinAlgError:
                                # Use regularized solve for ill-conditioned matrices
                                cond_Q_uu = np.linalg.cond(Q_uu)
                                print(f"Warning: Q_uu condition number {cond_Q_uu:.2e}, using regularized solve")
                                reg = 1e-6 * np.trace(Q_uu) / Q_uu.shape[0]
                                u_opt_scaled = -np.linalg.solve(Q_uu + reg * np.eye(Q_uu.shape[0]), Q_xu.T @ state_scaled)
                                u_opt_scaled = float(u_opt_scaled[0]) if hasattr(u_opt_scaled, '__len__') else float(u_opt_scaled)
                            
                            # Transform the scaled action back to original space
                            state_action_opt_scaled = np.concatenate([state_scaled, [u_opt_scaled]])
                            state_action_opt_original = scaler.inverse_transform(state_action_opt_scaled.reshape(1, -1))[0]
                            u_opt = state_action_opt_original[4]
                        else:
                            # No scaling - use Q-matrix directly
                            try:
                                u_opt = -np.linalg.solve(Q_uu, Q_xu.T @ state_scaled)
                                u_opt = float(u_opt[0]) if hasattr(u_opt, '__len__') else float(u_opt)
                            except np.linalg.LinAlgError:
                                # Use regularized solve for ill-conditioned matrices
                                cond_Q_uu = np.linalg.cond(Q_uu)
                                print(f"Warning: Q_uu condition number {cond_Q_uu:.2e}, using regularized solve")
                                reg = 1e-6 * np.trace(Q_uu) / Q_uu.shape[0]
                                u_opt = -np.linalg.solve(Q_uu + reg * np.eye(Q_uu.shape[0]), Q_xu.T @ state_scaled)
                                u_opt = float(u_opt[0]) if hasattr(u_opt, '__len__') else float(u_opt)
                        
                        u_opt = float(np.clip(u_opt, self.u_bounds[0], self.u_bounds[1]))
                        actions[i, 0] = u_opt
                        continue
                    
                    elif self.degree == 2:
                        print(f"Solving quadratic policy analytically for state {i}")
                        # For degree 2 polynomial features: phi = [x, u, x^2, xu, u^2, ...]
                        # The objective is quadratic in u: a*u^2 + b*u + c
                        # where a, b, c depend on x and Q
                        # Optimal u = -b/(2*a) (if a > 0)
                        
                        u_opt = self._solve_quadratic_policy_analytical(state_scaled, Q_matrix, poly, scaler)
                        if u_opt is not None and np.isfinite(u_opt):
                            # u_clipped = float(np.clip(u_opt, self.u_bounds[0], self.u_bounds[1]))
                            # actions[i, 0] = u_clipped
                            # if abs(u_opt - u_clipped) > 1e-6:
                            #     print(f"Warning: Control action clipped from {u_opt:.4f} to {u_clipped:.4f} for state {i}")
                            actions[i, 0] = u_opt
                            continue
                    
                    print(f"Analytical optimization failed for state {i}, using fallback")
                    actions[i, 0] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])
                        
                except Exception as e:
                    print(f"Exception in analytical optimization for state {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    actions[i, 0] = 0.5 * (self.u_bounds[0] + self.u_bounds[1])

            return actions

        return mm_policy_analytical
    
    def _solve_quadratic_policy_analytical(self, state, Q_matrix, poly, scaler=None):
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
        assert x.shape[0] == 4, "x must be 4D"
        
        # Handle action bounds in scaled space if scaling is used
        if scaler is not None:
            # Transform action bounds to scaled space using global scaler
            # Use the current state for more accurate scaling
            umin_temp = np.concatenate([x.reshape(1, -1), [[self.u_bounds[0]]]], axis=1)
            umax_temp = np.concatenate([x.reshape(1, -1), [[self.u_bounds[1]]]], axis=1)
            umin_scaled = scaler.transform(umin_temp)[0, 4]  # Extract action component
            umax_scaled = scaler.transform(umax_temp)[0, 4]  # Extract action component
            umin, umax = umin_scaled, umax_scaled
        else:
            umin, umax = self.u_bounds

        powers = poly.powers_            # shape (D, 5): exponents for [x1,x2,x3,x4,u]
        D = powers.shape[0]
        assert powers.shape[1] == 5, "poly must be fitted on 5 inputs [x1..x4,u]"
        assert Q_matrix.shape == (D, D)

        # For each feature j: phi_j(x,u) = c_j(x) * u^{e_j}
        # c_j depends only on x (product of x_i^power), e_j is the power on u.
        c = np.ones(D, dtype=float)
        e = powers[:, 4].astype(int)     # exponents of u: 0,1,2 for degree 2
        for j in range(D):
            # multiply x_i^{p_ji} for i=0..3
            for i in range(4):
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

        # Derivative cubic: 4 a4 u^3 + 3 a3 u^2 + 2 a2 u + a1 = 0
        # Handle degenerate lower-degree cases robustly
        coeffs = np.array([4*a4, 3*a3, 2*a2, a1], dtype=float)

        # Candidate set = stationary points + the two bounds
        candidates = [umin, umax]

        # If the polynomial is actually lower degree, trim leading zeros
        first_nz = np.argmax(np.abs(coeffs) > 1e-12)  # More robust threshold
        if np.any(np.abs(coeffs) > 1e-12):
            try:
                roots = np.roots(coeffs[first_nz:])  # closed-form cubic/quadratic/linear solver
                for r in roots:
                    if np.isreal(r) and not np.isnan(np.real(r)):
                        u = float(np.real(r))
                        # Only add candidates within bounds
                        if umin <= u <= umax:
                            candidates.append(u)
            except (np.linalg.LinAlgError, ValueError):
                # If root finding fails, just use bounds
                pass

        # Evaluate objective J(u) = a4 u^4 + a3 u^3 + a2 u^2 + a1 u + a0
        def J(u):
            try:
                return (((a4*u + a3)*u + a2)*u + a1)*u + a0  # Correct Horner's method for quartic
            except (OverflowError, ValueError):
                return np.inf

        # Evaluate all candidates and find minimum
        vals = []
        valid_candidates = []
        for u in candidates:
            try:
                val = J(u)
                if np.isfinite(val):
                    vals.append(val)
                    valid_candidates.append(u)
                else:
                    vals.append(np.inf)
                    valid_candidates.append(u)
            except:
                vals.append(np.inf)
                valid_candidates.append(u)

        if not valid_candidates:
            return None
            
        min_idx = int(np.argmin(vals))
        u_opt_scaled = float(valid_candidates[min_idx])
        
        # Transform back to original action space if scaling was used
        if scaler is not None:
            # Use the current state for inverse transform
            u_opt_temp = np.concatenate([x.reshape(1, -1), [[u_opt_scaled]]], axis=1)
            u_opt_original = scaler.inverse_transform(u_opt_temp)
            u_opt = u_opt_original[0, 4]  # Extract action component
        else:
            u_opt = u_opt_scaled
            
        return u_opt
    
    
    def _simulate_policy(self, system, policy, initial_state, horizon=1000):
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
                    success = False
                    break
                total_cost += (self.gamma ** step) * cost
            except Exception:
                success = False
                break
            
            # Update state
            try:
                x_current = system.step(x_current, u_current)
                trajectory.append(x_current.copy())
            except Exception:
                success = False
                break
            
            # Check for divergence during simulation
            # if np.any(np.abs(x_current) > 100):
            #     success = False
            #     total_cost = np.inf
            #     break
        # Check stability based on last 100 samples if we have enough data
        if success and len(trajectory) >= 100:
            # Get the last 100 states
            last_100_states = np.array(trajectory[-100:])
            
            # Check if all last 100 states are sufficiently close to equilibrium (origin)
            equilibrium_tolerance = 1.0  # Adjust this threshold as needed
            distances_to_equilibrium = np.linalg.norm(last_100_states, axis=1)
            
            # If any of the last 100 states are too far from equilibrium, mark as unstable
            if not np.all(distances_to_equilibrium <= equilibrium_tolerance):
                success = False
                total_cost = np.inf
        
        # print("Total cost: ", total_cost)
        
        return np.array(trajectory), total_cost, success, inputs
    
    def plot_policy_comparison(self, lqr_inputs, mm_inputs, save_path=None):
        """
        Plot control input comparison
        
        Args:
            lqr_inputs: LQR control inputs (T,) array
            mm_inputs: Moment matching control inputs (T,) array
            save_path: Optional path to save the plot
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
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Control input comparison plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_trajectory_comparison(self, lqr_traj, mm_traj, save_path=None):
        """
        Plot trajectory comparison with 4 separate subplots for each state dimension
        
        Args:
            lqr_traj: LQR trajectory (T, 4) array
            mm_traj: Moment matching trajectory (T, 4) array
            save_path: Optional path to save the plot
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
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory comparison plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_comprehensive_comparison(self, lqr_trajs, mm_trajs, lqr_inputs_list, mm_inputs_list, save_path=None):
        """
        Plot comprehensive comparison with both trajectories and control inputs
        
        Args:
            lqr_trajs: List of LQR trajectories, each (T, 4) array
            mm_trajs: List of moment matching trajectories, each (T, 4) array
            lqr_inputs_list: List of LQR control inputs, each (T,) array
            mm_inputs_list: List of moment matching control inputs, each (T,) array
            save_path: Optional path to save the plot
        """
        # State dimension names with proper units
        state_names = ['Position (m)', 'Velocity (m/s)', 'Angle (rad)', 'Angular Velocity (rad/s)']
        state_units = ['m', 'm/s', 'rad', 'rad/s']
        
        # Create figure with 5 subplots (4 states + 1 control)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each state dimension
        for i in range(4):
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
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add some styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Plot control inputs in the 5th subplot
        ax = axes[4]
        for j in range(len(lqr_inputs_list)):
            if len(lqr_inputs_list[j]) > 0 and len(mm_inputs_list[j]) > 0:
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
        
        # Hide the 6th subplot
        axes[5].set_visible(False)
        
        # Overall title
        fig.suptitle('Comprehensive Policy Comparison: LQR vs Moment Matching', fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comprehensive comparison plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def compare_policies(self, system, lqr_policy, mm_policy, test_states, horizon=1000):
        """
        Policy comparison with better error handling and statistics.
        Only evaluates costs for starting points where both methods converge.
        
        Args:
            system: dynamical system
            lqr_policy: LQR policy function
            mm_policy: moment matching policy function
            test_states: array of test states
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
        # if len(lqr_trajs) > 0 and len(mm_trajs) > 0:
        #     # Filter out empty trajectories
        #     valid_lqr_trajs = [traj for traj in lqr_trajs if len(traj) > 0]
        #     valid_mm_trajs = [traj for traj in mm_trajs if len(traj) > 0]
        #     valid_lqr_inputs = [inp for inp in lqr_inputs_list if len(inp) > 0]
        #     valid_mm_inputs = [inp for inp in mm_inputs_list if len(inp) > 0]
            
        #     if len(valid_lqr_trajs) > 0 and len(valid_mm_trajs) > 0:
        #         self.plot_comprehensive_comparison(valid_lqr_trajs, valid_mm_trajs, valid_lqr_inputs, valid_mm_inputs)
        
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
            state_action_pairs_scaled = scaler.transform(state_action_pairs)
        else:
            state_action_pairs_scaled = state_action_pairs
        
        # Transform to polynomial features
        phi_features = poly.transform(state_action_pairs_scaled)
        
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