import numpy as np
from scipy.linalg import solve_discrete_are, inv
from scipy.optimize import fsolve

class dlqr:
    def __init__(self, A, B, C, rho, gamma, sigma=0.0):
        """
        A, B, C     : NumPy arrays defining the LTI system
        rho         : scalar weight for control cost
        gamma       : discount factor (0 < gamma < 1)
        sigma       : process‐noise std. dev.
        """
        self.A = A
        self.B = B
        self.Q = C.T @ C
        self.R = rho * np.eye(B.shape[1])
        self.gamma = gamma
        self.sigma = sigma

        # dimensions
        self.N_x = B.shape[0]
        self.N_u = B.shape[1]

    def step(self, x, u):
        """
        Single-step state update for the linear system x_{t+1} = A x_t + B u_t.

        Args:
            x: state vector, shape (N_x,) or (N_x, 1)
            u: control input, scalar or shape (N_u,) or (N_u, 1)

        Returns:
            x_next: next state vector, shape (N_x,)
        """
        x_arr = np.asarray(x).reshape(-1)
        u_arr = np.asarray(u).reshape(-1)
        # If u is scalar for single-input systems, broadcast to (N_u,)
        if u_arr.size == 0:
            u_arr = np.zeros(self.N_u)
        if u_arr.size == 1 and self.N_u > 1:
            u_arr = np.full(self.N_u, float(u_arr.item()))
        x_next = self.A @ x_arr + self.B @ u_arr
        return x_next

    def generate_samples(self, x_bounds, u_bounds, n_samples):
        """
        Generate random state-action samples.
        Returns:
          X : shape (n_samples, N_x)  states
          U : shape (n_samples, N_u)  actions
        """
        X, U = self.generate_samples_auxiliary(x_bounds, u_bounds, n_samples)

        X_plus = (X @ self.A.T) + (U @ self.B.T)    # shape (n_samples, N_x, 1)
        U_plus = np.random.uniform(u_bounds[0], u_bounds[1], size=(n_samples, self.N_u))

        return X, U, X_plus, U_plus
    
    def generate_samples_auxiliary(self, x_bounds, u_bounds, n_samples):
        """
        Generate auxiliary samples for cost computation.
        Returns:
          X_aux : shape (n_samples, N_x)  auxiliary states
          U_aux : shape (n_samples, N_u)  auxiliary actions
        """
        X_aux = np.random.uniform(x_bounds[0], x_bounds[1], size=(n_samples, self.N_x))
        U_aux = np.random.uniform(u_bounds[0], u_bounds[1], size=(n_samples, self.N_u))

        return X_aux, U_aux

    def simulate(self, K, X, U):
        """
        Simulate one step (with optional noise) for a batch of states & controls.
        
        Args:
          K : number of steps (for noise shape)
          X : array, shape (batch, N_x)
          U : array, shape (batch, N_u)
        
        Returns:
          X_seq : shape (batch, K, N_x)  next‐state trajectories
          cost  : shape (batch, 1)       L_x + L_u per sample
          W      : shape (batch, K, N_x)  the noise realizations
        """
        batch = X.shape[0]

        # deterministic next state
        X_plus = np.matmul(self.A, X) + np.matmul(self.B, U)  
        # sample noise
        if self.sigma == 0.0:
            W = np.zeros((batch, K, self.N_x, 1))
        else:
            W = np.random.normal(
                loc=0.0,
                scale=self.sigma,
                size=(batch, K, self.N_x, 1)
            )

        # build full trajectory over K steps
        X_seq = X_plus[:, np.newaxis, :, :] + W

        return X_seq, W
    
    def cost(self, X, U):
        """
        Compute the quadratic cost for a batch of states and controls.

        Args:
          X : array, shape (batch, N_x)
          U : array, shape (batch, N_u)

        Returns:
          cost : array, shape (batch,)  the cost per sample
        """
        # State cost: x^T Q x
        L_x = np.sum(X @ self.Q * X, axis=1)
        # Control cost: u^T R u  
        L_u = np.sum(U @ self.R * U, axis=1)
        return L_x + L_u

    def optimal_solution(self):
        """
        Solve the infinite‐horizon LQR via the discrete ARE.
        For gamma = 1.0: standard undiscounted LQR
        For gamma < 1.0: discounted LQR
        """
        if self.gamma == 1.0:
            # Standard undiscounted LQR
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
            q = 0.0  # No noise cost for undiscounted case
        else:
            # Discounted LQR
            Ad = np.sqrt(self.gamma) * self.A
            Rd = self.R / self.gamma
            P = solve_discrete_are(Ad, self.B, self.Q, Rd)
            K = self.gamma * (np.linalg.inv(self.R + self.gamma * (self.B.T @ P @ self.B)) @ self.B.T @ P @ self.A)
            q = (self.sigma**2) * (self.gamma / (1 - self.gamma)) * np.trace(P)

        return P, K, q

    def optimal_q(self, P, e, c):
        """
        Given a cost‐to‐go matrix P, scalar e, and diagonal weights c (1D array),
        build the one‐step Q‐function quadratic form and compute:
          • Qstar  (the block matrix)
          • E_Qstar = sum(diag(c) * Qstar) + e
          • gap     = γ/(1–γ) ⋅ σ² ⋅ trace(Qxu ⋅ Quu⁻¹ ⋅ Qxuᵀ)
        """
        Qxx = self.Q + self.gamma * (self.A.T @ P @ self.A)
        Quu = self.R + self.gamma * (self.B.T @ P @ self.B)
        Qxu = self.gamma * (self.A.T @ P @ self.B)

        # block‐matrix
        Qstar = np.block([
            [Qxx,    Qxu],
            [Qxu.T,  Quu]
        ])

        # expected one‐step cost under diag(c)
        c_mat = c.reshape((Qstar.shape[0], Qstar.shape[0]))  # shape (d, d)
        E_Qstar = np.sum(c_mat * Qstar) + e

        # noise‐induced “gap”
        M = Qxu @ inv(Quu) @ Qxu.T
        gap = (self.gamma / (1 - self.gamma)) * (self.sigma**2) * np.trace(M)

        return Qstar, E_Qstar, gap


class cart_pole:
    def __init__(self, m_c, m_p, l, delta_t, C, rho, gamma, N, M):
        self.N_u = 1
        self.N_x = 4
        
        # sample sizes
        self.N = N
        # number of auxiliary samples
        self.M = M
        
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        
        self.g = 9.8
        self.delta_t = delta_t
        
        self.gamma = gamma

        self.lqr_k_cart: float = 2.0
        self.lqr_k_theta: float = 30.0
        self.exploration_std = 5.0  # for policy exploration

        self.C = C
        self.rho = rho

        self.Q = C.T @ C
        self.R = rho * np.eye(self.N_u)

    def linearized_system(self, use_backward_euler=False):
        """
        Linearized cart-pole dynamics around upright equilibrium
        
        Args:
            use_backward_euler: If True, use backward Euler discretization
                               If False, use forward Euler discretization
        """
        A = np.array([[0, 1, 0, 0], [0, 0, 3.0*self.m_p * self.g / (4.0*self.m_c + self.m_p), 0],\
                          [0, 0, 0, 1], [0, 0, (3.0*(self.m_p + self.m_c)*self.g)/((4.0*self.m_c + self.m_p)*self.l), 0]])
        B = np.array([[0], [4.0/(4.0*self.m_c + self.m_p)], [0], [3.0/(self.l*(4.0*self.m_c + self.m_p))]])
        # A = np.array([[0, 1, 0, 0], [0, 0, -self.m_p * self.g / self.m_c, 0],\
        #                   [0, 0, 0, 1], [0, 0, (self.m_p + self.m_c)*self.g/(self.m_c*self.l), 0]])
        # B = np.array([[0], [1.0/self.m_c], [0], [1.0/(self.m_c*self.l)]])
        
        if use_backward_euler:
            # Backward Euler: A_d = (I - dt*A)^(-1), B_d = dt*(I - dt*A)^(-1)*B
            I = np.eye(self.N_x)
            A_d = inv(I - self.delta_t * A)
            B_d = self.delta_t * A_d @ B
        else:
            # Forward Euler: A_d = I + dt*A, B_d = dt*B
            A_d = np.eye(self.N_x) + self.delta_t * A
            B_d = self.delta_t * B
        
        return A_d, B_d

    def _f1(self, theta, theta_dot, u):
        """Cart acceleration dynamics"""
        temp = (u - self.m_p * self.l * theta_dot**2 * np.sin(theta)) / (self.m_c + self.m_p)
        x_ddot = temp + self.m_p * self.l * self._f2(theta, theta_dot, u) * np.cos(theta) / (self.m_c + self.m_p)
        # return (u + self.m_p* self.l * (np.sin(theta) *(theta_dot**2) - self._f2(theta, theta_dot, u)*np.cos(theta))) / (self.m_c + self.m_p)
        # Solve the coupled system of equations simultaneously
        # x_ddot, theta_ddot = self._solve_dynamics(theta, theta_dot, u)
        return x_ddot

    def _f2(self, theta, theta_dot, u):
        """Pole angular acceleration dynamics"""
        temp = (u - self.m_p * self.l * theta_dot**2 * np.sin(theta)) / (self.m_c + self.m_p)
        D = self.l * (4.0/3.0 - (self.m_p * np.cos(theta)**2) / (self.m_c + self.m_p))

        theta_ddot = (self.g * np.sin(theta) + np.cos(theta) * temp) / D

        # return ((-u * np.cos(theta) - self.m_p*self.l*theta_dot**2*np.sin(theta)*np.cos(theta))/(self.m_p + self.m_c) + self.g*np.sin(theta))/D
        return theta_ddot
        # Solve the coupled system of equations simultaneously
        # x_ddot, theta_ddot = self._solve_dynamics(theta, theta_dot, u)
        # return theta_ddot

    def step(self, x, u, wrap_angles=False):
        """
        Forward Euler discretization
        x = [x_pos, x_vel, theta, theta_dot]
        u = force applied to cart
        
        Args:
            x: state vector [x_pos, x_vel, theta, theta_dot]
            u: force applied to cart
            wrap_angles: (deprecated, kept for compatibility) no angle wrapping for cartpole
        """
        x_pos, x_vel, theta, theta_dot = x
        u = float(u)
        
        # Compute derivatives
        x_pos_dot = x_vel
        x_vel_dot = self._f1(theta, theta_dot, u)
        theta_dot_dot = self._f2(theta, theta_dot, u)
        
        # Euler integration
        x_pos_new = x_pos + self.delta_t * x_pos_dot
        x_vel_new = x_vel + self.delta_t * x_vel_dot
        theta_new = theta + self.delta_t * theta_dot
        # No angle wrapping for cartpole
        theta_dot_new = theta_dot + self.delta_t * theta_dot_dot
        
        return np.array([x_pos_new, x_vel_new, theta_new, theta_dot_new])

    def generate_samples_auxiliary(self, x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples):
        """Generate random state-action samples"""
        if n_samples is None:
            n_samples = self.M
            
        states = np.column_stack([
            np.random.uniform(*x_bounds, size=n_samples),
            np.random.uniform(*x_dot_bounds, size=n_samples),
            np.random.uniform(*theta_bounds, size=n_samples),
            np.random.uniform(*theta_dot_bounds, size=n_samples)
        ])
        
        actions = np.random.uniform(*u_bounds, size=(n_samples, 1))
        
        return states, actions
    
    def generate_samples(self, x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=None, use_backward_euler=False, use_lqr_rollouts=False, lqr_rollout_length=500):
        """
        Generate state transition samples.
        
        Args:
            x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds: state bounds
            u_bounds: control bounds
            n_samples: number of samples to generate
            use_backward_euler: whether to use backward Euler discretization
            use_lqr_rollouts: if True, generate half samples from LQR rollouts near equilibrium
            lqr_rollout_length: length of LQR rollouts (default: 500)
        
        Returns:
            X, U, X_plus, U_plus: state-action samples and next states/actions
        """
        if n_samples is None:
            n_samples = self.N
        
        if use_lqr_rollouts:
            # Mixed sampling: half random, half from LQR rollouts
            n_random = n_samples // 2
            n_lqr = n_samples - n_random  # Handle odd numbers
            
            # Generate half samples randomly (current method)
            X_random, U_random = self.generate_samples_auxiliary(
                x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=n_random
            )
            
            # Compute next states for random samples
            if use_backward_euler:
                X_plus_random = np.array([self.step_backward_euler(x, u[0]) for x, u in zip(X_random, U_random)])
            else:
                X_plus_random = np.array([self.step(x, u[0], wrap_angles=False) for x, u in zip(X_random, U_random)])
            
            # Generate random next actions for random samples
            U_plus_random = np.random.uniform(*u_bounds, size=(n_random, 1))
            
            # Generate half samples from LQR rollouts starting near equilibrium
            X_lqr, U_lqr, X_plus_lqr, U_plus_lqr = self._generate_lqr_rollout_samples(
                x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds,
                n_samples=n_lqr, rollout_length=lqr_rollout_length, use_backward_euler=use_backward_euler
            )
            
            # Combine both sets
            X = np.vstack([X_random, X_lqr])
            U = np.vstack([U_random, U_lqr])
            X_plus = np.vstack([X_plus_random, X_plus_lqr])
            U_plus = np.vstack([U_plus_random, U_plus_lqr])
        else:
            # Original method: all random samples
            X, U = self.generate_samples_auxiliary(x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=n_samples)
            
            # Compute next states using either forward or backward Euler
            if use_backward_euler:
                X_plus = np.array([self.step_backward_euler(x, u[0]) for x, u in zip(X, U)])
            else:
                X_plus = np.array([self.step(x, u[0], wrap_angles=False) for x, u in zip(X, U)])
            
            # Generate random next actions
            U_plus = np.random.uniform(*u_bounds, size=(n_samples, 1))

        return X, U, X_plus, U_plus
    
    def _generate_lqr_rollout_samples(self, x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds,
                                      n_samples, rollout_length=500, use_backward_euler=False):
        """
        Generate samples from LQR rollouts starting near equilibrium.
        
        Args:
            x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds: state bounds
            u_bounds: control bounds
            n_samples: number of samples to generate
            rollout_length: length of each rollout
            use_backward_euler: whether to use backward Euler for discretization
        
        Returns:
            X, U, X_plus, U_plus: state-action samples from rollouts
        """
        # Get linearized system and compute LQR policy
        A_d, B_d = self.linearized_system(use_backward_euler=use_backward_euler)
        
        # Compute LQR gain matrix
        if self.gamma == 1.0:
            # Standard undiscounted LQR
            P = solve_discrete_are(A_d, B_d, self.Q, self.R)
            K_lqr = np.linalg.inv(self.R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        else:
            # Discounted LQR
            Ad_scaled = np.sqrt(self.gamma) * A_d
            Rd_scaled = self.R / self.gamma
            P = solve_discrete_are(Ad_scaled, B_d, self.Q, Rd_scaled)
            K_lqr = self.gamma * (np.linalg.inv(self.R + self.gamma * (B_d.T @ P @ B_d)) @ B_d.T @ P @ A_d)
        
        # LQR policy: u = -K @ x
        def lqr_policy(x):
            u = -K_lqr @ x
            # Clip to bounds
            u = np.clip(u, u_bounds[0], u_bounds[1])
            return u
        
        # Generate samples from rollouts
        X_list = []
        U_list = []
        X_plus_list = []
        U_plus_list = []
        
        # Equilibrium state (upright position)
        x_eq = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Perturbation scale: small perturbations near equilibrium
        # Use 10% of the range for each dimension
        perturbation_scale = 0.5
        x_perturb = np.array([
            (x_bounds[1] - x_bounds[0]) * perturbation_scale,
            (x_dot_bounds[1] - x_dot_bounds[0]) * perturbation_scale,
            (theta_bounds[1] - theta_bounds[0]) * perturbation_scale,
            (theta_dot_bounds[1] - theta_dot_bounds[0]) * perturbation_scale
        ])
        
        # Run rollouts until we have enough samples
        # Each rollout of length L gives us approximately L samples
        # So we need roughly n_samples / rollout_length rollouts
        n_rollouts = max(1, int(np.ceil(n_samples / rollout_length)))
        
        for _ in range(n_rollouts):
            # Start from small perturbation near equilibrium
            x_init = x_eq + np.random.uniform(-x_perturb, x_perturb)
            
            # Run rollout
            x_current = x_init.copy()
            rollout_states = [x_current]
            rollout_actions = []
            
            for step in range(rollout_length):
                # Get LQR action
                u_current = lqr_policy(x_current)
                rollout_actions.append(u_current)
                
                # Step forward (no angle wrapping for cartpole)
                x_next = self.step(x_current, u_current, wrap_angles=False)
                rollout_states.append(x_next)
                x_current = x_next
                
                # Early termination if state goes out of reasonable bounds
                if np.any(np.abs(x_current) > 20):
                    break
            
            # Collect state-action pairs from rollout
            # Use all pairs: (x_t, u_t, x_{t+1}, u_{t+1})
            for t in range(len(rollout_states) - 1):
                X_list.append(rollout_states[t])
                U_list.append(rollout_actions[t])
                X_plus_list.append(rollout_states[t + 1])
                # Next action is the action at next state (or random if at end)
                if t + 1 < len(rollout_actions):
                    U_plus_list.append(rollout_actions[t + 1])
                else:
                    U_plus_list.append(np.random.uniform(*u_bounds, size=1))
                
                # Stop if we have enough samples
                if len(X_list) >= n_samples:
                    break
            
            # Stop if we have enough samples
            if len(X_list) >= n_samples:
                break
        
        # Convert to arrays
        X = np.array(X_list)
        U = np.array(U_list).reshape(-1, 1)
        X_plus = np.array(X_plus_list)
        U_plus = np.array(U_plus_list).reshape(-1, 1)
        
        # If we collected more samples than needed, randomly sample
        if len(X) > n_samples:
            indices = np.random.choice(len(X), size=n_samples, replace=False)
            X = X[indices]
            U = U[indices]
            X_plus = X_plus[indices]
            U_plus = U_plus[indices]
        # If we collected fewer samples, pad with random samples
        elif len(X) < n_samples:
            n_needed = n_samples - len(X)
            X_pad, U_pad = self.generate_samples_auxiliary(
                x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=n_needed
            )
            X_plus_pad = np.array([self.step(x, u[0], wrap_angles=False) for x, u in zip(X_pad, U_pad)])
            U_plus_pad = np.random.uniform(*u_bounds, size=(n_needed, 1))
            
            X = np.vstack([X, X_pad])
            U = np.vstack([U, U_pad])
            X_plus = np.vstack([X_plus, X_plus_pad])
            U_plus = np.vstack([U_plus, U_plus_pad])
        
        return X, U, X_plus, U_plus

    def cost(self, X, U):
        """Compute quadratic cost using Q and R matrices: L = x^T Q x + u^T R u"""
        # State cost: x^T Q x
        L_x = np.sum(X @ self.Q * X, axis=1)
        
        # Control cost: u^T R u
        if U.ndim == 1:
            L_u = U @ self.R * U
        else:
            L_u = np.sum(U @ self.R * U, axis=1)
        
        return L_x + L_u


class cart_pole_v2:
    """
    Alternative cart-pole dynamics implementation with different linearization and dynamics equations.
    """
    def __init__(self, m_c, m_p, l, delta_t, C, rho, gamma, N, M):
        self.N_u = 1
        self.N_x = 4
        
        # sample sizes
        self.N = N
        # number of auxiliary samples
        self.M = M
        
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        
        self.g = 9.8
        self.delta_t = delta_t
        
        self.gamma = gamma

        self.lqr_k_cart: float = 2.0
        self.lqr_k_theta: float = 30.0
        self.exploration_std = 5.0  # for policy exploration

        self.C = C
        self.rho = rho

        self.Q = C.T @ C
        self.R = rho * np.eye(self.N_u)

    def linearized_system(self, use_backward_euler=False):
        """
        Linearized cart-pole dynamics around upright equilibrium (alternative formulation)
        
        Args:
            use_backward_euler: If True, use backward Euler discretization
                               If False, use forward Euler discretization
        """
        # State: [x_pos, x_vel, theta, theta_dot] = [X0, X1, X2, X3]
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.g, self.l],
            [0, 0, 0, 1],
            [0, 0, ((self.m_p + self.m_c) * self.g) / ((self.m_p + self.m_c) * self.l - self.m_p * self.l), 0]
        ], dtype=float)
        
        B = np.array([
            [0],
            [0],
            [0],
            [1 / ((self.m_p + self.m_c) * self.l - self.m_p * self.l)]
        ], dtype=float)
        
        if use_backward_euler:
            # Backward Euler: A_d = (I - dt*A)^(-1), B_d = dt*(I - dt*A)^(-1)*B
            I = np.eye(self.N_x)
            A_d = inv(I - self.delta_t * A)
            B_d = self.delta_t * A_d @ B
        else:
            # Forward Euler: A_d = I + dt*A, B_d = dt*B
            A_d = np.eye(self.N_x) + self.delta_t * A
            B_d = self.delta_t * B
        
        return A_d, B_d

    def _f1(self, X2, X3, U):
        """
        Cart acceleration dynamics (X1_dot = x_vel_dot)
        
        Args:
            X2: theta (angle)
            X3: theta_dot (angular velocity)
            U: control input (force)
        """
        # Convert to scalar if needed
        if isinstance(X2, np.ndarray):
            X2 = float(X2.item()) if X2.size == 1 else X2
        if isinstance(X3, np.ndarray):
            X3 = float(X3.item()) if X3.size == 1 else X3
        if isinstance(U, np.ndarray):
            U = float(U.item()) if U.size == 1 else U
        
        X2 = float(X2)
        X3 = float(X3)
        U = float(U)
        
        # Match PyTorch version: (l*f2 - g*sin(X2)) / cos(X2)
        f2_val = self._f2(X2, X3, U)
        cos_X2 = np.cos(X2)
        
        # Avoid division by zero when cos(X2) is near zero
        if abs(cos_X2) < 1e-10:
            cos_X2 = np.sign(cos_X2) * max(abs(cos_X2), 1e-10)
        
        return (self.l * f2_val - self.g * np.sin(X2)) / cos_X2

    def _f2(self, X2, X3, U):
        """
        Pole angular acceleration dynamics (X3_dot = theta_dot_dot)
        
        Args:
            X2: theta (angle)
            X3: theta_dot (angular velocity)
            U: control input (force)
        """
        # Convert to scalar if needed
        if isinstance(X2, np.ndarray):
            X2 = float(X2.item()) if X2.size == 1 else X2
        if isinstance(X3, np.ndarray):
            X3 = float(X3.item()) if X3.size == 1 else X3
        if isinstance(U, np.ndarray):
            U = float(U.item()) if U.size == 1 else U
        
        X2 = float(X2)
        X3 = float(X3)
        U = float(U)
        
        # Match PyTorch version exactly:
        # C = cos(X2) / ((m_p + m_c)*l - m_p*l*cos(X2)^2)
        cos_X2 = np.cos(X2)
        denominator = (self.m_p + self.m_c) * self.l - self.m_p * self.l * (cos_X2 ** 2)
        
        # Avoid division by zero
        if abs(denominator) < 1e-10:
            denominator = np.sign(denominator) * max(abs(denominator), 1e-10)
        
        C = cos_X2 / denominator
        
        # Avoid division by zero when cos(X2) is near zero
        cos_X2_safe = cos_X2
        if abs(cos_X2_safe) < 1e-10:
            cos_X2_safe = np.sign(cos_X2_safe) * max(abs(cos_X2_safe), 1e-10)
        
        # Match PyTorch: C * (U + (m_p + m_c)*g*sin(X2)/cos(X2) - m_p*l*X3^2*sin(X2))
        return C * (U + (self.m_p + self.m_c) * self.g * np.sin(X2) / cos_X2_safe - 
                    self.m_p * self.l * (X3 ** 2) * np.sin(X2))

    def step(self, x, u, wrap_angles=False):
        """
        Forward Euler discretization
        x = [x_pos, x_vel, theta, theta_dot] = [X0, X1, X2, X3]
        u = force applied to cart
        
        Args:
            x: state vector, shape (4,) or (4, 1)
            u: control input, scalar or shape (1,)
            wrap_angles: (deprecated, kept for compatibility) no angle wrapping for cartpole
            
        Returns:
            x_next: next state vector, shape (4,)
        """
        # Ensure x is a 1D array
        x = np.asarray(x).reshape(-1)
        u = float(u)
        
        # Split state into components
        X0 = x[0]  # x_pos
        X1 = x[1]  # x_vel
        X2 = x[2]  # theta
        X3 = x[3]  # theta_dot
        
        # Forward Euler integration
        X0_new = X0 + self.delta_t * X1
        X1_new = X1 + self.delta_t * self._f1(X2, X3, u)
        X2_new = X2 + self.delta_t * X3
        # No angle wrapping for cartpole
        X3_new = X3 + self.delta_t * self._f2(X2, X3, u)
        
        return np.array([X0_new, X1_new, X2_new, X3_new])

    def generate_samples_auxiliary(self, x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples):
        """Generate random state-action samples"""
        if n_samples is None:
            n_samples = self.M
            
        states = np.column_stack([
            np.random.uniform(*x_bounds, size=n_samples),
            np.random.uniform(*x_dot_bounds, size=n_samples),
            np.random.uniform(*theta_bounds, size=n_samples),
            np.random.uniform(*theta_dot_bounds, size=n_samples)
        ])
        
        actions = np.random.uniform(*u_bounds, size=(n_samples, 1))
        
        return states, actions
    
    def generate_samples(self, x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=None, use_backward_euler=False):
        """Generate state transition samples"""
        X, U = self.generate_samples_auxiliary(x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=n_samples)
        
        # Compute next states using either forward or backward Euler
        if use_backward_euler:
            # For backward Euler, we'd need to implement step_backward_euler
            # For now, use forward Euler
            X_plus = np.array([self.step(x, u[0], wrap_angles=False) for x, u in zip(X, U)])
        else:
            X_plus = np.array([self.step(x, u[0], wrap_angles=False) for x, u in zip(X, U)])
        
        # Generate random next actions
        U_plus = np.random.uniform(*u_bounds, size=(n_samples, 1))

        return X, U, X_plus, U_plus

    def cost(self, X, U):
        """Compute quadratic cost using Q and R matrices: L = x^T Q x + u^T R u"""
        # State cost: x^T Q x
        L_x = np.sum(X @ self.Q * X, axis=1)
        
        # Control cost: u^T R u
        if U.ndim == 1:
            L_u = U @ self.R * U
        else:
            L_u = np.sum(U @ self.R * U, axis=1)
        
        return L_x + L_u


# All non-cartpole classes removed - keeping only dlqr, cart_pole, and cart_pole_v2



class point_mass_2d_cubic_drag:
    """
    2D point-mass with linear spring to the origin and isotropic cubic damping
    (quadratic-in-speed drag), discretized with forward Euler.

    Continuous-time model:
        p_dot = v
        v_dot = -(k/m) p - (c/m) ||v||^2 v + (1/m) u

    State:
        x = [p_x, p_y, v_x, v_y] in R^4

    Input:
        u = [u_x, u_y] in R^2
    """

    def __init__(self, m=2.0, k=5.0, c=1.2, delta_t=0.02, C=None, rho=1.0, gamma=0.99, N=0, M=0):
        self.N_u = 2
        self.N_x = 4

        # sample sizes (kept for compatibility with existing pipeline)
        self.N = N
        self.M = M

        # parameters
        self.m = float(m)
        self.k = float(k)
        self.c = float(c)
        self.delta_t = float(delta_t)

        # cost / discount params (used by PolicyExtractor and other utilities)
        if C is None:
            self.C = np.eye(self.N_x)
        else:
            self.C = np.asarray(C)
        self.rho = float(rho)
        self.gamma = float(gamma)
        
        # Cost matrices for LQR and cost computation
        self.Q = self.C.T @ self.C
        self.R = self.rho * np.eye(self.N_u)

    def step(self, x, u, wrap_angles=False):
        """One forward-Euler step."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != 4:
            raise ValueError(f"Expected state dimension 4, got {x.size}")
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size == 1:
            # allow scalar broadcast (for quick tests); apply to both axes
            u = np.array([float(u.item()), float(u.item())], dtype=float)
        if u.size != 2:
            raise ValueError(f"Expected input dimension 2, got {u.size}")

        p = x[:2]
        v = x[2:]

        # dynamics
        p_dot = v
        v_norm_sq = float(v @ v)
        v_dot = -(self.k / self.m) * p - (self.c / self.m) * v_norm_sq * v + (1.0 / self.m) * u

        x_next = np.empty_like(x)
        x_next[:2] = p + self.delta_t * p_dot
        x_next[2:] = v + self.delta_t * v_dot
        return x_next

    def linearized_system(self, use_backward_euler=False):
        """
        Discrete-time linearization around the origin.

        At v=0, the cubic damping has zero Jacobian, so the continuous-time
        linearization is:
            p_dot = v
            v_dot = -(k/m) p + (1/m) u

        Returns:
            A_d, B_d : discrete-time matrices
        """
        I2 = np.eye(2)
        Z2 = np.zeros((2, 2))
        A_c = np.block([[Z2, I2],
                        [-(self.k / self.m) * I2, Z2]])
        B_c = np.vstack([Z2, (1.0 / self.m) * I2])

        dt = self.delta_t
        if use_backward_euler:
            # x_{k+1} = x_k + dt*(A_c x_{k+1} + B_c u_k)
            # => (I - dt A_c) x_{k+1} = x_k + dt B_c u_k
            I = np.eye(self.N_x)
            A_d = np.linalg.inv(I - dt * A_c)
            B_d = A_d @ (dt * B_c)
        else:
            A_d = np.eye(self.N_x) + dt * A_c
            B_d = dt * B_c
        return A_d, B_d

    def generate_samples(self, p_bounds, v_bounds, u_bounds, n_samples):
        """
        Generate state transition samples (x, u, x_plus, u_plus).

        Args:
            p_bounds: tuple (low, high) for each position component
            v_bounds: tuple (low, high) for each velocity component
            u_bounds: tuple (low, high) for each control component
            n_samples: int

        Returns:
            X: shape (n_samples, 4) - current states
            U: shape (n_samples, 2) - current controls
            X_plus: shape (n_samples, 4) - next states
            U_plus: shape (n_samples, 2) - next controls (random)
        """
        p_low, p_high = p_bounds
        v_low, v_high = v_bounds
        u_low, u_high = u_bounds

        X = np.zeros((n_samples, self.N_x))
        U = np.zeros((n_samples, self.N_u))

        X[:, 0:2] = np.random.uniform(p_low, p_high, size=(n_samples, 2))
        X[:, 2:4] = np.random.uniform(v_low, v_high, size=(n_samples, 2))
        U[:, 0:2] = np.random.uniform(u_low, u_high, size=(n_samples, 2))
        
        # Compute next states using step function
        X_plus = np.array([self.step(x, u, wrap_angles=False) for x, u in zip(X, U)])
        
        # Generate random next actions
        U_plus = np.random.uniform(u_low, u_high, size=(n_samples, self.N_u))
        
        return X, U, X_plus, U_plus

    def generate_samples_auxiliary(self, p_bounds, v_bounds, u_bounds, n_samples):
        """
        Generate auxiliary samples (x, u) without next states.
        Used for cost computation in the LP formulation.
        
        Args:
            p_bounds: tuple (low, high) for each position component
            v_bounds: tuple (low, high) for each velocity component
            u_bounds: tuple (low, high) for each control component
            n_samples: int

        Returns:
            X: shape (n_samples, 4) - states
            U: shape (n_samples, 2) - controls
        """
        p_low, p_high = p_bounds
        v_low, v_high = v_bounds
        u_low, u_high = u_bounds

        X = np.zeros((n_samples, self.N_x))
        U = np.zeros((n_samples, self.N_u))

        X[:, 0:2] = np.random.uniform(p_low, p_high, size=(n_samples, 2))
        X[:, 2:4] = np.random.uniform(v_low, v_high, size=(n_samples, 2))
        U[:, 0:2] = np.random.uniform(u_low, u_high, size=(n_samples, 2))
        
        return X, U
    
    def cost(self, X, U):
        """
        Compute quadratic cost using Q and R matrices: L = x^T Q x + u^T R u
        
        Args:
            X: array, shape (batch, N_x) - states
            U: array, shape (batch, N_u) - controls
            
        Returns:
            cost: array, shape (batch,) - cost per sample
        """
        # State cost: x^T Q x
        L_x = np.sum(X @ self.Q * X, axis=1)
        
        # Control cost: u^T R u
        if U.ndim == 1:
            L_u = U @ self.R * U
        else:
            L_u = np.sum(U @ self.R * U, axis=1)
        
        return L_x + L_u

import numpy as np


class point_mass_cubic_drag:
    """
    nD point-mass with linear spring to the origin and isotropic cubic damping
    (quadratic-in-speed drag), discretized with forward Euler.

    Continuous-time model:
        p_dot = v
        v_dot = -(k/m) p - (c/m) ||v||^2 v + (1/m) B u

    State:
        x = [p, v] in R^(2n), with p,v in R^n

    Input:
        u in R^m (default m=n). If m=n and B = I, the system is fully actuated.
    """

    def __init__(
        self,
        n=2,                  # position/velocity dimension
        m_u=None,             # input dimension (default: n)
        B=None,               # input map (n x m_u). If None -> identity when m_u==n
        mass=2.0,
        k=5.0,
        c=1.2,
        delta_t=0.02,
        C=None,
        rho=1.0,
        gamma=0.99,
        N=0,
        M=0,
    ):
        # dimensions
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("n must be a positive integer.")
        self.N_x = 2 * self.n

        self.N_u = int(self.n if m_u is None else m_u)
        if self.N_u <= 0:
            raise ValueError("m_u must be a positive integer.")

        # input map
        if B is None:
            if self.N_u != self.n:
                raise ValueError("If B is None, you must set m_u=n (fully actuated) or provide B.")
            self.B = np.eye(self.n)
        else:
            self.B = np.asarray(B, dtype=float)
            if self.B.shape != (self.n, self.N_u):
                raise ValueError(f"B must have shape ({self.n}, {self.N_u}), got {self.B.shape}")

        # sample sizes (compatibility with existing pipeline)
        self.N = N
        self.M = M

        # parameters
        self.m = float(mass)
        self.k = float(k)
        self.c = float(c)
        self.delta_t = float(delta_t)

        # cost / discount params
        if C is None:
            self.C = np.eye(self.N_x)
        else:
            self.C = np.asarray(C, dtype=float)
            if self.C.shape[1] != self.N_x:
                raise ValueError(f"C must have {self.N_x} columns (got {self.C.shape}).")

        self.rho = float(rho)
        self.gamma = float(gamma)

        # Cost matrices for LQR and cost computation
        self.Q = self.C.T @ self.C
        self.R = self.rho * np.eye(self.N_u)

    def _validate_x_u(self, x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.N_x:
            raise ValueError(f"Expected state dimension {self.N_x}, got {x.size}")

        u = np.asarray(u, dtype=float).reshape(-1)

        # allow scalar broadcast for quick tests
        if u.size == 1:
            u = np.full(self.N_u, float(u.item()), dtype=float)

        if u.size != self.N_u:
            raise ValueError(f"Expected input dimension {self.N_u}, got {u.size}")

        return x, u

    def step(self, x, u, wrap_angles=False):
        """One forward-Euler step."""
        x, u = self._validate_x_u(x, u)

        p = x[: self.n]
        v = x[self.n :]

        # dynamics
        p_dot = v
        v_norm_sq = float(v @ v)  # ||v||^2 (scalar)
        Bu = self.B @ u           # map to R^n

        v_dot = -(self.k / self.m) * p - (self.c / self.m) * v_norm_sq * v + (1.0 / self.m) * Bu

        x_next = np.empty_like(x)
        x_next[: self.n] = p + self.delta_t * p_dot
        x_next[self.n :] = v + self.delta_t * v_dot
        return x_next

    def linearized_system(self, use_backward_euler=False):
        """
        Discrete-time linearization around the origin.

        At v=0, the cubic damping has zero Jacobian, so the continuous-time linearization is:
            p_dot = v
            v_dot = -(k/m) p + (1/m) B u

        Returns:
            A_d, B_d : discrete-time matrices with shapes
                       A_d: (2n, 2n), B_d: (2n, m_u)
        """
        In = np.eye(self.n)
        Zn = np.zeros((self.n, self.n))

        A_c = np.block([[Zn, In],
                        [-(self.k / self.m) * In, Zn]])

        B_c = np.vstack([np.zeros((self.n, self.N_u)), (1.0 / self.m) * self.B])

        dt = self.delta_t
        if use_backward_euler:
            I = np.eye(self.N_x)
            A_d = np.linalg.inv(I - dt * A_c)
            B_d = A_d @ (dt * B_c)
        else:
            A_d = np.eye(self.N_x) + dt * A_c
            B_d = dt * B_c

        return A_d, B_d

    def generate_samples(self, p_bounds, v_bounds, u_bounds, n_samples):
        """
        Generate state transition samples (x, u, x_plus, u_plus).

        Bounds are tuples (low, high) applied componentwise.
        """
        p_low, p_high = p_bounds
        v_low, v_high = v_bounds
        u_low, u_high = u_bounds

        X = np.zeros((n_samples, self.N_x))
        U = np.zeros((n_samples, self.N_u))

        X[:, : self.n] = np.random.uniform(p_low, p_high, size=(n_samples, self.n))
        X[:, self.n :] = np.random.uniform(v_low, v_high, size=(n_samples, self.n))
        U[:, :] = np.random.uniform(u_low, u_high, size=(n_samples, self.N_u))

        X_plus = np.array([self.step(x, u, wrap_angles=False) for x, u in zip(X, U)])
        U_plus = np.random.uniform(u_low, u_high, size=(n_samples, self.N_u))

        return X, U, X_plus, U_plus

    def generate_samples_auxiliary(self, p_bounds, v_bounds, u_bounds, n_samples):
        """
        Generate auxiliary samples (x, u) without next states.
        Used for cost computation in the LP formulation.
        """
        p_low, p_high = p_bounds
        v_low, v_high = v_bounds
        u_low, u_high = u_bounds

        X = np.zeros((n_samples, self.N_x))
        U = np.zeros((n_samples, self.N_u))

        X[:, : self.n] = np.random.uniform(p_low, p_high, size=(n_samples, self.n))
        X[:, self.n :] = np.random.uniform(v_low, v_high, size=(n_samples, self.n))
        U[:, :] = np.random.uniform(u_low, u_high, size=(n_samples, self.N_u))

        return X, U

    def cost(self, X, U):
        """
        Quadratic cost:
            L = x^T Q x + u^T R u

        Args:
            X: (batch, N_x) or (N_x,)
            U: (batch, N_u) or (N_u,)
        Returns:
            (batch,) costs or scalar
        """
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if U.ndim == 1:
            U = U.reshape(1, -1)

        if X.shape[1] != self.N_x:
            raise ValueError(f"X must have {self.N_x} columns, got {X.shape}")
        if U.shape[1] != self.N_u:
            raise ValueError(f"U must have {self.N_u} columns, got {U.shape}")

        L_x = np.sum((X @ self.Q) * X, axis=1)
        L_u = np.sum((U @ self.R) * U, axis=1)
        return L_x + L_u


class point_mass_cubic_drag_2du(point_mass_cubic_drag):
    """
    nD point-mass with cubic drag and a FIXED input dimension du = 2,
    with MODAL COUPLING in the spring term.

    Continuous-time model:
        p_dot = v
        v_dot = -(1/m) K p - (c/m) ||v||^2 v + (1/m) B u

    where K is a coupled stiffness matrix built from a modal decomposition:
        K = Q Λ Q^T

    Q: orthonormal "mode shape" basis (default: DCT-II orthonormal basis)
    Λ: diagonal modal stiffness spectrum (default: λ_i = k0 * (i+1)^alpha)

    Underactuation:
        u ∈ R^2 always. If B is sparse in physical coordinates (recommended),
        coupling through K propagates control influence across coordinates.

    Notes:
    - We override step() and linearized_system() to use K (not the scalar k).
    - The parent class still stores self.k, but it is not used by this subclass'
      dynamics once K is defined.
    """

    def __init__(
        self,
        n=2,
        B=None,
        mass=2.0,
        # modal stiffness params
        k0=5.0,         # overall stiffness scale (replaces 'k' effectively)
        alpha=2.0,      # modal growth exponent; alpha=2 is "string/beam-ish"
        # drag / discretization
        c=1.2,
        delta_t=0.02,
        # cost / discount
        C=None,
        rho=1.0,
        gamma=0.99,
        N=0,
        M=0,
        # optional: custom modal basis / spectrum
        Q=None,         # (n,n) orthonormal basis; if None -> DCT-II orthonormal
        lambdas=None,   # (n,) diagonal entries for Λ; if None -> k0*(i+1)^alpha
    ):
        if n < 2:
            raise ValueError(f"n must be >= 2 for the 2-input variant (got n={n}).")
        self.n = int(n)

        # ---- Build modal basis Q (orthonormal) ----
        if Q is None:
            # Orthonormal DCT-II basis (no SciPy needed)
            # Q[k, i] = sqrt(2/n) * cos(pi/n * (i+0.5) * k) for k>0
            # Q[0, i] = 1/sqrt(n)
            i = np.arange(self.n, dtype=float)
            k = np.arange(self.n, dtype=float)[:, None]  # (n,1)
            Q = np.cos((np.pi / self.n) * (i + 0.5) * k)  # (n,n)
            Q[0, :] = 1.0
            Q *= np.sqrt(2.0 / self.n)
            Q[0, :] *= 1.0 / np.sqrt(2.0)  # makes first row 1/sqrt(n)

        Q = np.asarray(Q, dtype=float)
        if Q.shape != (self.n, self.n):
            raise ValueError(f"Q must have shape ({self.n}, {self.n}), got {Q.shape}")

        # (Optional) sanity check orthonormality; keep it light to avoid brittleness
        # You can comment this out if you prefer no checks.
        QtQ = Q.T @ Q
        if not np.allclose(QtQ, np.eye(self.n), atol=1e-6):
            raise ValueError("Q must be (approximately) orthonormal: Q.T @ Q ≈ I.")

        self.Q_modes = Q

        # ---- Build modal stiffness spectrum Λ ----
        if lambdas is None:
            idx = np.arange(1, self.n + 1, dtype=float)  # 1..n
            lambdas = float(k0) * (idx ** float(alpha))

        lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
        if lambdas.size != self.n:
            raise ValueError(f"lambdas must have length n={self.n}, got {lambdas.size}")
        if np.any(lambdas < 0):
            raise ValueError("lambdas must be nonnegative to keep K positive semidefinite.")

        self.lambdas = lambdas

        # Coupled stiffness matrix K = Q Λ Q^T
        self.K = self.Q_modes @ (self.lambdas[:, None] * self.Q_modes.T)

        # ---- Choose B (recommend: sparse physical actuation) ----
        # Default: actuate two physical coordinates (1 and n), which becomes broad in modal space.
        if B is None:
            B = np.zeros((self.n, 2), dtype=float)
            B[0, 0] = 1.0
            B[-1, 1] = 1.0

        B = np.asarray(B, dtype=float)
        if B.shape != (self.n, 2):
            raise ValueError(f"B must have shape ({self.n}, 2) for the 2-input variant, got {B.shape}")

        # Call parent to set up dimensions, costs, sampling helpers, etc.
        # We pass k=0.0 because this subclass uses self.K instead of scalar k.
        super().__init__(
            n=self.n,
            m_u=2,
            B=B,
            mass=mass,
            k=0.0,          # unused by this subclass' dynamics
            c=c,
            delta_t=delta_t,
            C=C,
            rho=rho,
            gamma=gamma,
            N=N,
            M=M,
        )

    def step(self, x, u, wrap_angles=False):
        """One forward-Euler step using coupled stiffness K."""
        x, u = self._validate_x_u(x, u)

        p = x[: self.n]
        v = x[self.n :]

        p_dot = v
        v_norm_sq = float(v @ v)
        Bu = self.B @ u

        # Coupled spring term: -(1/m) K p
        v_dot = -(1.0 / self.m) * (self.K @ p) - (self.c / self.m) * v_norm_sq * v + (1.0 / self.m) * Bu

        x_next = np.empty_like(x)
        x_next[: self.n] = p + self.delta_t * p_dot
        x_next[self.n :] = v + self.delta_t * v_dot
        return x_next

    def linearized_system(self, use_backward_euler=False):
        """
        Discrete-time linearization around the origin.

        At v=0, cubic drag Jacobian is zero, so:
            p_dot = v
            v_dot = -(1/m) K p + (1/m) B u
        """
        In = np.eye(self.n)
        Zn = np.zeros((self.n, self.n))

        A_c = np.block([
            [Zn, In],
            [-(1.0 / self.m) * self.K, Zn]
        ])

        B_c = np.vstack([
            np.zeros((self.n, self.N_u)),
            (1.0 / self.m) * self.B
        ])

        dt = self.delta_t
        if use_backward_euler:
            I = np.eye(self.N_x)
            A_d = np.linalg.inv(I - dt * A_c)
            B_d = A_d @ (dt * B_c)
        else:
            A_d = np.eye(self.N_x) + dt * A_c
            B_d = dt * B_c

        return A_d, B_d
