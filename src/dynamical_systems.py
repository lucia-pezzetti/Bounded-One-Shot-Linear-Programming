import numpy as np
from scipy.linalg import solve_discrete_are, inv

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
        Solve the infinite‐horizon discounted LQR via the discrete ARE:
          P = solve_discrete_are(√γ A, B, Q, R/γ)
        Then K = –γ (R + γ BᵀPB)⁻¹ Bᵀ P A.
        Also returns the baseline noise cost q = σ² γ/(1–γ) trace(P).
        """
        # scale A and R for discount
        Ad = np.sqrt(self.gamma) * self.A
        Rd = self.R / self.gamma

        # discrete algebraic Riccati
        P = solve_discrete_are(Ad, self.B, self.Q, Rd)

        # optimal gain
        inv_term = inv(self.R + self.gamma * (self.B.T @ P @ self.B))
        K = -self.gamma * (inv_term @ self.B.T @ P @ self.A)

        # baseline cost from process noise
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

    def linearized_system(self):
        """Linearized cart-pole dynamics around upright equilibrium"""
        A = np.array([[0, 1, 0, 0], 
                      [0, 0, -self.m_p*self.g/self.m_c, 0],
                      [0, 0, 0, 1], 
                      [0, 0, (self.m_p+self.m_c)*self.g/(self.m_c*self.l), 0]])
        B = np.array([[0], [1/self.m_c], [0], [1/(self.m_c*self.l)]])
        
        A_d = np.eye(self.N_x) + self.delta_t * A
        B_d = self.delta_t * B
        
        return A_d, B_d

    def _f1(self, theta, theta_dot, u):
        """Cart acceleration dynamics"""
        return (
            u + self.m_p*self.l*(theta_dot**2*np.sin(theta) - self._f2(theta, theta_dot, u)*np.cos(theta))
        ) / (self.m_p + self.m_c)

    def _f2(self, theta, theta_dot, u):
        """Pole angular acceleration dynamics"""
        den = (self.m_p + self.m_c)*self.l - self.m_p*self.l*np.cos(theta)**2  # = l*Delta
        return (
            -u*np.cos(theta)
            - self.m_p*self.l*theta_dot**2*np.sin(theta)*np.cos(theta)
            - (self.m_p + self.m_c)*self.g*np.sin(theta)
        ) / den

    def step(self, x, u):
        """
        Forward Euler discretization
        x = [x_pos, x_vel, theta, theta_dot]
        u = force applied to cart
        """
        x_pos, x_vel, theta, theta_dot = x
        
        # Compute derivatives
        x_pos_dot = x_vel
        x_vel_dot = self._f1(theta, theta_dot, u)
        theta_dot_new = theta_dot
        theta_dot_dot = self._f2(theta, theta_dot, u)
        
        # Euler integration
        x_pos_new = x_pos + self.delta_t * x_pos_dot
        x_vel_new = x_vel + self.delta_t * x_vel_dot
        theta_new = theta + self.delta_t * theta_dot_new
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
    
    def generate_samples(self, x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=None):
        """Generate state transition samples"""
        X, U = self.generate_samples_auxiliary(x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=n_samples)
        
        # Compute next states
        X_plus = np.array([self.step(x, u[0]) for x, u in zip(X, U)])
        
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