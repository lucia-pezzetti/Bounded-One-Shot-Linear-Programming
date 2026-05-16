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


class point_mass_cubic_drag:
    """
    nD point-mass with linear spring to the origin and isotropic cubic damping
    (quadratic-in-speed drag), discretized with forward Euler.

    Continuous-time model:
        p_dot = v
        v_dot = -(k/m) p + (1/m) G h(p) - (d/m) v
                - (c/m) ||v||^2 v + (1/m) B u

    where h(p)=p by default. Other supported choices are sin(p), tanh(p),
    and asinh(p), exposed as gravity_type="log" for a smooth signed
    logarithmic response.

    By default G=0, which recovers the original cubic-drag system exactly.

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
        linear_damping=0.0,
        delta_t=0.02,
        C=None,
        rho=1.0,
        gamma=0.99,
        N=0,
        M=0,
        gravity_diag=None,
        gravity_type="linear",
        integrator="euler",
        q4_p=0.0,
        q4_v=0.0,
        r4_u=0.0,
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
        self.linear_damping = float(linear_damping)
        if self.linear_damping < 0.0:
            raise ValueError("linear_damping must be nonnegative.")
        self.delta_t = float(delta_t)
        self.integrator = self._coerce_integrator(integrator)
        self.gravity_diag = self._coerce_gravity_diag(gravity_diag)
        self.gravity_type = self._coerce_gravity_type(gravity_type)
        self.G = np.diag(self.gravity_diag)
        self.has_gravity = bool(np.any(np.abs(self.gravity_diag) > 0.0))

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
        self.q4_p = self._coerce_cost_weights(q4_p, self.n, "q4_p")
        self.q4_v = self._coerce_cost_weights(q4_v, self.n, "q4_v")
        self.r4_u = self._coerce_cost_weights(r4_u, self.N_u, "r4_u")
        self.has_quartic_cost = bool(
            np.any(self.q4_p > 0.0) or np.any(self.q4_v > 0.0) or np.any(self.r4_u > 0.0)
        )

    @staticmethod
    def _coerce_gravity_type(gravity_type):
        if gravity_type is None:
            gravity_type = "linear"
        key = str(gravity_type).strip().lower()
        aliases = {
            "linear": "linear",
            "identity": "linear",
            "p": "linear",
            "position": "linear",
            "sin": "sin",
            "sine": "sin",
            "tanh": "tanh",
            "hyperbolic_tangent": "tanh",
            "log": "log",
            "asinh": "log",
            "arcsinh": "log",
            "signed_log": "log",
        }
        if key not in aliases:
            raise ValueError("gravity_type must be one of: linear, sin, tanh, log.")
        return aliases[key]

    @staticmethod
    def _coerce_integrator(integrator):
        if integrator is None:
            integrator = "euler"
        key = str(integrator).strip().lower()
        aliases = {
            "euler": "euler",
            "forward_euler": "euler",
            "rk4": "rk4",
            "runge_kutta": "rk4",
            "runge_kutta_4": "rk4",
        }
        if key not in aliases:
            raise ValueError("integrator must be one of: euler, rk4.")
        return aliases[key]

    @staticmethod
    def _coerce_cost_weights(weights, expected_dim, name):
        if isinstance(weights, str):
            raw_values = [v.strip() for v in weights.replace(";", ",").split(",") if v.strip()]
            values = np.asarray([float(v) for v in raw_values], dtype=float)
        else:
            values = np.asarray(weights, dtype=float).reshape(-1)
        if values.size == 1:
            values = np.full(expected_dim, float(values.item()))
        if values.size != expected_dim:
            raise ValueError(f"{name} must be scalar or length {expected_dim}, got {values.size}")
        if np.any(values < 0.0):
            raise ValueError(f"{name} must be nonnegative.")
        return values

    def _coerce_gravity_diag(self, gravity_diag):
        if gravity_diag is None:
            return np.zeros(self.n)
        values = np.asarray(gravity_diag, dtype=float).reshape(-1)
        if values.size == 1:
            values = np.full(self.n, float(values.item()))
        if values.size != self.n:
            raise ValueError(f"gravity_diag must be scalar or length {self.n}, got {values.size}")
        return values

    def stiffness_matrix(self):
        if hasattr(self, "K") and isinstance(self.K, np.ndarray) and self.K.shape == (self.n, self.n):
            return self.K
        return self.k * np.eye(self.n)

    def linearized_stiffness_minus_gravity(self):
        return self.stiffness_matrix() - self.G

    def _gravity_force(self, p):
        if not self.has_gravity:
            return np.zeros_like(p, dtype=float)
        if self.gravity_type == "linear":
            gravity_argument = p
        elif self.gravity_type == "sin":
            gravity_argument = np.sin(p)
        elif self.gravity_type == "tanh":
            gravity_argument = np.tanh(p)
        else:
            gravity_argument = np.arcsinh(p)
        return self.gravity_diag * gravity_argument

    def _spring_force(self, p):
        return self.k * p

    def _spring_force_batch(self, P):
        return self.k * P

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

    def continuous_dynamics(self, x, u):
        x, u = self._validate_x_u(x, u)
        p = x[: self.n]
        v = x[self.n :]
        p_dot = v
        v_norm_sq = float(v @ v)
        Bu = self.B @ u
        gravity_force = self._gravity_force(p)
        v_dot = (
            -(1.0 / self.m) * self._spring_force(p)
            + (1.0 / self.m) * gravity_force
            - (self.linear_damping / self.m) * v
            - (self.c / self.m) * v_norm_sq * v
            + (1.0 / self.m) * Bu
        )
        return np.concatenate([p_dot, v_dot])

    def _continuous_dynamics_batch(self, X, U):
        P = X[:, : self.n]
        V = X[:, self.n :]
        v_norm_sq = np.sum(V * V, axis=1, keepdims=True)
        Bu = U @ self.B.T
        gravity_force = self._gravity_force(P)
        V_dot = (
            -(1.0 / self.m) * self._spring_force_batch(P)
            + (1.0 / self.m) * gravity_force
            - (self.linear_damping / self.m) * V
            - (self.c / self.m) * v_norm_sq * V
            + (1.0 / self.m) * Bu
        )
        X_dot = np.empty_like(X)
        X_dot[:, : self.n] = V
        X_dot[:, self.n :] = V_dot
        return X_dot

    def step(self, x, u, wrap_angles=False):
        """One integration step."""
        x, u = self._validate_x_u(x, u)
        dt = self.delta_t
        if self.integrator == "euler":
            return x + dt * self.continuous_dynamics(x, u)

        k1 = self.continuous_dynamics(x, u)
        k2 = self.continuous_dynamics(x + 0.5 * dt * k1, u)
        k3 = self.continuous_dynamics(x + 0.5 * dt * k2, u)
        k4 = self.continuous_dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def vectorized_step(self, X, U):
        """Batched integration step for rollout evaluation."""
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
        if U.shape[0] == 1 and X.shape[0] > 1:
            U = np.repeat(U, X.shape[0], axis=0)
        if U.shape[0] != X.shape[0]:
            raise ValueError(f"X and U batch sizes must match, got {X.shape[0]} and {U.shape[0]}")

        dt = self.delta_t
        if self.integrator == "euler":
            return X + dt * self._continuous_dynamics_batch(X, U)

        k1 = self._continuous_dynamics_batch(X, U)
        k2 = self._continuous_dynamics_batch(X + 0.5 * dt * k1, U)
        k3 = self._continuous_dynamics_batch(X + 0.5 * dt * k2, U)
        k4 = self._continuous_dynamics_batch(X + dt * k3, U)
        return X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def linearized_system(self, use_backward_euler=False):
        """
        Discrete-time linearization around the origin.

        At v=0, the cubic damping has zero Jacobian. All supported h choices
        have h'(0)=1, so the continuous-time linearization is:
            p_dot = v
            v_dot = -(1/m) (K - G) p - (d/m) v + (1/m) B u

        Returns:
            A_d, B_d : discrete-time matrices with shapes
                       A_d: (2n, 2n), B_d: (2n, m_u)
        """
        In = np.eye(self.n)
        Zn = np.zeros((self.n, self.n))

        K_minus_G = self.linearized_stiffness_minus_gravity()
        A_c = np.block([[Zn, In],
                        [-(1.0 / self.m) * K_minus_G,
                         -(self.linear_damping / self.m) * In]])

        B_c = np.vstack([np.zeros((self.n, self.N_u)), (1.0 / self.m) * self.B])

        dt = self.delta_t
        if use_backward_euler:
            I = np.eye(self.N_x)
            A_d = np.linalg.inv(I - dt * A_c)
            B_d = A_d @ (dt * B_c)
        elif self.integrator == "rk4":
            A2 = A_c @ A_c
            A3 = A2 @ A_c
            A4 = A3 @ A_c
            A_d = (
                np.eye(self.N_x)
                + dt * A_c
                + 0.5 * (dt ** 2) * A2
                + (dt ** 3 / 6.0) * A3
                + (dt ** 4 / 24.0) * A4
            )
            B_d = dt * (
                np.eye(self.N_x)
                + 0.5 * dt * A_c
                + (dt ** 2 / 6.0) * A2
                + (dt ** 3 / 24.0) * A3
            ) @ B_c
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
        Stage cost:
            L = x^T Q x + u^T R u
                + sum_i q4_p_i p_i^4 + sum_i q4_v_i v_i^4
                + sum_j r4_u_j u_j^4

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
        if U.ndim == 0:
            U = U.reshape(1, 1)
        elif U.ndim == 1:
            if self.N_u == 1 and U.size == X.shape[0] and X.shape[0] > 1:
                U = U.reshape(-1, 1)
            else:
                U = U.reshape(1, -1)

        if X.shape[1] != self.N_x:
            raise ValueError(f"X must have {self.N_x} columns, got {X.shape}")
        if U.shape[1] != self.N_u:
            raise ValueError(f"U must have {self.N_u} columns, got {U.shape}")
        if U.shape[0] == 1 and X.shape[0] > 1:
            U = np.repeat(U, X.shape[0], axis=0)
        if U.shape[0] != X.shape[0]:
            raise ValueError(f"X and U batch sizes must match, got {X.shape[0]} and {U.shape[0]}")

        L_x = np.sum((X @ self.Q) * X, axis=1)
        L_u = np.sum((U @ self.R) * U, axis=1)
        if self.has_quartic_cost:
            P = X[:, : self.n]
            V = X[:, self.n :]
            L_x += np.sum(self.q4_p.reshape(1, -1) * (P ** 4), axis=1)
            L_x += np.sum(self.q4_v.reshape(1, -1) * (V ** 4), axis=1)
            L_u += np.sum(self.r4_u.reshape(1, -1) * (U ** 4), axis=1)
        return L_x + L_u


class point_mass_cubic_drag_1du(point_mass_cubic_drag):
    """
    nD point-mass with cubic drag and a FIXED input dimension du = 1,
    with MODAL COUPLING in the spring term.

    Continuous-time model:
        p_dot = v
        v_dot = -(1/m) K p + (1/m) G h(p) - (d/m) v
                - (c/m) ||v||^2 v + (1/m) B u

    where h(p)=p by default. Other supported choices are sin(p), tanh(p),
    and asinh(p), exposed as gravity_type="log" for a smooth signed
    logarithmic response.

    By default G=0, which recovers the original cubic-drag system exactly.

    where K is a coupled stiffness matrix built from a modal decomposition:
        K = Q Λ Q^T

    Q: orthonormal "mode shape" basis (default: DCT-II orthonormal basis)
    Λ: diagonal modal stiffness spectrum (default: λ_i = k0 * (i+1)^alpha)

    Underactuation:
        u ∈ R^1 always. You can choose a dense random B (benchmark default)
        or a sparse structured B (single-point actuation); coupling through K
        propagates influence across coordinates.

    Notes:
    - We override step() and linearized_system() to use K (not the scalar k).
    - The parent class still stores self.k, but it is not used by this subclass'
      dynamics once K is defined.
    """

    def __init__(
        self,
        n=2,
        B=None,
        B_mode="dense_random",
        B_scale=1.0,
        seed=None,
        rng=None,
        mass=2.0,
        # modal stiffness params
        k0=5.0,         # overall stiffness scale (replaces 'k' effectively)
        alpha=2.0,      # modal growth exponent; alpha=2 is "string/beam-ish"
        # drag / discretization
        c=1.2,
        linear_damping=0.0,
        delta_t=0.02,
        # cost / discount
        C=None,
        rho=1.0,
        gamma=0.99,
        N=0,
        M=0,
        gravity_diag=None,
        gravity_type="linear",
        integrator="euler",
        q4_p=0.0,
        q4_v=0.0,
        r4_u=0.0,
        # optional: custom modal basis / spectrum
        Q=None,         # (n,n) orthonormal basis; if None -> DCT-II orthonormal
        lambdas=None,   # (n,) diagonal entries for Λ; if None -> k0*(i+1)^alpha
    ):
        if n < 1:
            raise ValueError(f"n must be >= 1 for the 1-input variant (got n={n}).")
        self.n = int(n)

        # ---- Build modal basis Q (orthonormal) ----
        if Q is None:
            # Orthonormal DCT-II basis (no SciPy needed)
            i = np.arange(self.n, dtype=float)
            k = np.arange(self.n, dtype=float)[:, None]  # (n,1)
            Q = np.cos((np.pi / self.n) * (i + 0.5) * k)  # (n,n)
            Q[0, :] = 1.0
            Q *= np.sqrt(2.0 / self.n)
            Q[0, :] *= 1.0 / np.sqrt(2.0)  # makes first row 1/sqrt(n)

        Q = np.asarray(Q, dtype=float)
        if Q.shape != (self.n, self.n):
            raise ValueError(f"Q must have shape ({self.n}, {self.n}), got {Q.shape}")

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

        # ---- Choose B (input map, now a single column: (n, 1)) ----
        if B is None:
            if rng is None:
                rng = np.random.default_rng(seed)

            mode = str(B_mode).lower()
            if mode in ["dense_random", "dense", "random", "rand"]:
                B = rng.standard_normal((self.n, 1))
                # Normalize so the input has unit-norm actuation direction
                col_norm = np.linalg.norm(B)
                col_norm = max(col_norm, 1e-12)
                B = B / col_norm
                B = float(B_scale) * B
            elif mode in ["sparse_structured", "sparse", "structured", "endpoint"]:
                # Actuate the first physical coordinate only
                B = np.zeros((self.n, 1), dtype=float)
                B[0, 0] = 1.0
                B = float(B_scale) * B
            else:
                raise ValueError(
                    f"Unknown B_mode='{B_mode}'. Use 'dense_random' (default) or 'sparse_structured'."
                )

        B = np.asarray(B, dtype=float)
        if B.shape != (self.n, 1):
            raise ValueError(f"B must have shape ({self.n}, 1) for the 1-input variant, got {B.shape}")

        # Call parent to set up dimensions, costs, sampling helpers, etc.
        # We pass k=0.0 because this subclass uses self.K instead of scalar k.
        super().__init__(
            n=self.n,
            m_u=1,
            B=B,
            mass=mass,
            k=0.0,          # unused by this subclass' dynamics
            c=c,
            linear_damping=linear_damping,
            delta_t=delta_t,
            C=C,
            rho=rho,
            gamma=gamma,
            N=N,
            M=M,
            gravity_diag=gravity_diag,
            gravity_type=gravity_type,
            integrator=integrator,
            q4_p=q4_p,
            q4_v=q4_v,
            r4_u=r4_u,
        )

    def _spring_force(self, p):
        return self.K @ p

    def _spring_force_batch(self, P):
        return P @ self.K.T

    def linearized_system(self, use_backward_euler=False):
        """
        Discrete-time linearization around the origin.

        At v=0, cubic drag Jacobian is zero and h'(0)=1 for all supported
        gravity types, so:
            p_dot = v
            v_dot = -(1/m) (K - G) p - (d/m) v + (1/m) B u
        """
        In = np.eye(self.n)
        Zn = np.zeros((self.n, self.n))

        K_minus_G = self.linearized_stiffness_minus_gravity()
        A_c = np.block([
            [Zn, In],
            [-(1.0 / self.m) * K_minus_G,
             -(self.linear_damping / self.m) * In]
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
        elif self.integrator == "rk4":
            A2 = A_c @ A_c
            A3 = A2 @ A_c
            A4 = A3 @ A_c
            A_d = (
                np.eye(self.N_x)
                + dt * A_c
                + 0.5 * (dt ** 2) * A2
                + (dt ** 3 / 6.0) * A3
                + (dt ** 4 / 24.0) * A4
            )
            B_d = dt * (
                np.eye(self.N_x)
                + 0.5 * dt * A_c
                + (dt ** 2 / 6.0) * A2
                + (dt ** 3 / 24.0) * A3
            ) @ B_c
        else:
            A_d = np.eye(self.N_x) + dt * A_c
            B_d = dt * B_c

        return A_d, B_d


class controlled_duffing:
    """
    Controlled damped Duffing oscillator family with hardening cubic stiffness,
    discretized with the same forward-Euler convention used by the nonlinear
    point-mass systems in this module.

    For n=1, this is exactly the 2D system:
        x1_dot = x2
        x2_dot = -delta*x2 - alpha*x1 - beta*x1^3 + u

    For n>1, the state is x = [q_1,...,q_n, v_1,...,v_n] and:
        q_dot = v
        v_dot = -delta*v - K*q - beta*g(q) + B*u

    where g(q) is configurable:
        diagonal: g_i(q) = q_i^3
        radial:   g(q) = ||q||^2 q
        modal:    g(q) = Q (Q^T q)^3

    Stage cost:
        l(x, u) = q1*||q||^2 + q2*||v||^2 + q4*sum_i q_i^4 + rho*||u||^2

    If normalize_cost=True, the state terms are divided by n so the reported
    objective is an average state cost per oscillator plus actual control cost.
    """

    system_name = "duffing"

    def __init__(
        self,
        n=1,
        m_u=1,
        B=None,
        delta=0.2,
        alpha=1.0,
        beta=1.0,
        stiffness_growth=2.0,
        cubic_coupling="diagonal",
        Q_modes=None,
        delta_t=0.05,
        q1=1.0,
        q2=0.1,
        q4=0.05,
        rho=0.01,
        normalize_cost=False,
        gamma=0.99,
        N=0,
        M=0,
        state_bounds=((-2.5, 2.5), (-2.5, 2.5)),
        u_bounds=(-5.0, 5.0),
        seed=0,
        rng=None,
    ):
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("n must be a positive integer.")

        self.N_x = 2 * self.n
        self.N_u = int(m_u)
        if self.N_u <= 0:
            raise ValueError("m_u must be a positive integer.")

        self.N = N
        self.M = M

        self.delta = float(delta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stiffness_growth = float(stiffness_growth)
        self.cubic_coupling = str(cubic_coupling).strip().lower()
        if self.cubic_coupling not in ("diagonal", "radial", "modal"):
            raise ValueError("cubic_coupling must be one of: diagonal, radial, modal.")
        self.delta_t = float(delta_t)

        self.q1 = float(q1)
        self.q2 = float(q2)
        self.q4 = float(q4)
        self.rho = float(rho)
        self.normalize_cost = bool(normalize_cost)
        self.state_cost_scale = 1.0 / self.n if self.normalize_cost else 1.0
        self.gamma = float(gamma)

        idx = np.arange(1, self.n + 1, dtype=float)
        self.alphas = self.alpha * (idx ** self.stiffness_growth)
        self.K = np.diag(self.alphas)

        if Q_modes is None:
            if self.cubic_coupling == "modal":
                i = np.arange(self.n, dtype=float)
                k = np.arange(self.n, dtype=float)[:, None]
                Q_modes = np.cos((np.pi / self.n) * (i + 0.5) * k)
                Q_modes[0, :] = 1.0
                Q_modes *= np.sqrt(2.0 / self.n)
                Q_modes[0, :] *= 1.0 / np.sqrt(2.0)
            else:
                Q_modes = np.eye(self.n)
        self.Q_modes = np.asarray(Q_modes, dtype=float)
        if self.Q_modes.shape != (self.n, self.n):
            raise ValueError(f"Q_modes must have shape ({self.n}, {self.n}), got {self.Q_modes.shape}")
        if self.cubic_coupling == "modal":
            if not np.allclose(self.Q_modes.T @ self.Q_modes, np.eye(self.n), atol=1e-6):
                raise ValueError("Q_modes must be approximately orthonormal for modal cubic coupling.")

        if B is None:
            if self.N_u == self.n:
                B = np.eye(self.n)
            elif self.N_u == 1:
                if self.n == 1:
                    B = np.ones((1, 1), dtype=float)
                else:
                    if rng is None:
                        rng = np.random.default_rng(seed)
                    B = rng.standard_normal((self.n, 1))
                    B = B / max(np.linalg.norm(B), 1e-12)
            else:
                raise ValueError("If B is None, use m_u=1, m_u=n, or provide B explicitly.")
        self.B = np.asarray(B, dtype=float)
        if self.B.shape != (self.n, self.N_u):
            raise ValueError(f"B must have shape ({self.n}, {self.N_u}), got {self.B.shape}")

        self.Q = np.diag(np.concatenate([
            np.full(self.n, self.state_cost_scale * self.q1),
            np.full(self.n, self.state_cost_scale * self.q2),
        ]))
        self.R = self.rho * np.eye(self.N_u)

        sample_state_bounds = np.asarray(state_bounds, dtype=float)
        if sample_state_bounds.shape != (2, 2):
            raise ValueError(f"state_bounds must have shape (2, 2), got {sample_state_bounds.shape}")
        self.sample_state_bounds = sample_state_bounds
        self.state_bounds = np.vstack([
            np.tile(sample_state_bounds[0], (self.n, 1)),
            np.tile(sample_state_bounds[1], (self.n, 1)),
        ])

        u_bounds = np.asarray(u_bounds, dtype=float).reshape(-1)
        if u_bounds.size != 2:
            raise ValueError("u_bounds must be a pair (low, high).")
        self.u_bounds = (float(u_bounds[0]), float(u_bounds[1]))

    def _validate_x_u(self, x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.N_x:
            raise ValueError(f"Expected state dimension {self.N_x}, got {x.size}")

        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size == 1 and self.N_u == 1:
            u = np.array([float(u.item())], dtype=float)
        if u.size != self.N_u:
            raise ValueError(f"Expected input dimension {self.N_u}, got {u.size}")
        return x, u

    def continuous_dynamics(self, x, u):
        """Evaluate the continuous-time Duffing vector field."""
        x, u = self._validate_x_u(x, u)
        q = x[: self.n]
        v = x[self.n :]
        q_dot = v
        v_dot = -self.delta * v - (self.K @ q) - self.beta * self._cubic_force(q) + (self.B @ u)
        return np.concatenate([q_dot, v_dot])

    def _cubic_force(self, q):
        """Return g(q) for the selected cubic stiffness coupling."""
        if self.cubic_coupling == "diagonal":
            return q ** 3
        if self.cubic_coupling == "radial":
            return float(q @ q) * q
        z = self.Q_modes.T @ q
        return self.Q_modes @ (z ** 3)

    def _cubic_force_batch(self, Q_pos):
        """Batched version of _cubic_force for row-major states."""
        if self.cubic_coupling == "diagonal":
            return Q_pos ** 3
        if self.cubic_coupling == "radial":
            return np.sum(Q_pos * Q_pos, axis=1, keepdims=True) * Q_pos
        Z = Q_pos @ self.Q_modes
        return (Z ** 3) @ self.Q_modes.T

    def step(self, x, u, wrap_angles=False):
        """One forward-Euler step."""
        x, u = self._validate_x_u(x, u)
        return x + self.delta_t * self.continuous_dynamics(x, u)

    def vectorized_step(self, X, U):
        """Batched forward-Euler step for rollout evaluation."""
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

        Q_pos = X[:, : self.n]
        V = X[:, self.n :]
        Q_dot = V
        V_dot = -self.delta * V - (Q_pos @ self.K.T) - self.beta * self._cubic_force_batch(Q_pos) + (U @ self.B.T)

        X_dot = np.empty_like(X)
        X_dot[:, : self.n] = Q_dot
        X_dot[:, self.n :] = V_dot
        return X + self.delta_t * X_dot

    def linearized_system(self, use_backward_euler=False):
        """
        Discrete-time linearization around the origin.

        The quartic cost term is intentionally not part of the LQR baseline.
        """
        In = np.eye(self.n)
        Zn = np.zeros((self.n, self.n))
        A_c = np.block([
            [Zn, In],
            [-self.K, -self.delta * In],
        ])
        B_c = np.vstack([
            np.zeros((self.n, self.N_u)),
            self.B,
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

    def generate_samples(self, p_bounds, v_bounds, u_bounds, n_samples):
        """
        Generate state transition samples (x, u, x_plus, u_plus).

        Bounds are tuples (low, high) applied componentwise to all positions,
        velocities, and controls.
        """
        p_low, p_high = p_bounds
        v_low, v_high = v_bounds
        u_low, u_high = u_bounds

        X = np.empty((n_samples, self.N_x))
        U = np.empty((n_samples, self.N_u))
        X[:, : self.n] = np.random.uniform(p_low, p_high, size=(n_samples, self.n))
        X[:, self.n :] = np.random.uniform(v_low, v_high, size=(n_samples, self.n))
        U[:, :] = np.random.uniform(u_low, u_high, size=(n_samples, self.N_u))

        X_plus = self.vectorized_step(X, U)
        U_plus = np.random.uniform(u_low, u_high, size=(n_samples, self.N_u))
        return X, U, X_plus, U_plus

    def generate_samples_auxiliary(self, p_bounds, v_bounds, u_bounds, n_samples):
        """Generate auxiliary state-action samples."""
        p_low, p_high = p_bounds
        v_low, v_high = v_bounds
        u_low, u_high = u_bounds

        X = np.empty((n_samples, self.N_x))
        U = np.empty((n_samples, self.N_u))
        X[:, : self.n] = np.random.uniform(p_low, p_high, size=(n_samples, self.n))
        X[:, self.n :] = np.random.uniform(v_low, v_high, size=(n_samples, self.n))
        U[:, :] = np.random.uniform(u_low, u_high, size=(n_samples, self.N_u))
        return X, U

    def cost(self, X, U):
        """
        Duffing stage cost for a batch of states and controls.

        Args:
            X: (batch, N_x) or (N_x,)
            U: (batch, N_u), scalar, or (N_u,)
        Returns:
            (batch,) costs
        """
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if U.ndim == 0:
            U = U.reshape(1, 1)
        elif U.ndim == 1:
            U = U.reshape(1, -1) if self.N_u > 1 else U.reshape(-1, 1)

        if X.shape[1] != self.N_x:
            raise ValueError(f"X must have {self.N_x} columns, got {X.shape}")
        if U.shape[1] != self.N_u:
            raise ValueError(f"U must have {self.N_u} columns, got {U.shape}")
        if U.shape[0] == 1 and X.shape[0] > 1:
            U = np.repeat(U, X.shape[0], axis=0)
        if U.shape[0] != X.shape[0]:
            raise ValueError(f"X and U batch sizes must match, got {X.shape[0]} and {U.shape[0]}")

        q = X[:, : self.n]
        v = X[:, self.n :]
        L_q = self.state_cost_scale * self.q1 * np.sum(q * q, axis=1)
        L_v = self.state_cost_scale * self.q2 * np.sum(v * v, axis=1)
        L_q4 = self.state_cost_scale * self.q4 * np.sum(q ** 4, axis=1)
        L_u = np.sum((U @ self.R) * U, axis=1)
        return L_q + L_v + L_q4 + L_u

    def metadata(self):
        """Serializable benchmark metadata for result files."""
        return {
            "system": self.system_name,
            "title": "Controlled Duffing oscillator",
            "dynamics_label": "Damped hardening Duffing",
            "cost_label": "Quartic state cost" if self.q4 != 0.0 else "Quadratic state cost",
            "system_params": {
                "n_oscillators": self.n,
                "delta": self.delta,
                "alpha": self.alpha,
                "beta": self.beta,
                "stiffness_growth": self.stiffness_growth,
                "cubic_coupling": self.cubic_coupling,
                "B": self.B.tolist(),
                "Q_modes": self.Q_modes.tolist() if self.cubic_coupling == "modal" else None,
            },
            "cost_params": {
                "q1": self.q1,
                "q2": self.q2,
                "q4": self.q4,
                "r": self.rho,
                "normalize_cost": self.normalize_cost,
                "state_cost_scale": self.state_cost_scale,
            },
            "dt": self.delta_t,
            "integration": "forward_euler",
            "state_sampling_domain": self.state_bounds.tolist(),
            "action_sampling_domain": list(self.u_bounds),
        }
