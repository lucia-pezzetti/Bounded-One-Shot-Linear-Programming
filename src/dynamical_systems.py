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


class point_mass_cubic_drag_1du(point_mass_cubic_drag):
    """
    nD point-mass with cubic drag and a FIXED input dimension du = 1,
    with MODAL COUPLING in the spring term.

    Continuous-time model:
        p_dot = v
        v_dot = -(1/m) K p - (c/m) ||v||^2 v + (1/m) B u

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
