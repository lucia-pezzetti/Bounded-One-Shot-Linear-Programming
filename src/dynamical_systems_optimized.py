from scipy.linalg import solve_discrete_are, inv
import numpy as np

class dlqr:
    def __init__(self, A, B, C, rho, gamma, sigma=0.0):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)
        self.Q = C.T @ C
        self.R = rho * np.eye(self.B.shape[1])
        self.gamma = float(gamma)
        self.sigma = float(sigma)

    def generate_samples_auxiliary(self, x_bounds, u_bounds, n_samples, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        dx = self.A.shape[0]; du = self.B.shape[1]
        X = rng.uniform(x_bounds[0], x_bounds[1], size=(n_samples, dx))
        U = rng.uniform(u_bounds[0], u_bounds[1], size=(n_samples, du))
        return X, U

    def generate_samples(self, x_bounds, u_bounds, n_samples, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        X, U = self.generate_samples_auxiliary(x_bounds, u_bounds, n_samples, rng=rng)
        noise = self.sigma * rng.standard_normal(size=X.shape)
        X_plus = X @ self.A.T + U @ self.B.T + noise
        U_plus = rng.uniform(u_bounds[0], u_bounds[1], size=U.shape)
        return X, U, X_plus, U_plus

    def cost(self, X, U):
        X = np.atleast_2d(X); U = np.atleast_2d(U)
        Lx = np.einsum('ij,jk,ik->i', X, self.Q, X)
        Lu = np.einsum('ij,jk,ik->i', U, self.R, U)
        return Lx + Lu
    
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

