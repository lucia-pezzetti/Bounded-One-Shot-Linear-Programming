
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import linprog

def poly_features(z, degree=2, include_bias=True):
    """
    Sparse-friendly polynomial features for vector z (1D). 
    Returns a feature vector phi(z) including all monomials up to `degree`.
    Order: biases first (if include_bias), then degree-1 terms, then degree-2 (with i<=j), etc.
    This mirrors common polynomial feature maps used in LQR experiments.
    """
    z = np.asarray(z).ravel()
    feats = []
    if include_bias:
        feats.append(1.0)
    # degree 1
    feats.extend(z.tolist())
    if degree >= 2:
        for i, j in combinations_with_replacement(range(len(z)), 2):
            feats.append(z[i] * z[j])
    if degree >= 3:
        # simple cubic (diagonal only) to keep size moderate; extend as needed
        for i in range(len(z)):
            feats.append(z[i] ** 3)
    return np.array(feats, dtype=float)

def vec_symmetric(M):
    """Vectorize the upper-triangular part of a symmetric matrix M (including diagonal)."""
    n = M.shape[0]
    idx = np.triu_indices(n)
    return M[idx]

def mat_from_vec_symmetric(v, n):
    """Reconstruct a symmetric nxn matrix from its upper-triangular vector v."""
    M = np.zeros((n, n), dtype=float)
    idx = np.triu_indices(n)
    M[idx] = v
    M[(idx[1], idx[0])] = v  # mirror
    return M

def outer_vec(phi):
    """Return the vectorized upper-triangular of phi phi^T."""
    P = np.outer(phi, phi)
    return vec_symmetric(P)

def build_lp_matrices(Phi, Phi_next, ell, gamma=0.99):
    """
    Build A, b, c for the LP:
        minimize t
        s.t.  -t <= (phi_i^T Q phi_i) - (ell_i + gamma * phi_i+^T Q phi_i+) <= t
    Decision variables: q (sym upper-tri of Q) and scalar t at the end.
    """
    N, d = Phi.shape
    # precompute feature outer products (upper-tri vectorization)
    F = np.vstack([outer_vec(Phi[i]) for i in range(N)])        # shape (N, d*(d+1)/2)
    Fp = np.vstack([outer_vec(Phi_next[i]) for i in range(N)])  # same shape

    m = F.shape[1]  # number of unique symmetric elements
    # Variables ordering: [q (m entries); t (1 entry)]
    # For each i: (F_i - gamma Fp_i)·q - t <=  ell_i
    #             -(F_i - gamma Fp_i)·q - t <= -ell_i
    A_ub_top = np.hstack([ (F - gamma * Fp), -np.ones((N, 1)) ])
    b_ub_top =  ell.copy()
    A_ub_bot = np.hstack([-(F - gamma * Fp), -np.ones((N, 1)) ])
    b_ub_bot = -ell.copy()

    A_ub = np.vstack([A_ub_top, A_ub_bot])
    b_ub = np.concatenate([b_ub_top, b_ub_bot])

    # Objective: minimize t -> c = [0,...,0, 1]
    c = np.zeros(m + 1)
    c[-1] = 1.0

    # Bounds: q free, t >= 0
    bounds = [(None, None)] * m + [(0, None)]
    return A_ub, b_ub, c, bounds, m

def solve_classical_lp(Phi, Phi_next, ell, gamma=0.99, method='highs'):
    A_ub, b_ub, c, bounds, m = build_lp_matrices(Phi, Phi_next, ell, gamma)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    q_hat = res.x[:m]
    t_star = res.x[-1]
    Q_hat = mat_from_vec_symmetric(q_hat, Phi.shape[1])
    return Q_hat, t_star, res

def simulate_linear_system(A, B, Qc, Rc, x0, N, u_policy=None, noise_std=0.0, seed=0):
    """
    Simulate x_{k+1} = A x_k + B u_k + w_k, cost ell_k = x_k^T Qc x_k + u_k^T Rc u_k
    Returns arrays: X (N,n), U (N,m), Xp (N,n), ell (N,)
    """
    rng = np.random.default_rng(seed)
    n, m = B.shape[0], B.shape[1]
    X = np.zeros((N, n))
    U = np.zeros((N, m))
    Xp = np.zeros((N, n))
    ell = np.zeros(N)

    x = x0.copy()
    for k in range(N):
        if u_policy is None:
            u = rng.uniform(-1, 1, size=(m,))  # exploratory inputs
        else:
            u = u_policy(x)
        w = rng.normal(0, noise_std, size=(n,))
        xp = A @ x + B @ u + w
        cost = x @ Qc @ x + u @ Rc @ u
        X[k], U[k], Xp[k], ell[k] = x, u, xp, cost
        x = xp
    return X, U, Xp, ell

def build_feature_matrices(X, U, Xp, degree=2, include_bias=True):
    Z  = np.hstack([X, U])
    Zp = np.hstack([Xp, np.zeros_like(U)])  # for value of next state paired with next action = 0 (evaluation)
    # If you want to use a greedy/target policy for u^+, replace zeros with that policy.
    Phi  = np.vstack([poly_features(z, degree, include_bias) for z in Z])
    Phi_next = np.vstack([poly_features(zp, degree, include_bias) for zp in Zp])
    return Phi, Phi_next

def E_Q_metric(Phi, Phi_next, ell, Q, gamma=0.99):
    """Compute average absolute Bellman residual under identity weighting."""
    v  = np.einsum('ij, jk, ik -> i', Phi, Q, Phi)
    vp = np.einsum('ij, jk, ik -> i', Phi_next, Q, Phi_next)
    td = v - (ell + gamma * vp)
    return np.mean(np.abs(td))

def main():
    # Example system (2D state, 1D input)
    A = np.array([[0.9, 0.1],
                  [0.0, 0.95]])
    B = np.array([[0.1],
                  [0.05]])
    Qc = np.eye(2)
    Rc = np.eye(1) * 0.1
    gamma = 0.99

    # Simulate data
    x0 = np.array([1.0, -1.0])
    X, U, Xp, ell = simulate_linear_system(A, B, Qc, Rc, x0, N=2000, noise_std=0.01, seed=42)

    # Build *fixed* polynomial features (match lqr_optimized degree/bias as needed)
    degree = 2
    include_bias = True
    Phi, Phi_next = build_feature_matrices(X, U, Xp, degree, include_bias)

    # Solve classical LP with identity covariance (equal weights)
    Q_hat, t_star, res = solve_classical_lp(Phi, Phi_next, ell, gamma=gamma, method='highs')

    print("LP status:", res.message)
    print("Minimax residual t*:", t_star)
    print("Feature dim:", Phi.shape[1])
    print("E_Q (avg |TD|) under identity weights:", E_Q_metric(Phi, Phi_next, ell, Q_hat, gamma))

if __name__ == "__main__":
    main()