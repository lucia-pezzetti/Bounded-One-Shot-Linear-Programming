import numpy as np
import argparse
import os
import json
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
from scipy import sparse

# Local imports (optimized vectorized sampling)
from dynamical_systems_optimized import dlqr
from scipy.optimize import linprog

def _pack_upper_weighted(M: np.ndarray) -> np.ndarray:
    """
    Pack the upper-triangular of a symmetric matrix M into a vector
    with off-diagonals multiplied by 2, so that <Q, M> = v^T q for symmetric Q.
    """
    d = M.shape[0]
    iu = np.triu_indices(d)
    v = M[iu].astype(float).copy()
    off = iu[0] != iu[1]
    v[off] *= 2.0
    return v

def _vec_len_upper(d: int) -> int:
    return d * (d + 1) // 2

def classical_lp_identity(Phi_csr, Phi_next_csr, costs, gamma=0.99, method="highs"):
    """
    Solve the classical minimax LP with identity weighting:
        min_{Q symmetric, t>=0} t
        s.t.  -t <= phi_i^T Q phi_i - (ell_i + gamma * phi_i+^T Q phi_i+) <= t,  i=1..N

    Inputs:
        - Phi_csr:      N x d CSR matrix of features for z=[x;u]
        - Phi_next_csr: N x d CSR matrix of features for z^+
        - costs:        length-N vector with immediate costs ell_i
        - gamma:        discount factor
        - method:       SciPy linprog method (e.g., "highs")

    Returns:
        - Q_hat (dense d x d symmetric), t_star (float), linprog result object
    """
    import numpy as _np
    from scipy import sparse as _sp

    # Ensure CSR
    if not _sp.issparse(Phi_csr):
        Phi_csr = _sp.csr_matrix(Phi_csr)
    if not _sp.issparse(Phi_next_csr):
        Phi_next_csr = _sp.csr_matrix(Phi_next_csr)

    assert Phi_csr.shape == Phi_next_csr.shape
    N, d = Phi_csr.shape
    m = _vec_len_upper(d)

    rows = []
    rhs  = []

    # Build constraints
    for i in range(N):
        p  = Phi_csr.getrow(i).toarray().ravel()
        pn = Phi_next_csr.getrow(i).toarray().ravel()
        Mi = _np.outer(p, p) - gamma * _np.outer(pn, pn)
        vi = _pack_upper_weighted(Mi)  # length m

        # (vi)·q - t <=  ell_i
        row1 = _np.concatenate([vi, _np.array([-1.0])])
        rows.append(row1)
        rhs.append(float(costs[i]))

        # -(vi)·q - t <= -ell_i
        row2 = _np.concatenate([-vi, _np.array([-1.0])])
        rows.append(row2)
        rhs.append(-float(costs[i]))

    A_ub = _np.vstack(rows)
    b_ub = _np.array(rhs, dtype=float)

    # Objective: minimize t  => c = [0,...,0, 1]
    c = _np.zeros(m + 1)
    c[-1] = 1.0

    # Bounds: q free, t >= 0
    bounds = [(None, None)] * m + [(0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    if not res.success:
        raise RuntimeError(f"Classical LP failed: {res.message}")

    q_hat = res.x[:m]
    t_star = float(res.x[-1])

    # Reconstruct symmetric Q
    Q = _np.zeros((d, d), dtype=float)
    iu = _np.triu_indices(d)
    Q[iu] = q_hat
    Q[(iu[1], iu[0])] = q_hat  # mirror
    return Q, t_star, res

def poly_features_chunked(Z, degree, include_bias=False, chunk=20000):
    """
    Sparse polynomial features computed in chunks to avoid giant dense matrices.
    Returns a CSR matrix.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    # Fit once to set the input dimension
    poly.fit(np.zeros((1, Z.shape[1])))
    blocks = []
    for i in range(0, Z.shape[0], chunk):
        Zi = Z[i:i+chunk]
        Pi = poly.transform(Zi)
        if not sparse.issparse(Pi):
            Pi = sparse.csr_matrix(Pi)
        blocks.append(Pi)
    return sparse.vstack(blocks, format="csr")

def main():
    parser = argparse.ArgumentParser(description="Memory-optimized LQR via LP")
    parser.add_argument('--degree', type=int, default=1, help='Polynomial feature degree')
    parser.add_argument('--rho', type=float, default=0.1, help='Control cost weight')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--m_samples', type=int, default=500, help='Auxiliary samples')
    parser.add_argument('--x_bounds', type=float, nargs=2, default=(-3.0, 3.0), help='State bounds (x0, x1)')
    parser.add_argument('--u_bounds', type=float, nargs=2, default=(-1.0, 1.0), help='Action bounds (force)')
    parser.add_argument('--n_x', type=int, default=2, help='State dimension')
    parser.add_argument('--n_u', type=int, default=1, help='Action dimension')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default='None', help='Directory to load data')
    parser.add_argument('--chunk', type=int, default=20000, help='Chunk size for feature generation')
    parser.add_argument('--run_classical_lp', type=bool, default=False,
                    help='Also run classical LP with identity covariance')
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    rho = args.rho
    gamma = args.gamma
    N = int(args.n_samples)
    M = int(args.m_samples)
    dx = int(args.n_x); du = int(args.n_u)
    degree = int(args.degree)
    x_bounds = tuple(args.x_bounds); u_bounds = tuple(args.u_bounds)

    # System matrices
    if args.data_dir != 'None':
        with open(os.path.join(args.data_dir, f'dx_{args.n_x}.json'), 'r') as f:
            res = json.load(f)
            A = np.array(res['A'], dtype=float)
            B = np.array(res['B'], dtype=float)
            C = np.array(res['C'], dtype=float)
    else:
        A = np.array([[1.0, 0.1], [0.5, -0.5]], dtype=float)[:dx, :dx]
        B = np.array([[1.0], [0.5]], dtype=float)[:dx, :du]
        C = np.eye(dx, dtype=float)

    system = dlqr(A, B, C, rho, gamma)

    # Generate transitions (vectorized in dynamical_systems_optimized)
    x, u, x_plus, u_plus = system.generate_samples(x_bounds, u_bounds, n_samples=N, rng=rng)
    z = np.concatenate([x, u], axis=1)            # (N, dx+du)
    z_plus = np.concatenate([x_plus, u_plus], 1)  # (N, dx+du)

    # Auxiliary samples
    x_aux, u_aux = system.generate_samples_auxiliary(x_bounds, u_bounds, n_samples=M, rng=rng)
    y = np.concatenate([x_aux, u_aux], axis=1)    # (M, dx+du)

    # Sparse polynomial features (chunked)
    Z_all = np.vstack([z, z_plus])                # (2N, dx+du)
    P_all = poly_features_chunked(Z_all, degree=degree, include_bias=False, chunk=args.chunk)  # CSR (2N, d)
    d = P_all.shape[1]
    P_z = P_all[:N, :]
    P_z_next = P_all[N:, :]
    P_y = poly_features_chunked(y, degree=degree, include_bias=False, chunk=args.chunk)

    # =====================
    # LP #1: λ, μ (moment matching) without forming diag()
    # =====================
    lambda_var = cp.Variable(N, nonneg=True)
    mu_var = cp.Variable(M, nonneg=True)

    # Convert to CVXPY constants (sparse)
    Pz_c = cp.Constant(P_z)
    Pzn_c = cp.Constant(P_z_next)
    Py_c = cp.Constant(P_y)

    # Helper: weighted Gram without cp.diag: X^T diag(w) X == X^T (w .* X)
    def weighted_gram(Xc, w):
        return Xc.T @ cp.multiply(Xc, w[:, None])

    sum_PzPzT   = weighted_gram(Pz_c,  lambda_var)
    sum_PznPznT = weighted_gram(Pzn_c, lambda_var)
    sum_PyPyT   = weighted_gram(Py_c,  mu_var)

    moment_match = sum_PzPzT - gamma * sum_PznPznT - sum_PyPyT
    constraints = [moment_match == 0,
                   cp.sum(lambda_var) == 1]

    # Objective: make C ≈ I without materializing dense arrays
    I_d = cp.Constant(sparse.eye(d, format="csr"))
    C_approx = sum_PyPyT
    obj1 = cp.Minimize(cp.norm(C_approx - I_d, "fro"))

    prob1 = cp.Problem(obj1, constraints)
    try:
        prob1.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        try:
            prob1.solve(solver=cp.SCS, verbose=False)
        except Exception:
            prob1.solve(verbose=False)

    if prob1.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP#1 failed: {prob1.status}")

    lam = lambda_var.value
    mu = mu_var.value

    # =====================
    # LP #2: Q in feature space (avoid F_3D)
    # =====================
    Q = cp.Variable((d, d), symmetric=True)

    # Stage costs
    L_xu = system.cost(x, u)

    # Constraints: for each i, <Q, p p^T - gamma p+ p+^T> <= L_i
    cons2 = []
    Pz = P_z.tocsr(); Pzn = P_z_next.tocsr()
    for i in range(N):
        p = Pz.getrow(i).toarray().ravel()
        pn = Pzn.getrow(i).toarray().ravel()
        Mi = np.outer(p, p) - gamma * np.outer(pn, pn)
        cons2.append(cp.sum(cp.multiply(Q, Mi)) <= L_xu[i])

    # Objective: sum_i mu_i * <Q, y_i y_i^T>
    Py = P_y.tocsr()
    terms = []
    for i in range(M):
        yi = Py.getrow(i).toarray().ravel()
        terms.append(mu[i] * cp.sum(cp.multiply(Q, np.outer(yi, yi))))
    obj2 = cp.Maximize(cp.sum(terms))

    prob2 = cp.Problem(obj2, cons2)
    try:
        prob2.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        try:
            prob2.solve(solver=cp.SCS, verbose=False)
        except Exception:
            prob2.solve(verbose=False)

    if prob2.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP#2 failed: {prob2.status}")

    Q_learned = Q.value
    Q_learned_mat = Q_learned.reshape((d, d), order="C")
    print("Learned Q shape:", Q_learned.shape)

    E_Q_learned = prob2.value
    print("Learned E_Q:", E_Q_learned)

    print("Q_learned (feature‐space) matrix shape:", Q_learned_mat.shape)
    print("eigenvalues of Q_learned:", np.linalg.eigvalsh(Q_learned_mat))

    c = P_y.T @ np.diag(mu) @ P_y  # shape (d,)
    c_vec = c.flatten(order="C")  # shape (d²,)
    # ===========================================================
    # 5. Compare Q learned with optimal Q since we are in LQR
    # ===========================================================
    P, K, q = system.optimal_solution()

    # xu_weights = np.concatenate([np.ones(dx), np.ones(du)], axis=0)
    Q_star, E_Qstar, gap = system.optimal_q(P, q, c_vec)

    print("Optimal Q matrix shape:", Q_star.shape)
    print("E_Qstar:", E_Qstar)
    print("Gap:", gap)

    print("eigenvalues of optimal Q:", np.linalg.eigvalsh(Q_star))

    print("Optimality gap (E_Q_learned - E_Qstar):", E_Q_learned - gap - E_Qstar)

    # evaluate trace difference of the learned Q and the optimal Q
    trace_diff = np.trace(Q_learned_mat) - np.trace(Q_star)
    print("Trace of learned Q:", np.trace(Q_learned_mat))
    print("Trace of optimal Q:", np.trace(Q_star))
    print("Trace difference (learned Q - optimal Q):", trace_diff)

    # ===== Optional: Classical LP (identity covariance) over the SAME features =====
    if args.run_classical_lp:
        try:
            Phi_csr = P_z.tocsr() if hasattr(P_z, 'tocsr') else P_z
            Phi_next_csr = P_z_next.tocsr() if hasattr(P_z_next, 'tocsr') else P_z_next
            costs_vec = L_xu
        except NameError as e:
            raise RuntimeError(
                "Could not find feature matrices or costs in scope. "
                "Ensure P_z, P_z_next, and L_xu are defined before this block."
            ) from e

        Q_classic, t_star, lp_res = classical_lp_identity(
            Phi_csr, Phi_next_csr, costs_vec, gamma=args.gamma, method='highs'
        )

        # Compute identity-weight E_Q (avg |TD|) for reporting
        # v  = diag(Phi Q Phi^T), vp = diag(Phi+ Q Phi+^T)
        PhiQ  = Phi_csr @ Q_classic            # dense (N x d)
        v  = np.asarray(Phi_csr.multiply(PhiQ).sum(axis=1)).ravel()

        PhiQp = Phi_next_csr @ Q_classic       # dense (N x d)
        vp = np.asarray(Phi_next_csr.multiply(PhiQp).sum(axis=1)).ravel()
        td = v - (costs_vec + args.gamma * vp)
        E_Q_identity = float(np.mean(np.abs(td)))

        print("\n=== Classical LP (identity covariance) ===")
        print("LP status:", lp_res.message)
        print("Minimax residual t*:", t_star)
        print("E_Q (avg |TD|, identity weights):", E_Q_identity)


if __name__ == "__main__":
    main()
