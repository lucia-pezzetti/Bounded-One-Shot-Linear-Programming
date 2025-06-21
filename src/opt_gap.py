import numpy as np
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve_discrete_are, eigvals
import matplotlib.pyplot as plt
import mosek

def learn_Q(A, B, N=500, M=200, degree=1, gamma=0.99):
    """
    Given system matrices A∈ℝ^{dx×dx} and B∈ℝ^{dx×du},
    solve the vectorized LPs (λ, μ and then Q) to return:
      - Q_learned_mat ∈ ℝ^{d×d}  (feature‐space Q)
      - poly               (the fitted PolynomialFeatures object)
    If the LP fails, returns (None, None).
    """
    dx = A.shape[0]
    du = B.shape[1]

    # 1) Generate exploration data
    x = np.random.uniform(-10, 10, size=(N, dx))
    u = np.random.uniform(-10, 10, size=(N, du))
    x_next = x @ A.T + u @ B.T
    w = np.random.uniform(-10, 10, size=(N, du))

    z      = np.hstack([x, u])       # shape (N, dx+du)
    z_next = np.hstack([x_next, w])  # shape (N, dx+du)

    # 2) Build monomial features of degree=1 (no bias)
    poly = PolynomialFeatures(degree, include_bias=False)
    Z_all = np.vstack([z, z_next])        # shape (2N, dx+du)
    P_all = poly.fit_transform(Z_all)     # shape (2N, d)
    d = P_all.shape[1]

    P_z      = P_all[:N]      # (N, d)
    P_z_next = P_all[N:]      # (N, d)

    # Offline pool y (M random samples), and its features
    y   = np.random.uniform(-10, 10, size=(M, dx + du))
    P_y = poly.transform(y)   # (M, d)

    # 3) Solve LP for λ and μ (vectorized)
    lambda_var = cp.Variable(N, nonneg=True)
    mu_var     = cp.Variable(M, nonneg=True)

    Pz_const   = cp.Constant(P_z)       # (N, d)
    Pzn_const  = cp.Constant(P_z_next)  # (N, d)
    Py_const   = cp.Constant(P_y)       # (M, d)

    sum_PzPzT   = Pz_const.T @ cp.diag(lambda_var) @ Pz_const      # (d, d)
    sum_PznPznT = Pzn_const.T @ cp.diag(lambda_var) @ Pzn_const   # (d, d)
    sum_PyPyT   = Py_const.T @ cp.diag(mu_var) @ Py_const          # (d, d)

    moment_match = sum_PzPzT - gamma * sum_PznPznT - sum_PyPyT

    constraints = [
        moment_match == np.zeros((d, d)),  # matrix equality constraint
        cp.sum(lambda_var) == 1            # break scale invariance
    ]

    C_approx = sum_PyPyT
    I_d = np.eye(d)
    objective = cp.Minimize(cp.norm(C_approx - I_d, "fro"))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params={})
    print(f"LP #1 status: {prob.status}")

    mu_vals     = mu_var.value
    # Reconstruct C_approx as a numeric matrix and report
    C_num = P_y.T @ np.diag(mu_vals) @ P_y
    print("||C_approx - I||_F:", np.linalg.norm(C_num - I_d, "fro"))

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, None

    μ = mu_var.value                             # (M,)
    C_val = P_y.T @ np.diag(μ) @ P_y             # (d, d)
    c_vec = C_val.flatten(order="C")             # (d²,)

    # 4) Solve Q‐LP (vectorized)
    Q_var     = cp.Variable((d, d), PSD=True)
    Q_var_vec = cp.reshape(Q_var, (d * d,), order="C")

    L_xu = (x**2).sum(axis=1) + (u**2).sum(axis=1)  # (N,)

    # build F_mat: each row i = vec( p(z_i)p(z_i)ᵀ − γ p(z_i⁺)p(z_i⁺)ᵀ )
    F_3D = (P_z[:, :, None] * P_z[:, None, :]) \
         - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])  # shape (N, d, d)
    F_mat = F_3D.reshape((N, d * d), order="C")                    # shape (N, d²)

    constraints_lp = [F_mat @ Q_var_vec <= L_xu]   # vectorized inequalities
    objective_lp = cp.Maximize(c_vec @ Q_var_vec)
    prob_lp = cp.Problem(objective_lp, constraints_lp)
    prob_lp.solve(solver=cp.MOSEK, mosek_params={})

    if prob_lp.status not in ["optimal", "optimal_inaccurate"]:
        return None, None
    print(f"LP #2 status: {prob_lp.status}")

    Q_learned_vec = Q_var_vec.value                 # (d²,)
    Q_learned_mat = Q_learned_vec.reshape((d, d), order="C")  # (d, d)

    return Q_learned_mat, poly, C_num

def compute_Q_diff_matrix(A, B, Q_learned_mat, poly,
                           x_lo=-1, x_hi=1, u_lo=-1, u_hi=1,
                           degree=1, gamma=0.99):
    """
    Compute the explicit matrix difference Q_true - Q_learned
    using the analytic LQR expression:
      Q_true = [[Q + γ AᵀPA, γ AᵀPB], [γ BᵀPA, R + γ BᵀPB]]
    where P solves the discrete ARE.
    """
    dx = A.shape[0]
    du = B.shape[1]

    # Define cost matrices
    Q_cost = np.eye(dx)
    R_cost = np.eye(du)

    # Solve for true P via discounted ARE
    A_disc = np.sqrt(gamma) * A
    B_disc = np.sqrt(gamma) * B
    P_true = solve_discrete_are(A_disc, B_disc, Q_cost, R_cost)

    # Build the true Q block-matrix
    Q11 = Q_cost + gamma * A.T @ P_true @ A
    Q12 = gamma * A.T @ P_true @ B
    Q21 = gamma * B.T @ P_true @ A
    Q22 = R_cost + gamma * B.T @ P_true @ B
    Q_true_mat = np.block([[Q11, Q12], [Q21, Q22]])

    # Compute and return the difference matrix
    Q_diff_mat = Q_true_mat - Q_learned_mat
    return Q_diff_mat


# -----------------------------------------------
# MAIN: Sweep over state dimensions vs. optimality gap
# -----------------------------------------------
np.random.seed(0)

dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # you can extend beyond 5 if desired
gaps = []

for dx in dimensions:
    # Generate a random stable A (dx×dx) and random B (dx×1)
    R = np.random.randn(dx, dx)
    eigs = eigvals(R)
    scale = 0.8 / max(abs(eigs))       # force spectral radius < 1
    A_rand = R * scale
    B_rand = np.random.uniform(-1, 1, size=(dx, 1))
    # N = int(50 * (dx)**2)
    # M = int(20 * (dx)**2)
    N = 2000        # fixed over dimensions
    M = 1000        # fixed over dimensions

    # Learn feature-space Q by solving the two LPs
    Q_learned_mat, poly, C_num = learn_Q(A_rand, B_rand,
                                  N=N, M=M,
                                  degree=1, gamma=0.99)

    if Q_learned_mat is None:
        gaps.append(np.nan)
        print(f"Dimension {dx}: LP failed → gap = NaN")
        continue

    # Compute the optimality gap as the trace of the difference matrix
    Q_diff = compute_Q_diff_matrix(A_rand, B_rand, Q_learned_mat, poly,
                                    x_lo=-1, x_hi=1, u_lo=-1, u_hi=1,
                                    degree=1, gamma=0.99)
    print(f"Dimension {dx}: optimality gap ≈ {np.trace(Q_diff*C_num):.4f}")
    gaps.append(np.trace(Q_diff))

# 4) Plot state dimension vs. optimality gap
plt.figure(figsize=(6, 4))
plt.plot(dimensions, gaps, marker='o', linestyle='-')
plt.xlabel("State Dimension (dx)")
plt.ylabel("Optimality Gap")
plt.title("Optimality Gap vs. State Dimension")
plt.grid(True)
plt.tight_layout()
plt.show()
