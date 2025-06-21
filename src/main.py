import numpy as np
from scipy.linalg import solve_discrete_are
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures

# -------------------------
# 1. Problem & Data Setup
# -------------------------
dx = 2     # state dimension
du = 1     # input dimension
degree = 1 # degree of monomial basis 
gamma = 0.99

# Sample sizes
N = 2000      # number of exploration samples
M = 500       # offline pool size

def is_stable(A):
    """Returns True if spectral radius of A is <1."""
    eigs = np.linalg.eigvals(A)
    return np.max(np.abs(eigs)) < 1

def spectral_norm(A):
    """Returns the largest singular value of A."""
    return np.linalg.svd(A, compute_uv=False)[0]

A = np.array([[-0.128308,   0.40262567],
 [-1.42454314, -0.51783493]])
B = np.array([[-0.76709544],
 [-0.50440709]])

def run_pipeline(A, B, seed=0):
    """
    Given A (dx×dx) and B (dx×du), runs your data-gen → LP → Q-learning
    pipeline and returns metrics: (fro_norm, mse)
    """
    np.random.seed(seed)

    # Generate exploration data: x_i, u_i, x_i+, w_i
    x = np.random.uniform(-10.0, 10.0, size=(N, dx))
    u = np.random.uniform(-10.0, 10.0, size=(N, du))
    x_next = x @ A.T + u @ B.T
    w = np.random.uniform(-10.0, 10.0, size=(N, du))

    # z = [x, u],  z_next = [x_next, w]
    z = np.hstack([x, u])         # shape (N, dx+du)
    z_next = np.hstack([x_next, w])

    # ----------------------------------------------
    # 2. Feature map p(z): all monomials up to ‘degree’
    # ----------------------------------------------
    poly = PolynomialFeatures(degree, include_bias=False)

    # Combine z and z_next to fit the PolynomialFeatures
    Z_all = np.vstack([z, z_next])         # shape (2N, dx+du)
    P_all = poly.fit_transform(Z_all)      # shape (2N, d)
    d = P_all.shape[1]                     # feature dimension

    # Split back into P_z and P_z_next
    P_z = P_all[:N]        # shape (N, d)
    P_z_next = P_all[N:]   # shape (N, d)

    # Build P_y for the offline pool y
    y = np.random.uniform(-10.0, 10.0, size=(M, dx + du))
    P_y = poly.transform(y)   # shape (M, d)


    # =====================================================================
    # 3. Solve LP for λ and μ simultaneously (vectorized implementation)
    #    sum_i λ_i F_i = sum_j μ_j G_j  where F_i = p(z_i)p(z_i)ᵀ − γ p(z_i+)p(z_i+)ᵀ
    # =====================================================================

    # Create CVXPY variables
    lambda_var = cp.Variable(N, nonneg=True)  # λ ∈ ℝ^N₊
    mu_var     = cp.Variable(M, nonneg=True)  # μ ∈ ℝ^M₊

    # Convert P_z, P_z_next, P_y into CVXPY Constants
    Pz_const       = cp.Constant(P_z)        # shape (N, d)
    Pz_next_const  = cp.Constant(P_z_next)   # shape (N, d)
    Py_const       = cp.Constant(P_y)        # shape (M, d)

    # Build the moment-match matrix in one shot:
    #    sum_i λ_i [P_z[i] P_z[i]ᵀ] = P_zᵀ · diag(λ) · P_z
    sum_PzPzT = Pz_const.T @ cp.diag(lambda_var) @ Pz_const            # shape (d, d)
    sum_PznPznT = Pz_next_const.T @ cp.diag(lambda_var) @ Pz_next_const  # shape (d, d)
    sum_PyPyT = Py_const.T @ cp.diag(mu_var) @ Py_const                 # shape (d, d)

    moment_match = sum_PzPzT - gamma * sum_PznPznT - sum_PyPyT   # shape (d, d)

    constraints = []
    # Enforce moment_match == 0_(d×d)
    constraints += [ moment_match == np.zeros((d, d)) ]

    # Break scale-invariance:
    constraints += [ cp.sum(lambda_var) == 1 ]

    # Build C_approx = ∑ₘ μ_j [P_y[j] P_y[j]ᵀ] = P_yᵀ · diag(μ) · P_y
    C_approx = sum_PyPyT  # shape (d, d)

    # Objective: minimize ‖C_approx − I‖_F
    I_d = np.eye(d)
    objective = cp.Minimize(cp.norm(C_approx - I_d, "fro"))

    # Solve LP #1
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params={})

    print("Status (LP for λ,μ):", prob.status)
    if prob.status in ["optimal", "optimal_inaccurate"]:
        lambda_vals = lambda_var.value
        mu_vals     = mu_var.value
        # Reconstruct C_approx as a numeric matrix and report
        C_num = P_y.T @ np.diag(mu_vals) @ P_y
        print("||C_approx - I||_F:", np.linalg.norm(C_num - I_d, "fro"))
        print("Sum(λ):", np.sum(lambda_vals))
        print("Nonzero μ count:", np.sum(mu_vals > 1e-6))
    else:
        print("No valid solution; status", prob.status)

    # Extract the final λ, μ
    λ = lambda_var.value            # shape (N,)
    μ = mu_var.value                # shape (M,)

    # Reconstruct M_lambda and M_mu for diagnostic
    M_lambda = P_z.T @ np.diag(λ) @ P_z - gamma * (P_z_next.T @ np.diag(λ) @ P_z_next)
    M_mu     = P_y.T @ np.diag(μ) @ P_y
    diff_mom = M_lambda - M_mu
    print("||M_lambda - M_mu||_F:", np.linalg.norm(diff_mom, "fro"))
    print("max|M_lambda - M_mu|:", np.max(np.abs(diff_mom)))


    # ===========================================================
    # 4. New LP: maximize αᵀ vec(C_approx) subject to constraints
    # ===========================================================

    C_val = P_y.T @ np.diag(μ) @ P_y           # shape (d, d)
    c_vec = C_val.flatten(order="C")          # shape (d²,)

    # Define Q_var ∈ ℝ^{d×d}, PSD, and its vectorized form
    # REMOVE THE CONSTRAINT PSD=True, HOWEVER PAY ATTENTION WHEN NON LQR
    Q_var     = cp.Variable((d, d), PSD=True)               # matrix variable
    Q_var_vec = cp.reshape(Q_var, (d * d,), order="C")     # flatten in row-major

    # compute L_xu (toy cost) exactly as before:
    L_xu = (x**2).sum(axis=1) + (u**2).sum(axis=1)   # shape (N,)

    # Build a 3D array of shape (N, d, d) in one go, then flatten:
    F_3D = (P_z[:, :, None] * P_z[:, None, :]) - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])
    # Now reshape to (N, d*d), preserving row-major (“C”) ordering:
    F_mat = F_3D.reshape((N, d * d), order="C")     # shape (N, d²)

    # single vectorized constraint:
    constraints_lp = [ F_mat @ Q_var_vec <= L_xu ]

    # Objective: maximize c_vecᵀ · Q_var_vec
    objective_lp = cp.Maximize(c_vec @ Q_var_vec)

    # Solve LP #2
    prob_lp = cp.Problem(objective_lp, constraints_lp)
    try:
        # PAY ATTENTION: how the solver treats matrix symmetry
        prob_lp.solve(solver=cp.MOSEK, mosek_params={})
        print("Status (LP for Q):", prob_lp.status)
    except Exception as e:
        print("Solver failed with error:", e)

    Q_learned_vec = Q_var_vec.value    # shape (d²,)
    Q_learned_mat = Q_learned_vec.reshape((d, d), order="C")  # shape (d, d)

    # Quick feasibility check on a few random i:
    for i0 in np.random.choice(N, size=5, replace=False):
        Fi0 = (np.outer(P_z[i0], P_z[i0]) - gamma * np.outer(P_z_next[i0], P_z_next[i0]))
        lhs = Fi0.flatten(order="C") @ Q_learned_vec
        rhs = L_xu[i0]
        print(f"i0={i0}, (Fᵢ Q) = {lhs:.6f}, Lᵢ = {rhs:.6f}, diff = {lhs - rhs:.6f}")

    print("New LP status:", prob_lp.status)
    print("Q_learned (feature‐space) matrix:\n", Q_learned_mat)


    # ---------------------------------------------------
    # 7. Compare learned Q vs true Q (unchanged)
    # ---------------------------------------------------
    Q_cost = np.eye(dx)
    R_cost = np.eye(du)

    # solve discrete‐ARE with discount → scale A,B
    A_disc = np.sqrt(gamma) * A
    B_disc = np.sqrt(gamma) * B
    P_true = solve_discrete_are(A_disc, B_disc, Q_cost, R_cost)

    # Also print the “true Q matrix” in the original x‐u space for reference
    L_aug  = np.eye(dx + du)
    AB_cat = np.concatenate((A, B), axis=1)      # shape (2, 3)
    Q_true_mat = L_aug + gamma * AB_cat.T @ P_true @ AB_cat
    print("true Q matrix (x,u) form:\n", Q_true_mat)

    # Compute the difference between the true Q matrix and the learned Q matrix
    Q_diff = Q_true_mat - Q_learned_mat

    # Frobenius norm of the difference
    frobenius_norm = np.linalg.norm(Q_diff, "fro")
    print(f"Frobenius norm of the difference between true Q and learned Q: {frobenius_norm:.2e}")

    # Trace of the difference
    trace_diff = np.trace(Q_diff)
    print(f"Trace of the difference between true Q and learned Q: {trace_diff:.2e}")

    return frobenius_norm, trace_diff


num_trials = 50
results = []
for t in range(num_trials):
    # draw a random stable A: sample then shrink if needed
    A_rand = np.random.randn(dx, dx) * 0.8
    # enforce spectral radius <1 by scaling
    rho = max(abs(np.linalg.eigvals(A_rand)))
    if rho >= 1:
        A_rand = A_rand / (1.1*rho)
    B_rand = np.random.randn(dx, du)

    σ_max = spectral_norm(A_rand)   # largest singular value of A_rand
    fro_err, trace_diff = run_pipeline(A_rand, B_rand, seed=t)

    results.append({
        'trial': t,
        'sigma_max': σ_max,
        'frobenius': fro_err,
        'trace_diff': trace_diff,
        'spectral_stable': True
    })

# split into two groups and compare
group1 = [r for r in results if r['sigma_max'] < 1]
group2 = [r for r in results if r['sigma_max'] >= 1]
print(f"Avg Frobenius error σ_max<1: {np.nanmean([r['frobenius'] for r in group1]):.2e}")
print(f"Avg Frobenius error σ_max≥1: {np.nanmean([r['frobenius'] for r in group2]):.2e}")
print(f"Avg trace diff        σ_max<1: {np.nanmean([r['trace_diff']       for r in group1]):.2e}")
print(f"Avg trace diff        σ_max≥1: {np.nanmean([r['trace_diff']       for r in group2]):.2e}")