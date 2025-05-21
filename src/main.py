import numpy as np
from scipy.linalg import qr
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve_discrete_are


# ---------------------------------------------
# 1. Problem setup and synthetic data creation
# ---------------------------------------------
# Dimensions
dx = 2     # state dimension
du = 1     # input dimension
degree = 1 # degree of monomial basis
gamma = 0.9

# Sample sizes
N = 1000      # number of exploration samples
N_off = 5000  # offline pool size

# Define simple linear dynamics for data generation
A = np.array([[0.8, 0.1],
              [0.0, 0.9]])
B = np.array([[1.0],
              [0.5]])

# Optional: set seed for reproducibility
np.random.seed(0)

# Generate exploration data: x_i, u_i, x_i+, w_i
x = np.random.randn(N, dx)
u = np.random.randn(N, du)
# noise = 0.1 * np.random.randn(N, dx)
x_next = x @ A.T + u @ B.T
w = np.random.randn(N, du)

# Stack into z, z_next
z = np.hstack([x, u])         # shape (N, dx+du)
z_next = np.hstack([x_next, w])

# -------------------------------------------------
# 2. Feature map p(z): all monomials up to 'degree'
# -------------------------------------------------
poly = PolynomialFeatures(degree, include_bias=False)
# Fit and transform on combined data to ensure estimator is fitted
Z_all = np.vstack([z, z_next])         # shape (2N, dx+du)
P_all = poly.fit_transform(Z_all)      # shape (2N, d)
d = P_all.shape[1]                     # feature dimension
D = d * (d + 1) // 2                   # # free entries in symmetric matrix

# Split back into P_z and P_z_next
P_z = P_all[:N]        # shape (N, d)
P_z_next = P_all[N:]   # shape (N, d)

# ---------------------------------------------------------
# 3. Offline pool and pivoted-QR for selecting dictionary
# ---------------------------------------------------------
tilde_y = np.random.randn(N_off, dx + du)
P_tilde = poly.transform(tilde_y)   # shape (N_off, d)

# Build V: each column j is vec(p(tilde_y_j) p(tilde_y_j)^T)
tri_idx = np.triu_indices(d)
V = np.zeros((D, N_off))
for j in range(N_off):
    Mj = np.outer(P_tilde[j], P_tilde[j])
    V[:, j] = Mj[tri_idx]

# Pivoted QR to choose heavy columns
Q, R, pivots = qr(V, pivoting=True)
# Determine M by thresholding R's diagonal
tol = 1e-6
diagR = np.abs(np.diag(R))
M = np.sum(diagR > tol)
selected = pivots[:M]

# Extract the y_i's
y = tilde_y[selected]       # shape (M, dx+du)
P_y = P_tilde[selected]     # shape (M, d)
# Precompute atom matrices G_j = p(y_j)p(y_j)^T
G_atoms = [np.outer(P_y[j], P_y[j]) for j in range(M)]

# ---------------------------------------------
# 4. Build F_i = p(z_i)p(z_i)^T - gamma p(z_i+)p(z_i+)^T
# ---------------------------------------------
F_atoms = [np.outer(P_z[i], P_z[i]) - gamma * np.outer(P_z_next[i], P_z_next[i]) for i in range(N)]

# -------------------------------------------------
# 5. Solve LP for λ and μ simultaneously
#    sum_i λ_i F_i = sum_j μ_j G_j
# -------------------------------------------------
# Decision variables
lambda_var = cp.Variable(N, nonneg=True)
mu_var = cp.Variable(M, nonneg=True)

# Moment-matching equality constraint (matrix form)
moment_match = sum(lambda_var[i] * F_atoms[i] for i in range(N)) - \
                sum(mu_var[j]     * G_atoms[j] for j in range(M))
constraints = [moment_match == 0]

# Break scale invariance: enforce sum(lambda)=1 and match mu-weighted to identity
# constraints += [cp.sum(lambda_var) == 1]
I_d = np.eye(d)
C_approx = sum(mu_var[j] * G_atoms[j] for j in range(M))

# Objective: minimize Frobenius norm ||C_approx - I||_F
objective = cp.Minimize(cp.norm(C_approx - I_d, 'fro'))

# Solve LP
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)

print("Status:", prob.status)
if prob.status in ["optimal", "optimal_inaccurate"]:
    lambda_vals = lambda_var.value
    mu_vals = mu_var.value
    print("||C_approx - I||_F:", np.linalg.norm(
        sum(mu_vals[j] * G_atoms[j] for j in range(M)) - I_d, 'fro'))
    print("Sum(lambda):", np.sum(lambda_vals))
    print("Nonzero μ count:", np.sum(mu_vals > 1e-6))
    print("μ[:10]:", mu_vals[:10])
else:
    print("No valid solution; status", prob.status)

# ------------------------------------
# 6. New LP: maximize α^T vec_C_approx subject to constraints
# ------------------------------------
# Helper to vectorize p(z)p(z)^T
def outer_p(zu):
    p = poly.transform(np.atleast_2d(zu)).ravel()
    Mmat = np.outer(p, p)
    return Mmat[np.triu_indices(d)]  # returns D-vector


# Toy cost L
tmp_u = u.reshape(N, du)
L = (x**2).sum(axis=1) + (tmp_u**2).sum(axis=1)

# Decision variable α with bounds
alpha = cp.Variable(D, nonneg=True)
tri_idx = np.triu_indices(d)
C_approx = C_approx.value
vec_C_approx = cp.hstack([C_approx[i, j] for i, j in zip(*tri_idx)])

# Constraints: α^T [p(z_i)p(z_i)^T - γ p(z_i+)p(z_i+)^T]_vec ≤ L[i]
constraints_lp = []
for i in range(N):
    v_diff = outer_p(np.hstack([x[i], u[i]])) - gamma * outer_p(np.hstack([x_next[i], w[i]]))
    constraints_lp.append(alpha @ v_diff <= L[i])

objective_lp = cp.Maximize(alpha @ vec_C_approx)

# Solve LP
prob_lp = cp.Problem(objective_lp, constraints_lp)
try:
    prob_lp.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-4)
    print("Status:", prob_lp.status)
except Exception as e:
    print("Solver failed with error:", str(e))

# Display results
print("New LP status:", prob_lp.status)
print("Optimal α (first 10):", alpha.value[:10])


# === True Q-function computation for LQR ===
# def compute_optimal_q_function(A, B, Q, R):
#     # Solve the discrete-time algebraic Riccati equation
#     P = solve_discrete_are(A, B, Q, R)

#     # Compute the optimal policy gain K
#     K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

#     # Q-function parameters
#     def Q_function(x, u):
#         x = np.asarray(x).reshape(-1, 1)
#         u = np.asarray(u).reshape(-1, 1)
#         next_x = A @ x + B @ u
#         return float(x.T @ Q @ x + u.T @ R @ u + next_x.T @ P @ next_x)

#     return Q_function, P, K

# Q_cost = np.eye(dx)
# R_cost = np.eye(du)

# Q_func, P_opt, K_opt = compute_optimal_q_function(A, B, Q, R)

# # === Recover Q matrix from alpha ===
# def unvec_to_symmetric_mat(vec, dim, tri_idx):
#     mat = np.zeros((dim, dim))
#     mat[tri_idx] = vec
#     mat[(tri_idx[1], tri_idx[0])] = vec
#     return mat

# Q_learned_mat = unvec_to_symmetric_mat(alpha.value, d, tri_idx)

# # === Evaluate Q functions ===
# def true_Q(z):
#     return np.einsum('bi,ij,bj->b', z, Q_true_mat, z)

# def learned_Q(z):
#     pz = poly.transform(z)
#     return np.einsum('bi,ij,bj->b', pz, Q_learned_mat, pz)

# Q_true_vals = true_Q(z)
# Q_learned_vals = learned_Q(z)

# # trace_diff = np.trace((Q_learned_mat - Q_true_mat) @ (Q_learned_mat - Q_true_mat))
# mse = np.mean((Q_true_vals - Q_learned_vals)**2)

# # print("Trace of (Q_learned - Q_true)^2:", trace_diff)
# print("Mean squared error over samples:", mse)

Q = np.eye(dx)
R = np.eye(du)

# Solve for P
P = solve_discrete_are(A, B, Q, R)

# Define the true Q-function: Q(x, u) = x^T Q x + u^T R u + \gamma (Ax+Bu)^T P (Ax+Bu)
def true_q(x, u):
    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)
    xu_cost = x.T @ Q @ x + u.T @ R @ u
    next_x = A @ x + B @ u
    future_cost = gamma * (next_x.T @ P @ next_x)
    return (xu_cost + future_cost).item()

# ---------------------------------------------
# 3. Fit a Q-function using polynomial features
# ---------------------------------------------
poly = PolynomialFeatures(degree=degree)
Z = poly.fit_transform(np.hstack([x, u]))  # basis features for (x, u)
y = np.array([true_q(x[i], u[i]) for i in range(N)])  # targets: true Q(x, u)

# Solve least squares regression
theta = np.linalg.lstsq(Z, y, rcond=None)[0]

# Learned Q function
def learned_q(xi, ui):
    z_feat = poly.transform(np.hstack([xi, ui]).reshape(1, -1))
    return (z_feat @ theta).item()


# ---------------------------------------------
# 4. Compare learned Q vs true Q
# ---------------------------------------------
x_test = np.random.randn(100, dx)
u_test = np.random.randn(100, du)

# Print learned and true Q values for comparison
L = np.eye(dx + du)
Q_true_mat = L + gamma * np.concatenate((A, B), axis=1).T @ P @ np.concatenate((A, B), axis=1)
print("true Q matrix:\n", Q_true_mat)

true_vals = np.array([true_q(x_test[i], u_test[i]) for i in range(100)])
learned_vals = np.array([learned_q(x_test[i], u_test[i]) for i in range(100)])

mse = np.mean((true_vals - learned_vals)**2)
print(f"Mean Squared Error between true Q and learned Q: {mse:.2e}")