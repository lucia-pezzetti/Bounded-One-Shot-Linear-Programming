'''
This script demonstrates the implementation of the proposed method on a synthetic 1D dataset.

Not fix \lambda a priori and try to find both together.

Compute analytical solution of this LQR problem
'''

import numpy as np
import cvxpy as cp
from scipy.optimize import nnls
import matplotlib.pyplot as plt

# Step 1: Define basis functions
def p(z):
    x, u = z
    return np.array([1, x, u, x**2, x*u, u**2])  # 6D basis

def outer_p(z):
    v = p(z)
    return np.outer(v, v)

def compute_analytical_Q_matrix(A=0.9, B=0.1, Q=1.0, R=1.0, gamma=0.9, basis_func=None, grid_size=1000):
    assert basis_func is not None, "Pass your basis function p(z) as `basis_func`"

    # Step 1: Solve scalar Riccati equation numerically
    def riccati_fixed_point(P_prev):
        denom = R + gamma * (B**2) * P_prev
        return Q + gamma * (A**2) * P_prev - (gamma**2) * (A**2) * (B**2) * (P_prev**2) / denom

    P = 1.0
    for _ in range(100):
        P = riccati_fixed_point(P)

    # Step 2: Define Q(x, u)
    def Q_true_fn(x, u):
        x_next = A * x + B * u
        return x**2 + u**2 + gamma * (x_next**2) * P

    # Step 3: Generate dataset and fit to basis
    x_vals = np.random.uniform(0, 10, size=grid_size)
    u_vals = np.random.uniform(0, 10, size=grid_size)
    Z = np.stack([x_vals, u_vals], axis=1)

    Φ = np.stack([basis_func(z) for z in Z], axis=0)
    Y = np.array([Q_true_fn(x, u) for x, u in Z])

    # Fit using least squares
    Q_vec, _, _, _ = np.linalg.lstsq(Φ, Y, rcond=None)
    Q_mat = np.outer(Q_vec, np.ones_like(Q_vec)).reshape((Φ.shape[1], Φ.shape[1]))  # rank-1

    return Q_mat, Q_true_fn

# Step 2: Generate synthetic dataset
N = 200
np.random.seed(2025)
X = np.random.uniform(0, 10, size=N)
U = np.random.uniform(0, 10, size=N)
X_plus = 0.9 * X + 0.1 * U  # toy dynamics: x+ = 0.9x + 0.1u
W = np.random.uniform(0, 10, size=N)  # freely chosen
L = X**2 + U**2  # toy cost function

gamma = 0.9    # discount factor
dim_p = len(p([X[0], U[0]]))  # = 6
dim_vecQ = dim_p ** 2  # = 36

# Step 3a: Solve for λ s.t. C(λ) ≽ 0
λ = cp.Variable(N, nonneg=True)
C_expr = sum([
    λ[i] * (outer_p([X[i], U[i]]) - gamma * outer_p([X_plus[i], W[i]]))
    for i in range(N)
])
print("C(λ) shape:", C_expr.shape)
# objective = cp.Maximize(cp.trace(C_expr))
epsilon = 1e-3
objective = cp.Maximize(cp.log_det(C_expr + epsilon * np.eye(dim_p)))   # Regularize C(λ)
constraints = [C_expr >> 0]
prob = cp.Problem(objective, constraints)
prob.solve()
print("Solver status:", prob.status)

if prob.status == cp.UNBOUNDED:
    raise ValueError("C(λ) is not bounded: try different data or basis.")
if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
    raise ValueError("C(λ) is not PSD: try different data or basis.")
else:
    print("λ found:", λ.value)

print("C(λ) is PSD. Proceeding to decomposition...")

C_val = C_expr.value
print("C_val:", C_val)
vec_C = C_val.reshape(-1)

# Step 3b: Approximate C(λ) = ∑ μ_i p(y_i)p(y_i)^T via NNLS
M = 200  # Start small
residual_threshold = 1e-3  # Define an acceptable residual threshold

print("vec_C shape:", vec_C.shape)

while True:
    x_vals = np.random.uniform(0, 10, size=M)
    u_vals = np.random.uniform(0, 10, size=M)
    Y = np.stack([x_vals, u_vals], axis=1)

    dict_matrix = np.stack([outer_p(y).reshape(-1) for y in Y], axis=1)  # shape (36, M)

    # Solve NNLS
    mu, residual = nnls(dict_matrix, vec_C)

    if residual < residual_threshold:
        print("Residual:", residual)
        break  # Stop if residual is acceptable

    M += 200  # Increment M
    if M > 10000:
        raise ValueError("Failed to find a suitable decomposition.")

print("Optimal M found:", M)

# Filter small μ_i
threshold = 1e-6
support_idx = np.where(mu > threshold)[0]
mu_filtered = mu[support_idx]
y_filtered = Y[support_idx]
print("y_filtered:", y_filtered)

print(f" Decomposed C(λ) using {len(mu_filtered)} atoms. Residual: {residual:.2e}")

# Step 4: Solve LP using decomposed surrogate for ∫ϕ(z)c(dz)
vec_C_approx = dict_matrix[:, support_idx] @ mu_filtered

α = cp.Variable(dim_vecQ)
constraints_lp = [
    α @ (outer_p([X[i], U[i]]).reshape(-1) - gamma * outer_p([X_plus[i], W[i]]).reshape(-1)) <= L[i]
    for i in range(N)
]

objective = cp.Maximize(α @ vec_C_approx)
lp = cp.Problem(objective, constraints_lp)
lp.solve(solver=cp.SCS, verbose=False)

print("Solved LP.")
print("Optimal value:", lp.value)

# Step 5: Reconstruct Q(z) = p(z)^T Q p(z)
Q_vec = α.value
Q_mat = Q_vec.reshape((dim_p, dim_p))

def Q_function(x, u):
    v = p([x, u])
    return v.T @ Q_mat @ v

# Optionally test the learned Q
x_test, u_test = 0.5, -0.3
print(f"Q({x_test}, {u_test}) =", Q_function(x_test, u_test))


# Step 6: Compare with analytical solution
Q_mat_true, Q_fn_true = compute_analytical_Q_matrix(basis_func=p)

# Compute Frobenius error
error = np.linalg.norm(Q_mat - Q_mat_true, ord='fro')
print("Frobenius error to ground truth Q:", error)

# Compare predictions
print("Q_true(0.5, -0.3) =", Q_fn_true(0.5, -0.3))
print("Q_learned(0.5, -0.3) =", Q_function(0.5, -0.3))
