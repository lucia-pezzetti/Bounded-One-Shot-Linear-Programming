'''
This script demonstrates the implementation of the proposed method on a synthetic 1D dataset.
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

# Step 2: Generate synthetic dataset
N = 20
np.random.seed(2025)
X = np.random.uniform(0, 10, size=N)
U = np.random.uniform(0, 10, size=N)
X_plus = X + 0.1 * U  # toy dynamics: x+ = x + 0.1u
W = np.random.uniform(0, 10, size=N)  # freely chosen
L = X**2 + U**2  # toy cost function

gamma = 0    # discount factor
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
objective = cp.Maximize(cp.log_det(C_expr + epsilon * np.eye(dim_p)))
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
lp.solve()

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
