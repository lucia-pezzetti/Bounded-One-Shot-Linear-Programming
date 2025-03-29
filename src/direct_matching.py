'''
Determine λ and µ by directly matching the C(λ) and ∑ μ_i p(y_i)p(y_i)^T
Then compares the learned Q matrix with the true Q matrix obtained by solving analytically the LQR problem.
'''

import numpy as np
import cvxpy as cp

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
X_plus = 0.9 * X + 0.1 * U
W = np.random.uniform(0, 10, size=N)
L = X**2 + U**2
gamma = 0.9

dim_p = len(p([X[0], U[0]]))  # = 6
dim_vecQ = dim_p ** 2  # = 36

# Step 3: Fix M and sample y_i
M = 200
x_vals = np.random.uniform(0, 10, size=M)
u_vals = np.random.uniform(0, 10, size=M)
Y = np.stack([x_vals, u_vals], axis=1)

# Step 4: Define optimization variables
λ = cp.Variable(N, nonneg=True)
μ = cp.Variable(M, nonneg=True)

# Step 5: Construct C(λ)
C_lambda_expr = sum([
    λ[i] * (outer_p([X[i], U[i]]) - gamma * outer_p([X_plus[i], W[i]]))
    for i in range(N)
])

# Step 6: Construct the RHS matrix from μ and y_i
dict_tensors = np.stack([outer_p(y) for y in Y], axis=2)  # shape: (6, 6, M)
C_mu_expr = sum([
    μ[i] * dict_tensors[:, :, i]
    for i in range(M)
])

# Step 7: Define the optimization problem
# Match C(λ) and ∑ μ_i p(y_i)p(y_i)^T
delta = 1e-3
residual_matrix = C_lambda_expr - C_mu_expr
objective = cp.Minimize(cp.norm(residual_matrix, "fro"))
constraints = [λ >= 0, μ >= 0, cp.sum(λ) >= delta]      # Ensure λ is not all zeros

'''
How to properly set the constraints?
'''

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS, verbose=False)

# Step 8: Check results
""" print("Solver status:", problem.status)
print("||C(λ) - ∑ μ_i p(y_i)p(y_i)^T||_F =", cp.norm(residual_matrix, 'fro').value) """

""" print("λ values:", λ.value)
print("μ values:", μ.value) """

C_matrix = sum(μ.value[i] * outer_p(Y[i]) for i in range(M))
vec_C = C_matrix.reshape(-1)

α = cp.Variable(dim_vecQ)

constraints_lp = [
    α @ (outer_p([X[i], U[i]]).reshape(-1) - gamma * outer_p([X_plus[i], W[i]]).reshape(-1)) <= L[i]
    for i in range(N)
]

objective = cp.Maximize(α @ vec_C)
lp = cp.Problem(objective, constraints_lp)
lp.solve(solver=cp.SCS)

Q_vec = α.value
Q_mat = Q_vec.reshape((dim_p, dim_p))

def Q_learned(x, u):
    v = p([x, u])
    return v.T @ Q_mat @ v

print("Q_learned(0.5, -0.3):", Q_learned(0.5, -0.3))


Q_mat_true, Q_fn_true = compute_analytical_Q_matrix(basis_func=p)
print("Q_true(0.5, -0.3):", Q_fn_true(0.5, -0.3))

'''
The results seem to NOT match.
There are inconsistencies in the learned Q matrix... the results decrease quite drastically augmenting the number of datapoints, but even with 10000 points, the learned Q matrix is not close to the true Q matrix: 

Q_true(0.5, -0.3): 0.8823175325593224

200 points
Q_learned(0.5, -0.3): 32.1076866491031

1000 points
Q_learned(0.5, -0.3): 19.26375279613726

5000 points
Q_learned(0.5, -0.3): 12.599018795741038

10000 points:
Q_learned(0.5, -0.3): 2.855140177095811
'''