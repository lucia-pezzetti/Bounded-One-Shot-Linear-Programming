'''
Determine λ and µ by directly matching the C(λ) and ∑ μ_i p(y_i)p(y_i)^T
Then compares the learned Q matrix with the true Q matrix obtained by solving analytically the LQR problem.

Still linear, but more general than the previous example.
A and B more high dimensional 3x3. Uniform sampling of the state and action space.

Making sure that the LP method is stable, i.e. if I choose A and B randomly but stabilizable and L positive semidefinite
and L_uu positive definite, the value of the discount factor gamma, the learned Q function should be close to the true Q function.
Plus with the number of samples N, the learned Q function should be close to the true Q function.

How to set constraints on λ and μ?
Make some plots to see if it is stable when changing the lower bounds.
'''

import numpy as np
import cvxpy as cp

# Step 1: Define basis functions
def p(z):
    x, u = z
    # return np.array([1, x, u, x**2, x*u, u**2])  # 6D basis
    return np.array([x, u])

def outer_p(z):
    v = p(z)
    return np.outer(v, v)

def solve_riccati(A, B, gamma, L, tol=1e-6, max_iter=1000):
    '''
    Solve the Riccati equation using fixed-point iteration.
    Alternatively one can use scipy's scipy.optimize.fsolve or another numerical root finding method.
    '''
    # Initialize P with a reasonable guess (e.g., L[0,0])
    P = L[0, 0]
    for _ in range(max_iter):
        # Compute the next value of P, if P is a matrix:
        # P_next = L[0, 0] + gamma * A**2 * P - (L[0, 1] + gamma * A * B) * np.linalg.inv(L[1, 1] + gamma * B**2 * P) * (L[1, 0] + gamma * B * P * A)

        # if P is not a matrix:
        denominator = L[1, 1] + gamma * B**2 * P
        if denominator == 0:
            raise ValueError("Denominator became zero during iteration.")
        P_next = L[0, 0] + gamma * A**2 * P - (L[0, 1] + gamma * A * B) * (1 / denominator) * (L[1, 0] + gamma * B * P * A)
        
        # Check for convergence
        if abs(P_next - P) < tol:
            return P_next  # Converged solution
        
        # Update P for the next iteration
        P = P_next
    
    raise ValueError("Riccati equation did not converge within the maximum number of iterations")

def compute_analytical_Q_matrix(A=0.9, B=0.1, gamma=0.9, L=np.eye(2), basis_func=None, grid_size=1000):
    assert basis_func is not None, "Pass your basis function p(z) as `basis_func`"

    # Step 1: Solve Riccati equation numerically
    P = solve_riccati(A, B, gamma, L)

    # Step 2: Define Q(x, u)
    def Q_true_fn(x, u):
        x_next = A * x + B * u
        return x**2 + u**2 + gamma * (x_next**2) * P    # Q(x, u) = [x u]^T Q^* [x u] = [x u]^T * L * [x u] + gamma * x_next^T P x_next
    
    # Step 3: Compute Q matrix
    Q_matrix = L + gamma * np.array([[A, B]]) * P * np.array([[A, B]]).T
    return Q_true_fn, Q_matrix


# Step 2: Generate synthetic dataset
N = 500
np.random.seed(2025)

# state space: x ∈ [-10, 10], control u ∈ [-1, 1]
# state space
Xmin = -10
Xmax = 10
# action space
Umin = -1
Umax = 1

# sample data
X = np.random.uniform(Xmin, Xmax, size=N)
U = np.random.uniform(Umin, Umax, size=N)
# dynamics: x_next = A * x + B * u
A = 0.9
B = 0.1
X_next = np.clip(A * X + B * U, Xmin, Xmax)
W = np.random.uniform(Umin, Umax, size=N)
# cost matrix -> identity
L = np.eye(2)
# cost function: [x u]^T * L * [x u] = x^2 + u^2
l = np.array([ [x, u] @ L @ [x, u] for x, u in zip(X, U) ])

# discount factor
gamma = 0.9

dim_p = len(p([X[0], U[0]]))  # = 6
dim_vecQ = dim_p ** 2  # = 36

# Step 3: Fix M and sample y_i
M = N
x_vals = np.random.uniform(Xmin, Xmax, size=M)
u_vals = np.random.uniform(Umin, Umax, size=M)
Y = np.stack([x_vals, u_vals], axis=1)

# Step 4: Define optimization variables
λ = cp.Variable(N, nonneg=True)
μ = cp.Variable(M, nonneg=True)

# Step 5: Construct C(λ)
C_lambda_expr = sum([
    λ[i] * (outer_p([X[i], U[i]]) - gamma * outer_p([X_next[i], W[i]]))
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
delta = 1
residual_matrix = C_lambda_expr - C_mu_expr
objective = cp.Minimize(cp.norm(residual_matrix, "fro"))
constraints = [λ >= 0, μ >= 0, cp.sum(λ) >= delta]      # Ensure λ is not all zeros

'''
How to properly set the constraints?
'''

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS, verbose=False)
print("sum(λ):", np.sum(λ.value))

# Step 8: Check results
""" print("Solver status:", problem.status)
print("||C(λ) - ∑ μ_i p(y_i)p(y_i)^T||_F =", cp.norm(residual_matrix, 'fro').value) """

""" print("λ values:", λ.value)
print("μ values:", μ.value) """

C_matrix = sum(μ.value[i] * outer_p(Y[i]) for i in range(M))
vec_C = C_matrix.reshape(-1)

α = cp.Variable(dim_vecQ)

constraints_lp = [
    α @ (outer_p([X[i], U[i]]).reshape(-1) - gamma * outer_p([X_next[i], W[i]]).reshape(-1)) <= l[i]
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


Q_fn_true, Q_matrix = compute_analytical_Q_matrix(basis_func=p)
print("Q_true(0.5, -0.3):", Q_fn_true(0.5, -0.3))

# compare the two Q matrices cause now they are the same size
# print("norm diff Q_learned and Q_true:", np.linalg.norm(Q_matrix - Q_mat))
""" print("Q_learned matrix:\n", Q_mat)
print("Q_true matrix:\n", Q_matrix) """

# comparisons
# MSE
x = np.linspace(Xmin, Xmax, 100)
u = np.linspace(Umin, Umax, 100)
# X_grid, U_grid = np.meshgrid(x, u)
Q_learned_grid = np.array([[Q_learned(xi, ui) for ui in u] for xi in x])
Q_true_grid = np.array([[Q_fn_true(xi, ui) for ui in u] for xi in x])
mse = np.sqrt(np.mean((Q_learned_grid - Q_true_grid) ** 2))
print("Mean Squared Error (MSE):", mse)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Learned Q function")
plt.contourf(x, u, Q_learned_grid, levels=50, cmap='viridis')
plt.colorbar(label='Q value')
plt.xlabel('x')
plt.ylabel('u')
plt.subplot(1, 2, 2)
plt.title("True Q function")
plt.contourf(x, u, Q_true_grid, levels=50, cmap='viridis')
plt.colorbar(label='Q value')
plt.xlabel('x')
plt.ylabel('u')
plt.tight_layout()
plt.show()
# The learned Q function should closely match the true Q function.
# The MSE should be small if the learned Q function is accurate.
# The contour plots should show similar shapes for the learned and true Q functions.

# max absolute error
max_error = np.max(np.abs(Q_learned_grid - Q_true_grid))
print("Max absolute error:", max_error)

max_error_index = np.unravel_index(np.argmax(np.abs(Q_learned_grid - Q_true_grid)), Q_learned_grid.shape)
print("Max error index:", max_error_index)
print("Max error x:", x[max_error_index[0]])
print("Max error u:", u[max_error_index[1]])

# max and min values of Q_learned and Q_true
print("Max Q_learned:", np.max(Q_learned_grid))
print("Min Q_learned:", np.min(Q_learned_grid))
print("Max Q_true:", np.max(Q_true_grid))
print("Min Q_true:", np.min(Q_true_grid))

# errors accumulates at the edges of the state and action space. Why?