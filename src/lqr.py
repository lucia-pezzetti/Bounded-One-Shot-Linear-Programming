import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
from dynamical_systems import dlqr

parser = argparse.ArgumentParser(description='Solving LQR with LP approach.')

parser.add_argument('--degree', type=int, default=1, help='Polynomial feature degree')
parser.add_argument('--rho', type=float, default=0.1, help='Control cost weight')
parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples')
parser.add_argument('--m_samples', type=int, default=5000, help='Auxiliary samples')
parser.add_argument('--x_bounds', type=float, nargs=2, default=(-3.0, 3.0), help='State bounds (x0, x1)')
parser.add_argument('--u_bounds', type=float, nargs=2, default=(-1.0, 1.0), help='Action bounds (force)')   
parser.add_argument('--n_x', type=int, default=6, help='State dimension')
parser.add_argument('--n_u', type=int, default=2, help='Action dimension')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--data_dir', type=str, default='../data', help='Directory to load data')

args = parser.parse_args()

# seed for reproducibility
np.random.seed(args.random_seed)

# -------------------------
# 1. Problem & Data Setup
# -------------------------
# Cart-pole constants
rho = args.rho    # control cost weight
gamma = args.gamma  # discount factor

# Dataset sizes
N = args.n_samples      # number of samples
M = args.m_samples       # auxiliary samples
degree = 1    # polynomial feature degree

# Dimensions
dx = args.n_x   # [x0, x1]
du = args.n_u   # force

# State/action sampling bounds
x_bounds = args.x_bounds
u_bounds = args.u_bounds

# Initialize the dlqr system
if args.data_dir != 'None':
    with open(os.path.join(args.data_dir, f'dx_{args.n_x}.json'), 'r') as f:
        res = json.load(f)
        A = np.array(res['A'], dtype=float)
        B = np.array(res['B'], dtype=float)
        C = np.array(res['C'], dtype=float)
else:
    A = np.array([[1.0, 0.1], [0.5, -0.5]])
    B = np.array([[1.0], [0.5]])
    C = np.eye(dx)

system = dlqr(A, B, C, rho, gamma)

# Generate data
x, u, x_plus, u_plus = system.generate_samples(x_bounds, u_bounds, n_samples=N)
print("x shape:", x.shape)
print("u shape:", u.shape) 
print("x_plus shape:", x_plus.shape)
print("u_plus shape:", u_plus.shape)

z = np.concatenate([x, u], axis=1)          # shape (N, dx+du)
z_plus = np.concatenate([x_plus, u_plus], axis=1)  # shape (N, dx+du)

# ----------------------------------------------
# 2. Feature map p(z): all monomials up to 'degree'
# ----------------------------------------------
poly = PolynomialFeatures(degree, include_bias=False)

# Combine z and z_next to fit the PolynomialFeatures
Z_all = np.concatenate([z, z_plus], axis=0)       # shape (2N, dx+du)
P_all = poly.fit_transform(Z_all)                 # shape (2N, d)
d = P_all.shape[1]                                # feature dimension
print("P_all shape:", P_all.shape)

# Split back into P_z and P_z_next
P_z = P_all[:N]        # shape (N, d)
P_z_next = P_all[N:]   # shape (N, d)

x_aux, u_aux = system.generate_samples_auxiliary(x_bounds, u_bounds, n_samples=M)
y = np.concatenate([x_aux, u_aux], axis=1)  # shape (M, dx+du)

print("y shape:", y.shape)
P_y = poly.transform(y)   # shape (M, d)
print("P_y shape:", P_y.shape)

# ===============================================
# 3. First LP: Find λ, μ for moment matching
# ===============================================

# Create CVXPY variables
lambda_var = cp.Variable(N, nonneg=True)  # λ ∈ ℝ^N₊
mu_var = cp.Variable(M, nonneg=True)      # μ ∈ ℝ^M₊

# Convert to CVXPY Constants
Pz_const = cp.Constant(P_z)
Pz_next_const = cp.Constant(P_z_next)
Py_const = cp.Constant(P_y)

# Build the moment-match matrix
sum_PzPzT = Pz_const.T @ cp.diag(lambda_var) @ Pz_const
sum_PznPznT = Pz_next_const.T @ cp.diag(lambda_var) @ Pz_next_const
sum_PyPyT = Py_const.T @ cp.diag(mu_var) @ Py_const

moment_match = sum_PzPzT - gamma * sum_PznPznT - sum_PyPyT

constraints = []
# Enforce moment_match == 0_(d×d)
constraints += [moment_match == np.zeros((d, d))]

# Break scale-invariance:
constraints += [cp.sum(lambda_var) == 1]

# Build C_approx
C_approx = sum_PyPyT
print("C_approx shape:", C_approx.shape)

# Objective: minimize ‖C_approx − I‖_F
I_d = np.eye(d)
objective = cp.Minimize(cp.norm(C_approx - I_d, "fro"))

# Solve LP #1
prob = cp.Problem(objective, constraints)
try:
    prob.solve(solver=cp.MOSEK, verbose=False)
    print("Status (LP for λ,μ):", prob.status)
except:
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        print("Status (LP for λ,μ) with ECOS:", prob.status)
    except:
        prob.solve(verbose=False)
        print("Status (LP for λ,μ) with default solver:", prob.status)

if prob.status not in ["optimal", "optimal_inaccurate"]:
    print("First LP failed. Status:", prob.status)
    exit()

lambda_vals = lambda_var.value
mu_vals = mu_var.value

# Diagnostics
C_num = P_y.T @ np.diag(mu_vals) @ P_y
print("||C_approx - I||_F:", np.linalg.norm(C_num - I_d, "fro"))
print("Sum(λ):", np.sum(lambda_vals))
print("Nonzero μ count:", np.sum(mu_vals > 1e-6))

# Extract the final λ, μ
lam = lambda_var.value
mu = mu_var.value

# ===========================================================
# 4. Second LP: Learn Q matrix
# ===========================================================

C_val = P_y.T @ np.diag(mu) @ P_y
c_vec = C_val.flatten(order="C")
print("c_vec shape:", c_vec.shape)

# Define Q_var ∈ ℝ^{d×d}
Q_var = cp.Variable((d, d))
Q_var_vec = cp.reshape(Q_var, (d * d,), order="C")

# Compute quadratic stage costs
L_xu = system.cost(x, u)
L_xu_const = cp.Constant(L_xu)

# Build constraint matrix efficiently
F_3D = (P_z[:, :, None] * P_z[:, None, :]) - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])
F_mat = F_3D.reshape((N, d * d), order="C")

# Constraint: F_mat @ Q_var_vec <= L_xu
constraints_lp = [F_mat @ Q_var_vec <= L_xu_const]

# Objective: maximize c_vecᵀ · Q_var_vec
objective_lp = cp.Maximize(c_vec @ Q_var_vec)

# Solve LP #2
prob_lp = cp.Problem(objective_lp, constraints_lp)
try:
    prob_lp.solve(solver=cp.MOSEK, verbose=False)
    print("Status (LP for Q):", prob_lp.status)
except:
    try:
        prob_lp.solve(solver=cp.ECOS, verbose=False)
        print("Status (LP for Q) with ECOS:", prob_lp.status)
    except:
        prob_lp.solve(verbose=False)
        print("Status (LP for Q) with default solver:", prob_lp.status)

if prob_lp.status not in ["optimal", "optimal_inaccurate"]:
    print("Second LP failed. Status:", prob_lp.status)
    exit()

Q_learned_vec = Q_var_vec.value
Q_learned_mat = Q_learned_vec.reshape((d, d), order="C")

E_Q_learned = prob_lp.value
print("Learned E_Q:", E_Q_learned)

print("Q_learned (feature‐space) matrix shape:", Q_learned_mat.shape)
print("eigenvalues of Q_learned:", np.linalg.eigvalsh(Q_learned_mat))
print("c:", c_vec)

# ===========================================================
# 5. Policy Implementation and Simulation
# ===========================================================

def policy_learned(state, poly_features, Q_matrix, u_bounds, num_candidates=50):
    """
    Compute optimal control action for given state
    """
    s = np.asarray(state).flatten()
    
    def cost_fn(u_val):
        su = np.concatenate([s, [u_val]])  # shape (5,)
        phi = poly_features.transform(su.reshape(1, -1))  # shape (1, d)
        cost = phi @ Q_matrix @ phi.T
        return cost.item()  # Extract scalar value properly
    
    # Grid search for optimal u
    u_candidates = np.linspace(u_bounds[0], u_bounds[1], num_candidates)
    costs = [cost_fn(u) for u in u_candidates]
    best_idx = np.argmin(costs)
    
    return u_candidates[best_idx]

# ===========================================================
# 6. Compare Q learned with optimal Q since we are in LQR
# ===========================================================
P, K, q = system.optimal_solution()

# xu_weights = np.concatenate([np.ones(dx), np.ones(du)], axis=0)
Q_star, E_Qstar, gap = system.optimal_q(P, q, c_vec)

print("Optimal Q matrix shape:", Q_star.shape)
print("E_Qstar:", E_Qstar)
print("Gap:", gap)

print("eigenvalues of optimal Q:", np.linalg.eigvalsh(Q_star))

print("Optimality gap (E_Qstar - E_Q_learned):", np.abs(E_Qstar - E_Q_learned))

# evaluate trace difference of the learned Q and the optimal Q
trace_diff = np.trace(Q_learned_mat) - np.trace(Q_star)
print("Trace of learned Q:", np.trace(Q_learned_mat))
print("Trace of optimal Q:", np.trace(Q_star))
print("Trace difference (learned Q - optimal Q):", trace_diff)
