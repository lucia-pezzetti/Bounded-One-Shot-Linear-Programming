import numpy as np
from scipy.linalg import solve_discrete_are
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import torch
from dynamical_systems import cart_pole
from sklearn.preprocessing import StandardScaler

# seed for reproducibility
np.random.seed(0)

# -------------------------
# 1. Problem & Data Setup
# -------------------------
# Cart-pole constants
M_c = 4.0     # cart mass (kg)
M_p = 2.0     # pole mass (kg)
l = 1.0       # half-pole length (m)
dt = 0.001    # sampling time (s)
gamma = 0.9999  # discount factor

# Dataset sizes
N = 1000      # number of samples
M = 500       # offline pool size
degree = 1    # polynomial feature degree

# Dimensions
dx = 4   # [x, x_dot, theta, theta_dot]
du = 1   # force

# State/action sampling bounds
x_bounds = (-3.0, 3.0)
x_dot_bounds = (-3.0, 3.0)
theta_bounds = (-1.0, 1.0)
theta_dot_bounds = (-1.0, 1.0)
u_bounds = (-10.0, 10.0)

system = cart_pole(M_c, M_p, l, dt, gamma, N, M)

scaler = StandardScaler()

# Generate data
x, u, x_plus, u_plus = system.generate_samples(x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=N)

z = np.concatenate([x, u], axis=1)          # shape (N, dx+du)
z_plus = np.concatenate([x_plus, u_plus], axis=1)  # shape (N, dx+du)

# ----------------------------------------------
# 2. Feature map p(z): all monomials up to 'degree'
# ----------------------------------------------
poly = PolynomialFeatures(degree, include_bias=False)

# Combine z and z_next to fit the PolynomialFeatures
Z_all = np.concatenate([z, z_plus], axis=0)       # shape (2N, dx+du)
P_all = poly.fit_transform(Z_all)                 # shape (2N, d)
P_all = scaler.fit_transform(P_all)             # scale features
d = P_all.shape[1]                                # feature dimension
print("P_all shape:", P_all.shape)

# Split back into P_z and P_z_next
P_z = P_all[:N]        # shape (N, d)
P_z_next = P_all[N:]   # shape (N, d)

x_aux, u_aux = system.generate_samples_auxiliary(x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=M)
y = np.concatenate([x_aux, u_aux], axis=1)  # shape (M, dx+du)

print("y shape:", y.shape)
P_y = poly.transform(y)   # shape (M, d)
P_y = scaler.transform(P_y)  # scale features
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

# Define Q_var ∈ ℝ^{d×d}
Q_var = cp.Variable((d, d), PSD=True)
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

print("Q_learned (feature‐space) matrix shape:", Q_learned_mat.shape)
print("eigenvalues of Q_learned:", np.linalg.eigvalsh(Q_learned_mat))

# ===========================================================
# 5. Policy Implementation and Simulation
# ===========================================================

def policy_learned(state, poly_features, scaler, Q_matrix, u_bounds, num_candidates=100):
    """
    Compute optimal control action for given state
    """
    s = np.asarray(state).flatten()
    
    def cost_fn(u_val):
        su = np.concatenate([s, [u_val]])  # shape (5,)
        phi = poly_features.transform(su.reshape(1, -1))  # shape (1, d)
        phi = scaler.transform(phi)  # scale features
        cost = phi @ Q_matrix @ phi.T
        return cost.item()  # Extract scalar value properly
    
    # Grid search for optimal u
    u_candidates = np.linspace(u_bounds[0], u_bounds[1], num_candidates)
    costs = [cost_fn(u) for u in u_candidates]
    best_idx = np.argmin(costs)
    
    return u_candidates[best_idx]

# ===========================================================
# 6. Simulation
# ===========================================================

# Generate initial state
x_bounds_init = (-1.0, 1.0)
x_dot_bounds_init = (-1.0, 1.0)
theta_bounds_init = (-0.5, 0.5)
theta_dot_bounds_init = (-0.5, 0.5)
x_current, _ = system.generate_samples_auxiliary(x_bounds_init, x_dot_bounds_init, theta_bounds_init, theta_dot_bounds_init, u_bounds, n_samples=1)
x_current = x_current[0]  # Take first (and only) sample

trajectory_x = []
trajectory_u = []

print("Starting simulation...")
max_steps = 100000

for step in range(max_steps):
    # Compute control action
    u_current = policy_learned(x_current, poly, scaler, Q_learned_mat, u_bounds)
    
    # Store trajectory
    trajectory_x.append(x_current.copy())
    trajectory_u.append(u_current)
    
    # Update state
    x_current = system.step(x_current, u_current)
    
    # Check for instability
    if np.any(np.abs(x_current) > 50):
        print(f"Simulation became unstable at step {step}")
        break
    
    if step % 200 == 0:
        print(f"Step {step}: state = {x_current}, control = {u_current:.3f}")

# Convert to arrays
trajectory_x = np.array(trajectory_x)
trajectory_u = np.array(trajectory_u)

print(f"Simulation completed with {len(trajectory_x)} steps")
print(f"Final state: {x_current}")
print(f"State ranges: x∈[{np.min(trajectory_x[:,0]):.2f}, {np.max(trajectory_x[:,0]):.2f}], "
      f"θ∈[{np.min(trajectory_x[:,2]):.2f}, {np.max(trajectory_x[:,2]):.2f}]")

# ===========================================================
# 7. Visualization
# ===========================================================

plt.figure(figsize=(12, 10))

# State trajectory
# plt.subplot(3, 1, 1)
plt.plot(trajectory_x[:, 0], label='Cart Position (x)', linewidth=2)
plt.plot(trajectory_x[:, 1], label='Cart Velocity (ẋ)', linewidth=2)
plt.plot(trajectory_x[:, 2], label='Pole Angle (θ)', color='green', linewidth=2)
plt.plot(trajectory_x[:, 3], label='Pole Angular Velocity (θ̇)', color='orange', linewidth=2)
# plt.ylabel('Position & Velocity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Cart-Pole Trajectory (Data-Driven Policy)')

# plt.subplot(3, 1, 2)
# plt.plot(trajectory_x[:, 2], label='Pole Angle (θ)', color='green', linewidth=2)
# plt.plot(trajectory_x[:, 3], label='Pole Angular Velocity (θ̇)', color='orange', linewidth=2)
# plt.ylabel('Angle & Angular Velocity')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.subplot(3, 1, 3)
# plt.plot(trajectory_u, label='Control Force (u)', color='red', linewidth=2)
# plt.xlabel('Time Step')
# plt.ylabel('Force (N)')
# plt.legend()
# plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/cart_pole_trajectory.png", dpi=150, bbox_inches='tight')
plt.show()

# Additional diagnostics
print("\n" + "="*50)
print("FINAL DIAGNOSTICS")
print("="*50)
print(f"Feature dimension: {d}")
print(f"Q matrix condition number: {np.linalg.cond(Q_learned_mat):.2e}")
print(f"Mean absolute control: {np.mean(np.abs(trajectory_u)):.3f}")
print(f"Control range: [{np.min(trajectory_u):.2f}, {np.max(trajectory_u):.2f}]")
print(f"Final cost: {system.cost(x_current.reshape(1,-1), np.array([[trajectory_u[-1]]]))[0]:.3f}")