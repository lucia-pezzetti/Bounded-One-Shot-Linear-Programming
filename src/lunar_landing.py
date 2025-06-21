import numpy as np
from scipy.linalg import solve_discrete_are
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# -------------------------
# 1. Problem & Data Setup
# -------------------------
# Constants
g = 1.62      # lunar gravity (m/s^2)
k = 0.01       # fuel burn rate constant
dt = 0.1      # time step (s)
N = 250        # number of steps to simulate
M = 250       # size of offline pool
degree = 1
dx = 3     # state dimension (height, velocity, mass)
du = 1     # input dimension (control force)
gamma = 0.99  # discount factor

np.random.seed(0)  # for reproducibility
# Dynamics function: compute next state and sample new control
def lunar_landing_step(x, u, g=1.62, k=0.1, dt=0.1):
    h, v, m = x
    if m <= 0:
        m = 0.01  # Avoid division by zero

    # State updates (Euler method)
    h_next = h + v * dt
    v_next = v + (-g + u / m) * dt
    m_next = m - k * u * dt
    m_next = max(m_next, 0)  # Prevent negative mass

    x_next = (h_next, v_next, m_next)

    # Sample new control w in [0, 1]
    w = np.random.uniform(0, 1)
    
    return x_next, w

# Sampling bounds
h0, v0, m0 = 10, -10, 0.2  # initial state (height, velocity, mass)      10, -10, 10
h_bounds = (0, h0)    # height in meters                0, 10
v_bounds = (-10, 0)    # velocity (descending)          -10, 0
m_bounds = (0, 0.2)   # mass in kg                0, 10

# Generate N samples
def generate_samples(N):
    samples = []
    for _ in range(N):
        h = np.random.uniform(*h_bounds)
        v = np.random.uniform(*v_bounds)
        m = np.random.uniform(*m_bounds)
        u = np.random.uniform(0, 1)

        x = (h, v, m)
        x_next, w = lunar_landing_step(x, u, g, k, dt)
        row = [*x, u, *x_next, w]
        samples.append(row)

    return np.array(samples)

samples = generate_samples(N)
print("samples shape:", samples.shape)  # should be (N, 2*dx + du + du)
print("First 5 samples:\n", samples[:5])

# Extract x = [h, v, m], u, x_next = [h', v', m'], w from data
x       = samples[:, :dx]   # h, v, m
u       = samples[:, dx:(dx+du)]  # u (keep 2D for hstack)
x_next  = samples[:, (dx+du):(2*dx+du)]  # h', v', m'
w       = samples[:, (2*dx+du):2*(dx+du)]  # w (keep 2D)

# Concatenate to form z and z_next
z      = np.hstack([x, u])      # shape (N, 4)
print("z shape:", z.shape)  # should be (N, 4)
z_next = np.hstack([x_next, w]) # shape (N, 4)
print("z_next shape:", z_next.shape)  # should be (N, 4)

# ----------------------------------------------
# 2. Feature map p(z): all monomials up to ‘degree’
# ----------------------------------------------
poly = PolynomialFeatures(degree, include_bias=False)

    # Combine z and z_next to fit the PolynomialFeatures
Z_all = np.vstack([z, z_next])         # shape (2N, dx+du)
P_all = poly.fit_transform(Z_all)      # shape (2N, d)
d = P_all.shape[1]                     # feature dimension
print("P_all shape:", P_all.shape)  # should be (2N, d)

# Split back into P_z and P_z_next
P_z = P_all[:N]        # shape (N, d)
P_z_next = P_all[N:]   # shape (N, d)

# Build P_y for the offline pool y
h_y = np.random.uniform(*h_bounds, size=(M,1))
v_y = np.random.uniform(*v_bounds, size=(M,1))
m_y = np.random.uniform(*m_bounds, size=(M,1))
u_y = np.random.uniform(0.0, 1.0,  size=(M,1))
# concatenate into y of shape (M, dx+du)
y = np.hstack([h_y, v_y, m_y, u_y])      # (M,4)

print("y shape:", y.shape)  # should be (M, 4)
print("First 5 y samples:\n", y[:5])

P_y = poly.transform(y)   # shape (M, d)


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

Q = Q_learned_mat
# state–action ordering is [h, v, m, u]
Q_xu = Q[:3, 3]    # shape (3,)
Q_uu = Q[3, 3]     # scalar

# --- 2) Define the arg-min control rule from your Q
def policy_learned(state):
    """
    state: array-like [h, v, m]
    returns u in [0,1] which minimizes φ(s,u)^T Q φ(s,u),
    where φ = [h, v, m, u].
    """
    h, v, m = state
    # gradient wrt u: 2 x^T Q_xu + 2 u Q_uu -> set to zero
    u_star = - (np.dot([h, v, m], Q_xu)) / Q_uu
    return float(np.clip(u_star, 0.0, 1.0))

traj = []
u_learn = []

# initial conditions (must match the ones you used above)
h, v, m = h0, v0, m0
traj.append((h, v, m))

for i in range(int(10 / dt)):   # max 10 s, will break on touchdown
    u_t = policy_learned((h, v, m))
    u_learn.append(u_t)
    # discrete dynamics
    h = h + v * dt
    v = v + (-g + u_t / m) * dt
    m = max(m - k * u_t * dt, 0)
    traj.append((h, v, m))
    if h <= 0:
        break

u_learn = np.array(u_learn)
t = np.arange(len(u_learn)) * dt

# --- 4) Build a pure bang–bang trajectory by switching where your policy first exceeds 0.5
u_bang = np.zeros_like(u_learn)
switch_pts = np.where(u_learn > 0.5)[0]
if switch_pts.size > 0:
    s = switch_pts[0]
    u_bang[s:] = 1.0
    t_switch = s * dt
else:
    t_switch = None

# --- 5) Plot both controls
plt.figure(figsize=(8,3))
plt.step(t, u_learn, where='post', label='Learned policy')
plt.step(t, u_bang, where='post', linestyle='--', label='Ideal bang–bang')
if t_switch is not None:
    plt.axvline(t_switch, color='gray', linestyle=':', label=f'switch @ {t_switch:.2f}s')
plt.xlabel('Time (s)')
plt.ylabel('u (thrust fraction)')
plt.title('Learned vs. Bang–Bang Control')
plt.legend()
plt.tight_layout()
plt.savefig("lunar_landing_control.png")

# --- 6) Quick saturation check
frac_zero = np.mean(u_learn < 1e-3)
frac_one  = np.mean(u_learn > 1-1e-3)
print(f"Learned policy saturates to 0 on {frac_zero:.1%} of steps, to 1 on {frac_one:.1%} of steps.")



# Very high dependence on the number of samples N and M
# increasing the degree leads to infeasible LPs
# Very high dependence on the choice of the initial conditions and bounds over the state and control
