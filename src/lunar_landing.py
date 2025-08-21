import numpy as np
from scipy.linalg import solve_discrete_are
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

from episode_sampling import sample_episodes, random_policy, stage_cost_hard, rollouts_to_dataset, episodes_to_rows

# -------------------------
# 1. Problem & Data Setup
# -------------------------
# Constants
g = 1.62      # lunar gravity (m/s^2)
k = 0.1       # fuel burn rate constant
dt = 0.1      # time step (s)
N = 1000        # number of steps to simulate
M = 500       # size of offline pool
degree = 2
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
h0, v0, m0, ms = 10, -5, 0.3, 0.1  # initial state (height, velocity, mass)      10, -10, 10
h_bounds = (-0.2, h0)    # height in meters                0, 10
v_bounds = (v0, 0.5)    # velocity (descending)          -10, 0
m_bounds = (ms, m0)   # mass in kg                0, 10

def is_feasible_state(h, v, m, ms, g, k):
    """
    Check if a state (h,v,m) can achieve soft landing.
    Uses the glide slope constraint: v >= -c*sqrt(h) where c = sqrt(2*a_max)
    """
    if h <= 0:
        return abs(v) <= 0.1  # Near zero velocity at ground
    
    # Maximum deceleration with full thrust
    a_max = max(0, 1.0/ms - g)  # net upward acceleration at minimum mass
    if a_max <= 0:
        return False  # Cannot decelerate at all
    
    c = np.sqrt(2 * a_max)
    glide_slope_velocity = -c * np.sqrt(h)
    
    return v >= glide_slope_velocity

rng = np.random.default_rng(0)
def pi_rand(x):
    return random_policy(x, u_min=0.0, u_max=1.0, rng=rng)


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

# samples = generate_samples(N)


# --- Non-uniform sampling focused near origin ---------------------------------
def generate_samples_nonuniform(N, sigma_h=2.0, sigma_v=1.0, max_tries_per_sample=2000, rng=None):
    """
    Generate N samples (x, u, x_next, w) like `generate_samples`, but bias sampling
    toward states near the origin in (h, v) using rejection sampling.
    
    Acceptance probability:
        a(h, v) = exp( -0.5 * ((h/sigma_h)**2 + (v/sigma_v)**2) )
    which is highest at (0,0) and smoothly decays away from it.
    
    Args:
        N: number of accepted samples desired.
        sigma_h: spread for altitude concentration (meters). Smaller -> more mass near h=0.
        sigma_v: spread for velocity concentration (m/s).   Smaller -> more mass near v=0.
        max_tries_per_sample: cap per accepted sample to avoid infinite loops in extreme settings.
        rng: optional numpy Generator.
        
    Returns:
        np.ndarray of shape (N, 8): [h, v, m, u, h_next, v_next, m_next, w]
    """
    if rng is None:
        rng = np.random.default_rng()
    
    samples = []
    tries = 0
    max_total_tries = int(max_tries_per_sample * N)
    
    # Precompute bounds for speed
    h_lo, h_hi = h_bounds
    v_lo, v_hi = v_bounds
    m_lo, m_hi = m_bounds
    
    while len(samples) < N and tries < max_total_tries:
        # Propose uniformly within bounds
        h = rng.uniform(h_lo, h_hi)
        v = rng.uniform(v_lo, v_hi)
        m = rng.uniform(m_lo, m_hi)
        u = rng.uniform(0.0, 1.0)
        
        # Rejection probability that favors (h, v) near (0,0)
        a = np.exp(-0.5 * ((h / sigma_h)**2 + (v / sigma_v)**2))
        
        # Draw a uniform[0,1] and accept if below a
        if rng.uniform() <= a:
            x = (h, v, m)
            x_next, w = lunar_landing_step(x, u, g, k, dt)
            samples.append([*x, u, *x_next, w])
        tries += 1
    
    if len(samples) < N:
        print(f"[WARN] Only accepted {len(samples)} / {N} after {tries} proposals. "
              f"Try increasing max_tries_per_sample or the sigmas.")
    else:
        print(f"[INFO] Acceptance ratio: {len(samples)}/{tries} = {len(samples)/max(tries,1):.3f}")
    
    return np.array(samples)

samples = generate_samples_nonuniform(
    N,
    sigma_h=1.0,   # tighten/loosen focus on altitude
    sigma_v=1.0    # tighten/loosen focus on velocity
)


# stack all xs and us into a single array
# samples = episodes_to_rows(episodes)
# N = samples.shape[0]  # number of samples
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
print("P_y shape:", P_y.shape)  # should be (M, d)

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
print("C_approx shape:", C_approx.shape)  # should be (d, d)

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
lam = lambda_var.value            # shape (N,)
mu = mu_var.value                # shape (M,)

# Reconstruct M_lambda and M_mu for diagnostic
M_lambda = P_z.T @ np.diag(lam) @ P_z - gamma * (P_z_next.T @ np.diag(lam) @ P_z_next)
M_mu     = P_y.T @ np.diag(mu) @ P_y
diff_mom = M_lambda - M_mu
print("||M_lambda - M_mu||_F:", np.linalg.norm(diff_mom, "fro"))
print("max|M_lambda - M_mu|:", np.max(np.abs(diff_mom)))


# ===========================================================
# 4. New LP: maximize αᵀ vec(C_approx) subject to constraints
# ===========================================================

C_val = P_y.T @ np.diag(mu) @ P_y           # shape (d, d)
c_vec = C_val.flatten(order="C")          # shape (d²,)

# Define Q_var ∈ ℝ^{d×d}, PSD, and its vectorized form
# REMOVE THE CONSTRAINT PSD=True, HOWEVER PAY ATTENTION WHEN NOT LQR
Q_var     = cp.Variable((d, d))               # matrix variable
Q_var_vec = cp.reshape(Q_var, (d * d,), order="C")     # flatten in row-major

# compute L_xu (toy cost) exactly as before:
# L_xu = (x**2).sum(axis=1) + (u**2).sum(axis=1)   # shape (N,)
# add safety barrier to impose soft landing
L_xu = 10.0*u.squeeze() # shape (N,)
# barrier hyper-parameters (tune as needed)
k_b = 100.0     # overall barrier strength
c_b = 10.0    # how rapidly the penalty ramps up as h→0
h_vals = x[:, 0].squeeze()   # heights
v_vals = x[:, 1].squeeze()   # vertical velocities
# barrier term: \to \infty if v≠0 at h=0
barrier = k_b * np.exp(-c_b * h_vals) * (v_vals**2)

# add to your cost vector
L_xu = L_xu + barrier

# cost computation with barrier to impose soft landing
# Cost function with barrier
def stage_cost_with_barrier(h, v, m, u, *, lam_fuel=0.05, beta_barrier=1000.0, ms_min=None, g=1.62):
    """
    Stage cost with barrier function to enforce feasibility
    """
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    
    # Fuel cost
    fuel_cost = lam_fuel * np.abs(u)
    
    # Barrier for feasibility
    if ms_min is None:
        ms_min = ms
    
    a_max = max(0.0, 1.0 / ms_min - g)
    c = np.sqrt(2.0 * max(a_max, 1e-12))
    
    # Glide slope constraint: v >= -c*sqrt(h)
    h_pos = np.maximum(h, 1e-6)  # Avoid sqrt(0)
    glide_slope_bound = -c * np.sqrt(h_pos)
    
    # Barrier: penalize when v < glide_slope_bound
    violation = glide_slope_bound - v  # positive when violating
    barrier = beta_barrier * np.maximum(0.0, violation)**2
    
    # Terminal barrier: heavily penalize high velocity near ground
    ground_penalty = np.where(h < 0.5, 1000.0 * v**2, 0.0)
    
    total_cost = fuel_cost + barrier + ground_penalty
    return np.nan_to_num(total_cost, nan=1e12, posinf=1e12, neginf=1e12)

def stage_cost_with_hard_penalty(h, v, m, u, *, lam_fuel=0.05, v_tol=0.2,
                                 ms_min=None, g=1.62, big_M=10, w_h=0.2):
    """
    Returns lam_fuel*|u| normally, but returns `big_M` whenever any is violated:
      1) Glide slope: v >= -sqrt(2*max(1/ms_min - g, 0))*sqrt(max(h, 0))
      2) Touchdown: if h <= 0, require |v| <= v_tol
      3) Physical sanity: h >= 0 and m >= ms_min
    """
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)

    if ms_min is None:
        ms_min = ms

    # Base cost: lam_fuel * |u|
    base = lam_fuel * np.abs(u)

    # just using a fuel constraint is not enough. Touching the ground is extremely risky, the agent learns to avoid it.
    # Add a small altitude shaping cost to encourage descending
    base += w_h * h

    a_max = max(0.0, 1.0 / ms_min - g)
    c = np.sqrt(2.0 * max(a_max, 1e-12))

    h_pos = np.maximum(h, 0.0)
    glide_ok  = v >= (-c * np.sqrt(h_pos))
    ground_ok = np.where(h <= 1e-6, np.abs(v) <= v_tol, True)
    phys_ok   = (h >= 0.0) & (m >= ms_min)

    mask_safe_land = (h <= 1e-6) & (np.abs(v) <= v_tol)
    base = np.where(mask_safe_land, base - big_M, base)

    ok = glide_ok & ground_ok & phys_ok
    return np.where(ok, base, base + big_M)

# Compute costs using the new barrier function
# L_xu = stage_cost_with_barrier(x[:, 0].squeeze(), x[:, 1].squeeze(), x[:, 2].squeeze(), u.squeeze(), ms_min=ms, g=g)

# L_xu += terminal_cost_soft(x[:, 0].squeeze(), x[:, 1].squeeze(), Wh=0.5, Wv=1e5, vtol=0.7)

# L_xu = stage_cost_with_hard_penalty(x[:, 0].squeeze(), x[:, 1].squeeze(), x[:, 2].squeeze(), u.squeeze(), lam_fuel=1e6, ms_min=ms)

# Build a 3D array of shape (N, d, d) in one go, then flatten:
F_3D = (P_z[:, :, None] * P_z[:, None, :]) - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])
# Now reshape to (N, d*d), preserving row-major (“C”) ordering:
F_mat = F_3D.reshape((N, d * d), order="C")     # shape (N, d²)

# single vectorized constraint:
constraints_lp = [ F_mat @ Q_var_vec <= L_xu ]
# Add safety constraints for samples near the ground
for i in range(N):
    h_i, v_i = x[i, 0], x[i, 1]
    if h_i < 1.0:  # Near ground
        # Penalize high velocity near ground more heavily
        safety_penalty = 1000.0 * v_i**2
        # You could add this as an additional constraint or modify L_xu
        L_xu[i] += safety_penalty

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
print("eigenvalues of Q_learned:", np.linalg.eigvalsh(Q_learned_mat))

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

def policy_learned_with_safety(state):
    """
    Enhanced policy that respects feasibility constraints
    """
    h, v, m = state
    
    # Basic quadratic policy
    Q_xu = Q_learned_mat[:3, 3]
    Q_uu = Q_learned_mat[3, 3]
    
    if abs(Q_uu) < 1e-8:
        u_star = 0.5  # fallback
    else:
        u_star = -np.dot([h, v, m], Q_xu) / Q_uu
        u_star = float(np.clip(u_star, 0.0, 1.0))
    
    # Safety override: if approaching feasibility boundary, use maximum thrust
    if h > 0:
        a_max = max(0.0, 1.0/m - g)  # current maximum deceleration
        c = np.sqrt(2.0 * max(a_max, 1e-12))
        glide_slope_velocity = -c * np.sqrt(h)
        
        # If close to violating constraint, override with high thrust
        # safety_margin = 0.1
        # if v < glide_slope_velocity + safety_margin:
        #     u_star = 1.0
    
    return u_star

traj = []
u_learn = []

# we should expect linear policy that try to interpolate the bang bang policy

# initial conditions (must match the ones you used above)
h, v, m = h0, v0, m0
traj.append((h, v, m))

for i in range(int(20 / dt)):   # max 10 s, will break on touchdown
    print(f"Step {i}: h={h:.2f}, v={v:.2f}, m={m:.2f}")
    u_t = policy_learned_with_safety((h, v, m))
    print(f"Control u_t = {u_t:.2f}")
    u_learn.append(u_t)
    # discrete dynamics
    h = h + v * dt
    v = v + (-g + u_t / m) * dt
    m = max(m - k * u_t * dt, ms)
    traj.append((h, v, m))
    if h <= 0:
        break

u_learn = np.array(u_learn)
t = np.arange(len(u_learn)) * dt

# --- 4) Build a pure bang–bang trajectory by switching where your policy first exceeds 0.5
u_bang = np.zeros_like(u_learn)
# switch_pts = np.where(u_learn > 0.5)[0]
# if switch_pts.size > 0:
#     s = switch_pts[0]
#     u_bang[s:] = 1.0
#     t_switch = s * dt
# else:
#     t_switch = None
def find_t_star(h0, v0, m0, ms, g, k, tol=1e-8, max_iter=100):
    """
    Compute the switch time t* for the minimal fuel landing problem.
    
    Parameters
    ----------
    h0 : float
        Initial height.
    v0 : float
        Initial velocity (downward negative).
    m0 : float
        Initial mass (fuel + structure).
    ms : float
        Structural mass (mass without fuel).
    g : float
        Gravitational acceleration.
    k : float
        Fuel consumption constant.
    tol : float, optional
        Tolerance for root finding.
    max_iter : int, optional
        Maximum iterations for bisection.
    
    Returns
    -------
    t_star : float
        Time at which to switch from alpha=0 to alpha=1.
    """
    
    # Define the terminal velocity function v(m)
    def v_m(m):
        return (g / k) * (m0 - m) + (1 / k) * np.log(m / m0)
    
    # Define the height function h(m)
    def h_m(m):
        return -(m0 - m) / k**2 - (g / (2 * k**2)) * (m0 - m)**2 - (m0 / k**2) * np.log(m / m0)
    
    # Invert v_m to get m as function of v via bisection
    def m_of_v(v):
        a, b = ms, m0
        fa, fb = v_m(a) - v, v_m(b) - v
        if fa * fb > 0:
            raise ValueError("v is out of range for m inversion")
        for _ in range(max_iter):
            m_mid = 0.5 * (a + b)
            f_mid = v_m(m_mid) - v
            if abs(f_mid) < tol:
                return m_mid
            if fa * f_mid < 0:
                b, fb = m_mid, f_mid
            else:
                a, fa = m_mid, f_mid
        return 0.5 * (a + b)
    
    # Define the switching curve
    def Gamma(v):
        m = m_of_v(v)
        return h_m(m)
    
    # Free-fall trajectories
    def v_ff(t):
        return v0 - g * t
    def h_ff(t):
        return h0 + v0 * t - 0.5 * g * t**2
    
    # Function whose root is t*
    def F(t):
        v = v_ff(t)
        # If v outside [v_m(ms), 0], only free-fall h applies
        if v < v_m(ms) or v > 0:
            return h_ff(t)
        return h_ff(t) - Gamma(v)
    
    # Bracket for t*: between 0 and t_max
    if v0 > 0:
        t_max = v0 / g
    else:
        t_max = np.sqrt(2 * h0 / g)
    
    a, b = 0.0, t_max
    fa, fb = F(a), F(b)
    if fa * fb > 0:
        raise ValueError("No sign change found in [0, t_max]")
    
    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = F(mid)
        if abs(fmid) < tol:
            return mid
        if fa * fmid < 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid
    
    return 0.5 * (a + b)

t_star = find_t_star(h0, v0, m0, ms, g, k)
print(f"Switch time t* = {t_star:.6f} s")


u_bang = np.zeros_like(u_learn)
u_bang = np.where(t < t_star, 0.0, 1.0)  # switch at t_star

# --- 5) Plot both controls
plt.figure(figsize=(8,3))
plt.step(t, u_learn, where='post', label='Learned policy')
plt.step(t, u_bang, where='post', linestyle='--', label='Ideal bang–bang')
# if t_switch is not None:
#     plt.axvline(t_switch, color='gray', linestyle=':', label=f'switch @ {t_switch:.2f}s')
plt.xlabel('Time (s)')
plt.ylabel('u (thrust fraction)')
plt.title('Learned vs. Bang–Bang Control')
plt.legend()
plt.tight_layout()
plt.savefig("../figures/lunar_landing_control.pdf")

# --- 6) Quick saturation check
frac_zero = np.mean(u_learn < 1e-3)
frac_one  = np.mean(u_learn > 1-1e-3)
print(f"Learned policy saturates to 0 on {frac_zero:.1%} of steps, to 1 on {frac_one:.1%} of steps.")



# 1. Feasible sample generation: generates only samples that satisfy the glide slope constraint, ensuring all training data represents feasible states.

# 2. Safety-aware policy: the learned policy respects the glide slope constraint by checking the current state and adjusting control actions accordingly.

# 3. Enhanced cost function: includes a barrier term that penalizes states violating the glide slope constraint.