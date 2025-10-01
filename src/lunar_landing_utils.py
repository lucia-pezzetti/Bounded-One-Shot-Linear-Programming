import numpy as np
from scipy.linalg import solve_discrete_are
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Dynamics function: compute next state and sample new control
def lunar_landing_step(x, u, g=1.62, k=0.1, dt=0.1):
    h, v, m = x
    if m <= 0:
        m = 0.01  # Avoid division by zero

    # State updates (Euler method)
    h_next = h + v * dt                 # position update
    v_next = v + (-g + u / m) * dt      # velocity update (acceleration due to gravity and thrust)
    m_next = m - k * u * dt
    m_next = max(m_next, 0)             # Prevent negative mass

    x_next = (h_next, v_next, m_next)

    # Sample new control w in [0, 1]
    w = np.random.uniform(0, 1)
    
    return x_next, w

def is_feasible_state(h, v, m, ms, g, k):
    """
    Check if a state (h,v,m) can achieve soft landing.
    Uses the glide slope constraint: v >= -c*sqrt(h) where c = sqrt(2*a_max)
    """
    # at touchdown, the speed should be close to zero
    if h <= 0:
        return abs(v) <= 0.1  # Near zero velocity at ground
    
    a_max = max(0, 1.0/m - g)  # net upward acceleration at current mass
    if a_max <= 0:
        return False  # Cannot decelerate at all
    
    c = np.sqrt(2 * a_max)
    glide_slope_velocity = -c * np.sqrt(h)
    
    # if the downward velocity is too high, the agent will not be able to land safely
    return v >= glide_slope_velocity


# Generate N samples
def generate_samples(N, h_bounds, v_bounds, m_bounds, g, k, dt):
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

def generate_feasible_samples(N, h_bounds, v_bounds, m_bounds, ms, g, k, dt, max_attempts=10000):
    """
    Generate N samples that satisfy feasibility constraints
    """
    samples = []
    attempts = 0
    
    while len(samples) < N and attempts < max_attempts:
        h = np.random.uniform(*h_bounds)
        v = np.random.uniform(*v_bounds)
        m = np.random.uniform(*m_bounds)
        u = np.random.uniform(0, 1)
        
        # Check if state is feasible
        if is_feasible_state(h, v, m, ms, g, k):
            x = (h, v, m)
            x_next, w = lunar_landing_step(x, u, g, k, dt)
            row = [*x, u, *x_next, w]
            samples.append(row)
        
        attempts += 1
    
    if len(samples) < N:
        print(f"Warning: Only generated {len(samples)} feasible samples out of {N} requested")
    
    return np.array(samples)

# --- Non-uniform sampling focused near origin ---------------------------------
def generate_samples_nonuniform(N, h_bounds, v_bounds, m_bounds, g, k, dt, sigma_h=2.0, sigma_v=1.0, max_tries_per_sample=2000, rng=None):
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

# cost computation with barrier to impose soft landing
# Cost function with barrier
def stage_cost_with_barrier(h, v, m, ms, u, *, lam_fuel=0.05, beta_barrier=1000.0, ms_min=None, g=1.62):
    """
    Stage cost with barrier function to enforce feasibility
    """
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.asarray(m, dtype=float)
    u = np.asarray(u, dtype=float)
    
    # Fuel cost (penalize control effort)
    fuel_cost = lam_fuel * np.abs(u)
    
    # Barrier for feasibility
    if ms_min is None:
        ms_min = ms
    
    # feasibility check
    a_max = np.maximum(0.0, 1.0 / np.maximum(m, ms_min) - g)
    c = np.sqrt(2.0 * np.maximum(a_max, 1e-12))
    
    # Glide slope constraint: v >= -c*sqrt(h)
    h_pos = np.maximum(h, 1e-6)  # Avoid sqrt(0)
    glide_slope_bound = -c * np.sqrt(h_pos)
    
    # Barrier: penalize when v < glide_slope_bound
    violation = glide_slope_bound - v  # positive when violating
    barrier = beta_barrier * np.maximum(0.0, violation)**2
    
    # Terminal barrier: heavily penalize high velocity near ground
    ground_penalty = np.where(h < 2.0, 1000.0 * v**2, 0.0)
    
    # Additional penalty for negative heights
    # height_penalty = np.where(h < 0, 1e6, 0.0)
    
    total_cost = fuel_cost + barrier + ground_penalty # + height_penalty
    return np.nan_to_num(total_cost, nan=1e12, posinf=1e12, neginf=1e12)

def stage_cost_with_hard_penalty(h, v, m, ms, u, *, lam_fuel=0.05, v_tol=0.2,
                                 ms_min=None, g=1.62, big_M=10, w_h=0.2):
    """
    Returns lam_fuel*|u| normally, but returns `big_M` whenever any is violated:
      1) Glide slope: v >= -sqrt(2*max(1/m - g, 0))*sqrt(max(h, 0))
      2) Touchdown: if h <= 0, require |v| <= v_tol
      3) Physical sanity: h >= 0 and m >= ms_min
    """
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.asarray(m, dtype=float)
    u = np.asarray(u, dtype=float)

    if ms_min is None:
        ms_min = ms

    # Base cost: lam_fuel * |u|
    base = lam_fuel * np.abs(u)

    # Add a small altitude shaping cost to encourage descending
    base += w_h * h

    # Use current mass for more accurate constraint checking
    a_max = np.maximum(0.0, 1.0 / np.maximum(m, ms_min) - g)
    c = np.sqrt(2.0 * np.maximum(a_max, 1e-12))

    h_pos = np.maximum(h, 0.0)
    glide_ok  = v >= (-c * np.sqrt(h_pos))
    ground_ok = np.where(h <= 1e-6, np.abs(v) <= v_tol, True)
    phys_ok   = (h >= 0.0) & (m >= ms_min)

    # Reward safe landing
    mask_safe_land = (h <= 1e-6) & (np.abs(v) <= v_tol)
    base = np.where(mask_safe_land, base - big_M, base)

    ok = glide_ok & ground_ok & phys_ok
    return np.where(ok, base, base + big_M)