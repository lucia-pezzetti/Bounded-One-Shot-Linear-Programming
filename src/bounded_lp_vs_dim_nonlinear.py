import numpy as np
import cvxpy as cp
from scipy import sparse
from scipy.linalg import solve_discrete_are
from polynomial_features import FilteredPolynomialFeatures, StateOnlyPolynomialFeatures
from dynamical_systems_polished import point_mass_cubic_drag, point_mass_cubic_drag_2du
from config import (
    GAMMA, M_POINT_MASS_2D, K_POINT_MASS_2D, C_POINT_MASS_2D, DT_POINT_MASS_2D,
    RHO_POINT_MASS_2D, C_P_POINT_MASS, C_V_POINT_MASS,
    P_BOUNDS_POINT_MASS, V_BOUNDS_POINT_MASS, U_BOUNDS_POINT_MASS,
    SIGMA_M_POINT_MASS, SIGMA_K_POINT_MASS, SIGMA_C_POINT_MASS,
    OMEGA_MAX_POINT_MASS
)

# -------------------------
# Config (defaults; can be overridden by CLI)
# -------------------------
degree = 1           # polynomial feature degree
M_offline = 500      # auxiliary pool size for building P_y


def sample_system_params(seed, m0=M_POINT_MASS_2D, k0=K_POINT_MASS_2D, c0=C_POINT_MASS_2D,
                         sigma_m=SIGMA_M_POINT_MASS, sigma_k=SIGMA_K_POINT_MASS, 
                         sigma_c=SIGMA_C_POINT_MASS):
    """
    Sample system parameters from LogNormal distributions.
    
    X ~ LogNormal(μ, σ) where μ = log(X0) - 0.5*σ² ensures E[X] = X0.
    
    Args:
        seed: Random seed for reproducibility
        m0, k0, c0: Nominal values (means of the distributions)
        sigma_m, sigma_k, sigma_c: LogNormal scale parameters (controls CV)
        
    Returns:
        m, k, c: Sampled physical parameters
    """
    rng = np.random.RandomState(seed + 99999)  # Offset to avoid collision with data sampling
    
    # Sample from LogNormal: X = exp(μ + σ*Z) where Z ~ N(0,1)
    # With μ = log(X0) - 0.5*σ² to ensure E[X] = X0
    m = m0 * np.exp(sigma_m * rng.randn() - 0.5 * sigma_m**2)
    k = k0 * np.exp(sigma_k * rng.randn() - 0.5 * sigma_k**2)
    c = c0 * np.exp(sigma_c * rng.randn() - 0.5 * sigma_c**2)
    
    return m, k, c


def sample_modal_params(seed, n, alpha0=2.0, alpha_half_width=0.2):
    """
    Sample modal parameters for point_mass_cubic_drag_2du.

    Randomises:
      • Q_modes  ~ Haar(O(n))        — random orthogonal modal basis
      • B        — two randomly chosen physical actuators (sparse)
      • alpha    ~ Uniform(alpha0 ± alpha_half_width)

    Args:
        seed: Random seed for reproducibility
        n: Position/velocity dimension (number of DOFs)
        alpha0: Centre of the alpha range (default 2.0)
        alpha_half_width: Half-width of the alpha range (default 0.2)

    Returns:
        Q_modes: (n, n) orthogonal matrix drawn from Haar(O(n))
        B: (n, 2) sparse input map with two random physical actuators
        alpha: Sampled modal growth exponent
    """
    rng = np.random.RandomState(seed + 77777)  # Offset to avoid collision

    # --- Q_modes ~ Haar(O(n)) via QR of Gaussian matrix with sign correction ---
    H = rng.randn(n, n)
    Q, R = np.linalg.qr(H)
    # Multiply columns of Q by sign(diag(R)) to get a proper Haar sample
    Q = Q @ np.diag(np.sign(np.diag(R)))

    # --- B: pick 2 distinct physical coordinates uniformly at random ---
    actuators = rng.choice(n, size=2, replace=False)
    B = np.zeros((n, 2), dtype=float)
    B[actuators[0], 0] = 1.0
    B[actuators[1], 1] = 1.0

    # --- alpha ~ Uniform[alpha0 - hw, alpha0 + hw] ---
    alpha = rng.uniform(alpha0 - alpha_half_width, alpha0 + alpha_half_width)

    return Q, B, alpha


def create_point_mass_system(dx, N=0, M=0, mass=None, k=None, c=None, fixed_du=False,
                             Q_modes=None, B_modal=None, alpha=None):
    """
    Create a point_mass_cubic_drag system for given state dimension.
    
    Args:
        dx: State dimension (must be even: dx = 2n where n is position/velocity dimension)
        N: Number of samples (for compatibility)
        M: Auxiliary pool size (for compatibility)
        mass: Mass parameter (if None, uses config default M_POINT_MASS_2D)
        k: Spring constant / stiffness scale k0 (if None, uses config default K_POINT_MASS_2D)
        c: Damping coefficient (if None, uses config default C_POINT_MASS_2D)
        fixed_du: If True, use point_mass_cubic_drag_2du (du=2, under-actuated).
                  If False, use point_mass_cubic_drag (du=n, fully actuated).
        Q_modes: (n, n) orthogonal modal basis for the 2du variant (if None, uses class default DCT-II)
        B_modal: (n, 2) input map for the 2du variant (if None, uses class default [e_1 | e_n])
        alpha: Modal growth exponent for the 2du variant (if None, uses class default 2.0)
    
    Returns:
        system: point_mass_cubic_drag or point_mass_cubic_drag_2du instance
        C_cost: Cost matrix (dx, dx)
    """
    if dx % 2 != 0:
        raise ValueError(f"dx must be even for point mass system (dx={dx}). State is [p, v] where p and v are n-dimensional, so dx = 2n.")
    
    n = dx // 2  # Position/velocity dimension
    
    # Use provided parameters or fall back to config defaults
    mass_val = mass if mass is not None else M_POINT_MASS_2D
    k_val = k if k is not None else K_POINT_MASS_2D
    c_val = c if c is not None else C_POINT_MASS_2D
    
    # Build cost matrix C using config variables
    C_cost = np.diag(np.sqrt(np.concatenate([
        np.full(n, C_P_POINT_MASS),  # Position weights
        np.full(n, C_V_POINT_MASS)   # Velocity weights
    ])))
    
    common_kwargs = dict(
        n=n,
        mass=mass_val,
        c=c_val,
        delta_t=DT_POINT_MASS_2D,
        C=C_cost,
        rho=RHO_POINT_MASS_2D,
        gamma=GAMMA,
        N=N,
        M=M,
    )
    
    if fixed_du:
        # Under-actuated: du=2 with modal coupling (K = Q Λ Q^T)
        # Compute k0 so that the highest modal frequency equals OMEGA_MAX:
        #   lambda_n = k0 * n^alpha  =>  omega_n = sqrt(lambda_n / m) = OMEGA_MAX
        #   => k0 = m * OMEGA_MAX^2 / n^alpha
        alpha_val = alpha if alpha is not None else 2.0  # class default
        k0 = mass_val * OMEGA_MAX_POINT_MASS**2 / (n ** alpha_val)
        
        modal_kwargs = dict(k0=k0)
        if Q_modes is not None:
            modal_kwargs["Q"] = Q_modes
        if B_modal is not None:
            modal_kwargs["B"] = B_modal
        if alpha is not None:
            modal_kwargs["alpha"] = alpha
        system = point_mass_cubic_drag_2du(**modal_kwargs, **common_kwargs)
        print(f"  Using point_mass_cubic_drag_2du: dx={dx}, du=2, n={n}"
              f", k0={k0:.4f}, alpha={alpha_val:.2f}"
              f", omega_max={np.sqrt(system.lambdas[-1]/mass_val):.2f}")
    else:
        # Fully actuated: du=n
        system = point_mass_cubic_drag(m_u=n, B=None, k=k_val, **common_kwargs)
    
    return system, C_cost

def generate_dataset(system, N, seed=0):
    """
    Generate random (x,u) and compute x_next using system.step().
    Returns x, u, x_plus, u_plus (N samples).
    """
    np.random.seed(seed)
    x, u, x_plus, u_plus = system.generate_samples(
        P_BOUNDS_POINT_MASS,
        V_BOUNDS_POINT_MASS,
        U_BOUNDS_POINT_MASS,
        n_samples=N
    )
    return x, u, x_plus, u_plus

def auxiliary_samples(system, M, seed=0):
    """Auxiliary pool for building P_y"""
    np.random.seed(seed + 12345)
    x_aux, u_aux = system.generate_samples_auxiliary(
        P_BOUNDS_POINT_MASS,
        V_BOUNDS_POINT_MASS,
        U_BOUNDS_POINT_MASS,
        n_samples=M
    )
    return x_aux, u_aux

def solve_moment_matching_Q(P_z, P_z_next, P_y, L_xu, gamma, N, M, seed):
    """
    Two-stage approach for point mass systems (features on [x,u]).
    Returns: (status_stage2, status_stage1, Q_learned, C_val, E_Q_learned, mu)
    """
    d = P_z.shape[1]
    
    # -------------------------
    # Stage 1 (moment matching)
    # minimize || P_y^T Diag(mu) P_y - (P_z^T Diag(λ) P_z - γ P_z+^T Diag(λ) P_z+) ||_F
    # s.t.   1^T λ = 1, λ >= 0, mu >= 0
    # -------------------------
    lambda_var = cp.Variable(N, nonneg=True)
    mu_var     = cp.Variable(M, nonneg=True)
    
    Pz_const      = cp.Constant(P_z)
    Pz_next_const = cp.Constant(P_z_next)
    Py_const      = cp.Constant(P_y)
    
    def weighted_gram(Xc, w):
        return Xc.T @ cp.multiply(Xc, w[:, None])
    
    sum_PzPzT   = weighted_gram(Pz_const,  lambda_var)
    sum_PznPznT = weighted_gram(Pz_next_const, lambda_var)
    sum_PyPyT   = weighted_gram(Py_const,  mu_var)
    
    moment_match = sum_PzPzT - gamma * sum_PznPznT - sum_PyPyT
    
    constraints = [
        moment_match == 0,
        cp.sum(lambda_var) == 1,
    ]
    
    # Objective: make C ≈ I
    I_d = cp.Constant(sparse.eye(d, format="csr"))
    C_approx = sum_PyPyT
    # objective = cp.Minimize(cp.norm(C_approx - I_d, "fro"))
    objective = cp.Minimize(cp.norm(moment_match, "fro"))
    
    prob = cp.Problem(objective, constraints)
    mosek_params = {
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-9,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-9,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-9,
        "MSK_IPAR_INTPNT_BASIS": 1
    }
    try:
        prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)
        status1 = prob.status
    except Exception as e:
        print(f"    MOSEK failed in Stage 1: {e}")
        return "solver_error", "solver_error", None, None, None, None
    
    if status1 not in ("optimal", "optimal_inaccurate"):
        return "failed_stage1", status1, None, None, None, None
    
    mu = mu_var.value
    # C_val = P_y^T Diag(mu) P_y
    C_val = 0 
    for i in range(M):
        C_val += mu[i] * np.outer(P_y[i], P_y[i])
    
    # -------------------------
    # Stage 2 (LP):  maximize trace(C_val @ Q)
    # s.t. forall i:  trace(Q @ (p_i p_i^T - γ p^+_i p^+_i^T)) <= ℓ(x_i,u_i)
    # -------------------------
    Q = cp.Variable((d, d))
    
    cons = []
    for i in range(N):
        p = P_z[i]
        pn = P_z_next[i]
        Mi = np.outer(p, p) - gamma * np.outer(pn, pn)
        cons.append(cp.sum(cp.multiply(Q, Mi)) <= L_xu[i])
    
    # Objective: sum_i mu_i * <Q, y_i y_i^T>
    terms = []
    for i in range(M):
        yi = P_y[i]
        terms.append(mu[i] * cp.sum(cp.multiply(Q, np.outer(yi, yi))))
        
    obj = cp.Maximize(cp.sum(terms))
    prob_lp = cp.Problem(obj, cons)
    mosek_params = {
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-9,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-9,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-9,
        "MSK_IPAR_INTPNT_BASIS": 1
    }
    try:
        prob_lp.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)
        status2 = prob_lp.status
    except Exception as e:
        print(f"    MOSEK failed in Stage 2: {e}")
        return "solver_error", status1, None, None, None, None
    
    # Return the learned Q matrix, polynomial features, and optimization values if successful
    if status2 in ("optimal", "optimal_inaccurate"):
        Q_learned = Q.value
        E_Q_learned = prob_lp.value
        return status2, status1, Q_learned, C_val, E_Q_learned, mu
    else:
        return status2, status1, None, None, None, None

def solve_identity_Q(P_z, P_z_next, L_xu, gamma, N, seed, dx, du):
    """
    Baseline: min trace(Q) s.t.  p_i^T Q p_i - γ p^+_i^T Q p^+_i <= ℓ(x_i,u_i)  for all i.
    
    Args:
        P_z: Polynomial features for (x, u) pairs (N, d)
        P_z_next: Polynomial features for (x_plus, u_plus) pairs (N, d)
        L_xu: Stage costs (N,)
        gamma: Discount factor
        N: Number of samples
        seed: Random seed
        dx: State dimension
        du: Input dimension
    """
    d = P_z.shape[1]
    
    Q_id = cp.Variable((d, d))
    
    # Individual constraints for better numerical stability
    cons = []
    for i in range(N):
        p = P_z[i]
        pn = P_z_next[i]
        Mi = np.outer(p, p) - gamma * np.outer(pn, pn)
        cons.append(cp.trace(Q_id @ cp.Constant(Mi)) <= L_xu[i])
    
    # Create identity matrix C = I
    C_matrix = np.eye(d)
    
    objective = cp.Maximize(cp.trace(cp.Constant(C_matrix) @ Q_id))
    prob = cp.Problem(objective, cons)
    
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        return prob.status
    except Exception as e:
        print(f"    MOSEK failed in Identity LP: {e}")
        return "solver_error"

# -------------------------
# Monte Carlo Policy Evaluation
# -------------------------

def extract_greedy_gain(Q_learned, dx, du):
    """
    Extract the linear greedy policy gain from a learned Q-matrix.
    
    For degree-1 features z = [x, u] (no bias), Q(x,u) = z^T Q_learned z.
    The greedy policy is u*(x) = K @ x  with  K = -Q_uu^{-1} Q_xu^T.
    
    Args:
        Q_learned: (d, d) learned Q-matrix where d = dx + du
        dx: state dimension
        du: input dimension
        
    Returns:
        K: (du, dx) gain matrix, or None if Q_uu is singular
    """
    # Symmetrise (the LP has no symmetry constraint)
    Q_sym = 0.5 * (Q_learned + Q_learned.T)
    
    Q_xu = Q_sym[:dx, dx:dx+du]   # (dx, du)
    Q_uu = Q_sym[dx:dx+du, dx:dx+du]  # (du, du)
    
    try:
        K = -np.linalg.solve(Q_uu, Q_xu.T)  # (du, dx)
        if not np.isfinite(K).all():
            return None
        return K
    except np.linalg.LinAlgError:
        return None


def compute_lqr_gain(system, gamma):
    """
    Compute the LQR gain for the linearised system.
    
    Args:
        system: point_mass_cubic_drag or point_mass_cubic_drag_2du instance
        gamma: discount factor
        
    Returns:
        K_lqr: (du, dx) gain matrix
    """
    A_d, B_d = system.linearized_system(use_backward_euler=False)
    Q_cost = system.Q   # dx × dx
    R_cost = system.R   # du × du
    
    if gamma == 1.0:
        P = solve_discrete_are(A_d, B_d, Q_cost, R_cost)
        K = np.linalg.inv(R_cost + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
    else:
        Ad = np.sqrt(gamma) * A_d
        Rd = R_cost / gamma
        P = solve_discrete_are(Ad, B_d, Q_cost, Rd)
        K = gamma * np.linalg.inv(R_cost + gamma * B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
    
    return K  # u = -K @ x


def mc_policy_cost(system, K, gamma, n_rollouts=200, T=500, seed=0,
                   u_lo=U_BOUNDS_POINT_MASS[0], u_hi=U_BOUNDS_POINT_MASS[1]):
    """
    Monte Carlo estimate of the discounted cost under policy u = -K @ x.
    
    Rolls out trajectories from random initial states drawn uniformly from
    [P_BOUNDS] × [V_BOUNDS] and computes:
        J = (1/n_rollouts) Σ_traj Σ_{t=0}^{T-1} γ^t ℓ(x_t, u_t)
    
    Args:
        system: dynamical system with .step() and .cost()
        K: (du, dx) gain matrix (policy is u = -K @ x, clipped to bounds)
        gamma: discount factor
        n_rollouts: number of trajectories
        T: trajectory horizon
        seed: random seed for initial states
        u_lo, u_hi: control bounds for clipping
        
    Returns:
        mean_cost: average discounted cost across rollouts
        std_cost: std of discounted cost across rollouts
    """
    rng = np.random.RandomState(seed + 55555)
    dx = system.N_x
    n = dx // 2
    
    # Draw random initial states
    x0s = np.zeros((n_rollouts, dx))
    x0s[:, :n] = rng.uniform(P_BOUNDS_POINT_MASS[0], P_BOUNDS_POINT_MASS[1], size=(n_rollouts, n))
    x0s[:, n:] = rng.uniform(V_BOUNDS_POINT_MASS[0], V_BOUNDS_POINT_MASS[1], size=(n_rollouts, n))
    
    costs = np.zeros(n_rollouts)
    discount = np.power(gamma, np.arange(T))  # γ^0, γ^1, ..., γ^{T-1}
    
    for r in range(n_rollouts):
        x = x0s[r].copy()
        traj_cost = 0.0
        for t in range(T):
            u = -K @ x
            u = np.clip(u, u_lo, u_hi)
            c = float(system.cost(x.reshape(1, -1), u.reshape(1, -1))[0])
            traj_cost += discount[t] * c
            x = system.step(x, u)
            # Early termination if state blows up
            if np.any(np.abs(x) > 1e6):
                # Add a large penalty for remaining steps
                traj_cost += discount[t:].sum() * c
                break
        costs[r] = traj_cost
    
    return float(np.mean(costs)), float(np.std(costs))


def evaluate_Q_quality(Q_learned, system, dx, du, gamma, seed):
    """
    Evaluate the quality of a learned Q-matrix via Monte Carlo policy evaluation.
    
    Extracts the greedy policy from Q_learned, rolls out trajectories, and
    compares against the LQR baseline (from the linearised system).
    
    Args:
        Q_learned: (d, d) learned Q-matrix
        system: dynamical system instance
        dx: state dimension
        du: input dimension
        gamma: discount factor
        seed: random seed
        
    Returns:
        dict with keys:
            mm_cost_mean, mm_cost_std: MC cost of the MM greedy policy
            lqr_cost_mean, lqr_cost_std: MC cost of the LQR policy
            cost_ratio: mm_cost_mean / lqr_cost_mean  (< 1 means MM is better)
            K_mm_norm: Frobenius norm of MM gain
            K_lqr_norm: Frobenius norm of LQR gain
            Q_uu_cond: condition number of Q_uu block
        or None if the greedy policy cannot be extracted
    """
    K_mm = extract_greedy_gain(Q_learned, dx, du)
    if K_mm is None:
        print("    [eval] Cannot extract greedy gain (Q_uu singular)")
        return None
    
    # Check Q_uu conditioning
    Q_sym = 0.5 * (Q_learned + Q_learned.T)
    Q_uu = Q_sym[dx:dx+du, dx:dx+du]
    Q_uu_cond = float(np.linalg.cond(Q_uu))
    
    # LQR baseline
    try:
        K_lqr = compute_lqr_gain(system, gamma)
    except Exception as e:
        print(f"    [eval] LQR gain computation failed: {e}")
        K_lqr = np.zeros((du, dx))  # fallback: zero control
    
    # MC rollouts
    mm_mean, mm_std = mc_policy_cost(system, K_mm, gamma, seed=seed)
    lqr_mean, lqr_std = mc_policy_cost(system, K_lqr, gamma, seed=seed)
    
    # Normalized difference: (J_MM - J_LQR) / J_LQR
    norm_diff = (mm_mean - lqr_mean) / lqr_mean if lqr_mean > 0 else float('inf')
    
    print(f"    [eval] MM cost: {mm_mean:.2f} ± {mm_std:.2f}  |  "
          f"LQR cost: {lqr_mean:.2f} ± {lqr_std:.2f}  |  "
          f"norm_diff: {norm_diff:.4f}")
    
    return {
        "mm_cost_mean": mm_mean,
        "mm_cost_std": mm_std,
        "lqr_cost_mean": lqr_mean,
        "lqr_cost_std": lqr_std,
        "cost_norm_diff": norm_diff,
        "K_mm_norm": float(np.linalg.norm(K_mm, 'fro')),
        "K_lqr_norm": float(np.linalg.norm(K_lqr, 'fro')),
        "Q_uu_cond": Q_uu_cond,
    }


def run_one(seed, dx, N, gamma, M_offline, degree, exclude_u_squared=False, randomize_system=False, fixed_du=False):
    """Generate a dataset and solve both LPs; return boundedness flags.
    
    Args:
        seed: Random seed
        dx: State dimension
        N: Number of samples
        gamma: Discount factor
        M_offline: Auxiliary pool size
        degree: Polynomial feature degree
        exclude_u_squared: Exclude u^2 terms from features
        randomize_system: If True, sample m, k, c from LogNormal distributions per seed
        fixed_du: If True, use point_mass_cubic_drag_2du (du=2, under-actuated)
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Sample system parameters if randomization is enabled
    if randomize_system:
        mass, k, c = sample_system_params(seed)
        print(f"  Sampled params: m={mass:.3f}, k={k:.3f}, c={c:.3f}")
    else:
        mass, k, c = None, None, None  # Use defaults
    
    # Sample modal parameters for the 2du variant (Q_modes, B, alpha)
    n = dx // 2
    Q_modes, B_modal, alpha = None, None, None
    if randomize_system and fixed_du:
        Q_modes, B_modal, alpha = sample_modal_params(seed, n)
        actuated = np.flatnonzero(B_modal.any(axis=1)).tolist()
        print(f"  Sampled modal: alpha={alpha:.3f}, actuators={actuated}")
    
    # Create system for this dimension (with optional sampled parameters)
    system, C_cost = create_point_mass_system(
        dx, N=0, M=0, mass=mass, k=k, c=c, fixed_du=fixed_du,
        Q_modes=Q_modes, B_modal=B_modal, alpha=alpha,
    )
    
    # Generate dataset
    x, u, x_plus, u_plus = generate_dataset(system, N=N, seed=seed)
    
    # Compute costs
    L_xu = system.cost(x, u)
    
    # Compute polynomial features once for both methods
    z = np.concatenate([x, u], axis=1)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)
    
    # Use StateOnlyPolynomialFeatures for point mass systems (consistent with nonlinear pipeline)
    if exclude_u_squared and degree > 1:
        poly = FilteredPolynomialFeatures(degree=degree, include_bias=False, dx=dx, du=system.N_u, exclude_u_squared=True)
    else:
        # Use StateOnlyPolynomialFeatures for all cases (works for degree 1 and higher)
        poly = StateOnlyPolynomialFeatures(degree=degree, include_bias=False, dx=dx, du=system.N_u)
    
    Z_all = np.concatenate([z, z_plus], axis=0)
    P_all = poly.fit_transform(Z_all)
    P_z = P_all[:N]
    P_z_next = P_all[N:]
    
    # Auxiliary samples for moment matching
    x_aux, u_aux = auxiliary_samples(system, M_offline, seed=seed)
    y_aux = np.concatenate([x_aux, u_aux], axis=1)
    P_y = poly.transform(y_aux)
    
    # Solve both methods using the same polynomial features
    status_m2, status_m1, Q_learned, C_val, E_Q_learned, mu = solve_moment_matching_Q(P_z, P_z_next, P_y, L_xu, gamma, N, M_offline, seed)
    status_id = solve_identity_Q(P_z, P_z_next, L_xu, gamma, N, seed, dx, system.N_u)
    
    def is_bounded(status):
        return status in ("optimal", "optimal_inaccurate")

    print(f'Moment matching status: {status_m2}, bounded: {is_bounded(status_m2)}')
    print(f'Identity status: {status_id}, bounded: {is_bounded(status_id)}')
    
    result = {
        "seed": seed,
        "dx": dx,
        "du": system.N_u,
        "moment_matching_status": status_m2,
        "moment_matching_bounded": is_bounded(status_m2),
        "moment_stage1_status": status_m1,
        "identity_status": status_id,
        "identity_bounded": is_bounded(status_id),
    }
    
    # Monte Carlo policy evaluation (only when MM LP is bounded)
    if is_bounded(status_m2) and Q_learned is not None:
        eval_result = evaluate_Q_quality(Q_learned, system, dx, system.N_u, gamma, seed)
        if eval_result is not None:
            result.update(eval_result)
    
    # Store sampled system parameters if randomization was used
    if randomize_system:
        result["mass"] = float(system.m)
        result["c"] = float(system.c)
        if fixed_du:
            # k0 is derived from (m, omega_max, n, alpha); store it for traceability
            n = dx // 2
            alpha_val = alpha if alpha is not None else 2.0
            result["k0"] = float(system.m * OMEGA_MAX_POINT_MASS**2 / (n ** alpha_val))
            result["alpha"] = float(alpha_val)
            if B_modal is not None:
                result["actuators"] = np.flatnonzero(B_modal.any(axis=1)).tolist()
        else:
            result["k"] = float(system.k)
    
    return result

def sweep_over_dims(dims, seeds, N, gamma, M_offline, degree, exclude_u_squared=False, randomize_system=False, fixed_du=False):
    results = []
    for dx in dims:
        if dx % 2 != 0:
            print(f"[WARN] Skipping dx={dx} (must be even for point mass system)")
            continue
        for s in seeds:
            print(f"Running seed {s} for dx={dx}")
            try:
                results.append(run_one(seed=int(s), dx=int(dx), N=N, gamma=gamma, M_offline=M_offline, degree=degree, exclude_u_squared=exclude_u_squared, randomize_system=randomize_system, fixed_du=fixed_du))
            except Exception as e:
                print(f"[WARN] Failed for dx={dx}, seed={s}: {e}. Skipping.")
                continue
    return results

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Boundedness vs state dimension (nonlinear point mass systems).")
    parser.add_argument("--dims", type=str, default="4,6,8,10,12,14,16,18,20", help="Comma-separated list of state dimensions (e.g., '2,4,6,8,10'). Default: 2,4,6,8,10")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per dimension (default: 10)")
    parser.add_argument("--N", type=int, default=1000, help="Number of samples for dataset (default: 1000)")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor (default: from config)")
    parser.add_argument("--M_offline", type=int, default=500, help="Offline pool size (default: 500)")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial feature degree (default: 1)")
    parser.add_argument("--exclude_u_squared", action="store_true", help="Exclude u^2 terms from polynomial features (for degree 2)")
    parser.add_argument("--randomize_system", action="store_true", help="Sample m, k, c from LogNormal distributions per seed (default: use fixed values)")
    parser.add_argument("--fixed_du", action="store_true", help="Use point_mass_cubic_drag_2du (du=2 fixed, under-actuated). Default: fully actuated (du=n).")
    parser.add_argument("--out_json", type=str, default=None, help="Raw results JSON (if None, auto-generated with N)")
    parser.add_argument("--plot_dir", type=str, default=None, help="Plot filename (if None, auto-generated with N)")
    args = parser.parse_args()
    
    # Parse dimensions
    if args.dims.strip():
        dims = [int(s) for s in args.dims.split(",") if s.strip()]
    else:
        dims = [2, 4, 6, 8, 10]  # Default: dx = 2, 4, 6, 8, 10
    
    # Generate seeds for each experiment
    seeds = list(range(0, args.seeds + 0))
    
    # Auto-generate filenames with N if not provided
    if args.out_json is None:
        args.out_json = f"bounded_vs_dim_results_nonlinear_N_{args.N}_fixed_du_{args.fixed_du}_degree_{args.degree}.json"
    if args.plot_dir is None:
        args.plot_dir = f"../figures/bounded_vs_dim_percentages_nonlinear_N_{args.N}_fixed_du_{args.fixed_du}_degree_{args.degree}.pdf"
    
    results = sweep_over_dims(dims, seeds, N=args.N, gamma=args.gamma, M_offline=args.M_offline, degree=args.degree, exclude_u_squared=args.exclude_u_squared, randomize_system=args.randomize_system, fixed_du=args.fixed_du)
    
    if len(results) == 0:
        print("No results (no valid dimensions or files). Exiting.")
        raise SystemExit(0)
    
    df = pd.DataFrame(results)
    
    # --- Boundedness aggregation ---
    agg = df.groupby("dx").agg(
        moment_bounded_mean=("moment_matching_bounded", "mean"),
        identity_bounded_mean=("identity_bounded", "mean"),
        n=("seed", "count"),
    ).reset_index()
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_mean"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_mean"]
    
    # --- MC policy cost aggregation (only for bounded MM runs) ---
    has_eval = "cost_norm_diff" in df.columns and df["cost_norm_diff"].notna().any()
    if has_eval:
        eval_df = df.dropna(subset=["cost_norm_diff"])
        eval_agg = eval_df.groupby("dx").agg(
            cost_norm_diff_mean=("cost_norm_diff", "mean"),
            cost_norm_diff_std=("cost_norm_diff", "std"),
            mm_cost_mean=("mm_cost_mean", "mean"),
            lqr_cost_mean=("lqr_cost_mean", "mean"),
            n_eval=("cost_norm_diff", "count"),
        ).reset_index()
        # Merge into main agg
        agg = agg.merge(eval_agg, on="dx", how="left")
    
    # Save raw + aggregated
    df.to_json(args.out_json, orient="records", indent=2)
    agg_json_name = f"bounded_vs_dim_percentages_nonlinear_N_{args.N}_fixed_du_{args.fixed_du}_degree_{args.degree}.json"
    agg.to_json(agg_json_name, orient="records", indent=2)
    
    # Print compact table
    print("\n=== Boundedness ===")
    cols = ["dx", "moment_bounded_pct", "identity_bounded_pct"]
    print(agg[cols].to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    
    if has_eval:
        print("\n=== MC Policy Evaluation (bounded MM runs only) ===")
        eval_cols = ["dx", "cost_norm_diff_mean", "cost_norm_diff_std", "mm_cost_mean", "lqr_cost_mean", "n_eval"]
        available_cols = [c for c in eval_cols if c in agg.columns]
        print(agg[available_cols].dropna().to_string(index=False, float_format=lambda v: f"{v:6.4f}"))
    
    # --- Plotting ---
    n_panels = 2 if has_eval else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    
    # Panel 1: Boundedness percentages
    ax1 = axes[0]
    ax1.plot(agg["dx"], agg["moment_bounded_pct"],
                    marker="o", label="Moment matching", color="blue")
    ax1.plot(agg["dx"], agg["identity_bounded_pct"],
                    marker="s", label="Identity covariance", color="red")
    ax1.set_xlabel("State dimension (dx)")
    ax1.set_ylabel("Bounded LPs (%)")
    ax1.set_title("Bounded LP percentage vs dimension")
    ax1.legend()
    ax1.grid(True)
    
    # Panel 2: MC normalized cost difference (J_MM - J_LQR) / J_LQR
    if has_eval:
        ax2 = axes[1]
        mask = agg["cost_norm_diff_mean"].notna()
        dx_vals = agg.loc[mask, "dx"].values
        mean_vals = agg.loc[mask, "cost_norm_diff_mean"].values
        std_vals = agg.loc[mask, "cost_norm_diff_std"].fillna(0).values
        ax2.plot(dx_vals, mean_vals, marker="D", color="green",
                 label=r"$(J_{\mathrm{MM}} - J_{\mathrm{LQR}}) / J_{\mathrm{LQR}}$")
        ax2.fill_between(dx_vals, mean_vals - std_vals, mean_vals + std_vals,
                         color="green", alpha=0.2)
        ax2.set_xlabel("State dimension (dx)")
        ax2.set_ylabel("Normalized cost difference")
        ax2.set_title("MC policy cost: MM vs LQR (linearised)")
        ax2.legend()
        ax2.grid(True)
    
    fig.suptitle("Point Mass with Cubic Drag", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(args.plot_dir, dpi=150, bbox_inches="tight")
    print("\nSaved:", args.out_json)
    print("Aggregated results saved:", agg_json_name)
    print("Figures saved in", args.plot_dir)
