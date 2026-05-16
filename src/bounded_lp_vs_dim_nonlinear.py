import time
import numpy as np
import cvxpy as cp
from scipy import sparse
from scipy.linalg import solve_discrete_are
from polynomial_features import FilteredPolynomialFeatures, StateOnlyPolynomialFeatures
from dynamical_systems import point_mass_cubic_drag, point_mass_cubic_drag_1du
from config import (
    GAMMA, M_POINT_MASS_2D, K_POINT_MASS_2D, C_POINT_MASS_2D, DT_POINT_MASS_2D,
    RHO_POINT_MASS_2D, C_P_POINT_MASS, C_V_POINT_MASS,
    P_BOUNDS_POINT_MASS, V_BOUNDS_POINT_MASS, P_BOUNDS_POINT_MASS_TEST, V_BOUNDS_POINT_MASS_TEST, U_BOUNDS_POINT_MASS,
    SIGMA_M_POINT_MASS, SIGMA_K_POINT_MASS, SIGMA_C_POINT_MASS,
    OMEGA_MAX_POINT_MASS
)

SYSTEM_ALIASES = {
    "point_mass": "point_mass",
    "pointmass": "point_mass",
    "point_mass_cubic_drag": "point_mass",
}


def normalize_system_name(system_name):
    """Normalize user-facing system identifiers."""
    key = str(system_name).strip().lower()
    if key not in SYSTEM_ALIASES:
        allowed = ", ".join(sorted(SYSTEM_ALIASES))
        raise ValueError(f"Unknown system '{system_name}'. Allowed identifiers: {allowed}")
    return SYSTEM_ALIASES[key]


def parse_bounds_pair(text, expected_dim=None):
    """
    Parse compact bounds strings.

    Accepted forms:
      low,high                 -> one interval
      low,high;low,high;...    -> one interval per dimension
    """
    parts = [p.strip() for p in str(text).split(";") if p.strip()]
    bounds = []
    for part in parts:
        vals = [float(v.strip()) for v in part.split(",") if v.strip()]
        if len(vals) != 2:
            raise ValueError(f"Bounds entry '{part}' must contain exactly two values.")
        if vals[0] >= vals[1]:
            raise ValueError(f"Bounds lower value must be less than upper value: '{part}'")
        bounds.append((vals[0], vals[1]))

    if expected_dim is not None and len(bounds) == 1 and expected_dim > 1:
        bounds = bounds * expected_dim
    if expected_dim is not None and len(bounds) != expected_dim:
        raise ValueError(f"Expected {expected_dim} bounds entries, got {len(bounds)} from '{text}'")
    return bounds


def _coerce_diag_values(values, expected_dim, name):
    if isinstance(values, str):
        raw_values = [v.strip() for v in values.replace(";", ",").split(",") if v.strip()]
        arr = np.asarray([float(v) for v in raw_values], dtype=float)
    else:
        arr = np.asarray(values, dtype=float).reshape(-1)

    if arr.size == 1 and expected_dim > 1:
        arr = np.full(expected_dim, float(arr.item()))
    if arr.size != expected_dim:
        raise ValueError(f"{name} must be scalar or length {expected_dim}, got {arr.size}.")
    return arr


def _coerce_nonnegative_diag_values(values, expected_dim, name):
    arr = _coerce_diag_values(values, expected_dim, name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return arr


def _has_positive_nonnegative_values(values, name):
    if isinstance(values, str):
        raw_values = [v.strip() for v in values.replace(";", ",").split(",") if v.strip()]
        arr = np.asarray([float(v) for v in raw_values], dtype=float)
    else:
        arr = np.asarray(values, dtype=float).reshape(-1)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return bool(np.any(arr > 0.0))


def normalize_point_mass_gravity_type(gravity_type):
    if gravity_type is None:
        gravity_type = "linear"
    key = str(gravity_type).strip().lower()
    aliases = {
        "linear": "linear",
        "identity": "linear",
        "p": "linear",
        "position": "linear",
        "sin": "sin",
        "sine": "sin",
        "tanh": "tanh",
        "hyperbolic_tangent": "tanh",
        "log": "log",
        "asinh": "log",
        "arcsinh": "log",
        "signed_log": "log",
    }
    if key not in aliases:
        raise ValueError("--point_mass_gravity_type must be one of: linear, sin, tanh, log.")
    return aliases[key]


def point_mass_gravity_argument(P, gravity_type):
    gravity_type = normalize_point_mass_gravity_type(gravity_type)
    if gravity_type == "linear":
        return P
    if gravity_type == "sin":
        return np.sin(P)
    if gravity_type == "tanh":
        return np.tanh(P)
    return np.arcsinh(P)


def resolve_point_mass_gravity_diag(n, gravity_enabled=False, gravity_diag=None,
                                    stiffness_scalar=None, stiffness_matrix=None,
                                    gravity_margin=0.1):
    """
    Resolve the diagonal G for the optional point-mass gravity term +G h(p).

    If no explicit diagonal is supplied, choose an isotropic diagonal just above
    the smallest eigenvalue of K, so K-G has at least one negative eigenvalue.
    """
    if gravity_diag is not None:
        return _coerce_diag_values(gravity_diag, n, "point_mass_gravity_diag")
    if not gravity_enabled:
        return None
    if gravity_margin <= 0.0:
        raise ValueError("--point_mass_gravity_margin must be positive when auto-selecting G.")

    if stiffness_matrix is not None:
        min_stiffness = float(np.min(np.linalg.eigvalsh(np.asarray(stiffness_matrix, dtype=float))))
    elif stiffness_scalar is not None:
        min_stiffness = float(stiffness_scalar)
    else:
        raise ValueError("Need stiffness_scalar or stiffness_matrix to auto-select point-mass gravity.")

    return np.full(n, min_stiffness + float(gravity_margin))


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
    rng = np.random.RandomState(seed + 99999)
    
    # Sample from LogNormal: X = exp(μ + σ*Z) where Z ~ N(0,1)
    # With μ = log(X0) - 0.5*σ² to ensure E[X] = X0
    m = m0 * np.exp(sigma_m * rng.randn() - 0.5 * sigma_m**2)
    k = k0 * np.exp(sigma_k * rng.randn() - 0.5 * sigma_k**2)
    c = c0 * np.exp(sigma_c * rng.randn() - 0.5 * sigma_c**2)
    
    return m, k, c


def sample_modal_params(seed, n, du=1, alpha0=2.0, alpha_half_width=0.2):
    """
    Sample modal parameters for point_mass_cubic_drag_1du.

    Randomises:
      • Q_modes  ~ Haar(O(n))        — random orthogonal modal basis
      • B        — dense random actuation directions (unit-norm columns)
      • alpha    ~ Uniform(alpha0 ± alpha_half_width)

    Args:
        seed: Random seed for reproducibility
        n: Position/velocity dimension (number of DOFs)
        du: Number of control inputs (currently only 1 is supported here)
        alpha0: Centre of the alpha range (default 2.0)
        alpha_half_width: Half-width of the alpha range (default 0.2)

    Returns:
        Q_modes: (n, n) orthogonal matrix drawn from Haar(O(n))
        B: (n, du) dense input map with unit-norm columns
        alpha: Sampled modal growth exponent
    """
    rng = np.random.RandomState(seed + 77777)

    # --- Q_modes ~ Haar(O(n)) via QR of Gaussian matrix with sign correction ---
    H = rng.randn(n, n)
    Q, R = np.linalg.qr(H)
    # Multiply columns of Q by sign(diag(R)) to get a proper Haar sample
    Q = Q @ np.diag(np.sign(np.diag(R)))

    # --- B: dense random actuation directions ---
    B_NORM = 5.0
    B = rng.randn(n, du)
    col_norms = np.linalg.norm(B, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-12)
    B = B_NORM * B / col_norms

    # --- alpha ~ Uniform[alpha0 - hw, alpha0 + hw] ---
    alpha = rng.uniform(alpha0 - alpha_half_width, alpha0 + alpha_half_width)

    return Q, B, alpha


def create_point_mass_system(dx, N=0, M=0, mass=None, k=None, c=None, fixed_du=0,
                             Q_modes=None, B_modal=None, alpha=None,
                             gravity_enabled=False, gravity_diag=None,
                             gravity_margin=0.1, gravity_type="linear",
                             integrator="rk4", linear_damping=0.0,
                             q4_p=0.0, q4_v=0.0, r4_u=0.0):
    """
    Create a point_mass_cubic_drag system for given state dimension.
    
    Args:
        dx: State dimension (must be even: dx = 2n where n is position/velocity dimension)
        N: Number of samples (for compatibility)
        M: Auxiliary pool size (for compatibility)
        mass: Mass parameter (if None, uses config default M_POINT_MASS_2D)
        k: Spring constant / stiffness scale k0 (if None, uses config default K_POINT_MASS_2D)
        c: Damping coefficient (if None, uses config default C_POINT_MASS_2D)
        fixed_du: 0 = fully actuated (du=n), 1 = du=1 (under-actuated).
        Q_modes: (n, n) orthogonal modal basis for the fixed-du variants (if None, uses class default DCT-II)
        B_modal: (n, du) input map for the fixed-du variants (if None, uses class default)
        alpha: Modal growth exponent for the fixed-du variants (if None, uses class default 2.0)
        gravity_enabled: add optional +G h(p) term to acceleration.
        gravity_diag: scalar or length-n diagonal entries for G. If None and gravity_enabled, G is auto-selected.
        gravity_margin: positive margin used for auto-selecting G above the smallest stiffness eigenvalue.
        gravity_type: h(p) form, one of "linear", "sin", "tanh", or "log".
        integrator: point-mass discrete-time integrator, "euler" or "rk4".
        linear_damping: optional viscous damping coefficient d in -d v.
        q4_p, q4_v, r4_u: scalar or per-dimension quartic weights for position,
            velocity, and input stage-cost terms.
    
    Returns:
        system: point_mass_cubic_drag or point_mass_cubic_drag_1du instance
        C_cost: Cost matrix (dx, dx)
    """
    if dx % 2 != 0:
        raise ValueError(f"dx must be even for point mass system (dx={dx}). State is [p, v] where p and v are n-dimensional, so dx = 2n.")
    
    n = dx // 2  # Position/velocity dimension
    gravity_type = normalize_point_mass_gravity_type(gravity_type)
    integrator = str(integrator).strip().lower()
    if integrator not in ("euler", "rk4"):
        raise ValueError("--point_mass_integrator must be one of: euler, rk4.")
    linear_damping = float(linear_damping)
    if linear_damping < 0.0:
        raise ValueError("--point_mass_linear_damping must be nonnegative.")
    q4_p_val = _coerce_nonnegative_diag_values(q4_p, n, "point_mass_q4_p")
    q4_v_val = _coerce_nonnegative_diag_values(q4_v, n, "point_mass_q4_v")
    
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
        linear_damping=linear_damping,
        delta_t=DT_POINT_MASS_2D,
        integrator=integrator,
        C=C_cost,
        rho=RHO_POINT_MASS_2D,
        gamma=GAMMA,
        N=N,
        M=M,
        q4_p=q4_p_val,
        q4_v=q4_v_val,
    )
    
    if fixed_du == 1:
        # Under-actuated: du=1 with modal coupling (K = Q Λ Q^T)
        # Compute k0 so that the highest modal frequency equals OMEGA_MAX:
        #   lambda_n = k0 * n^alpha  =>  omega_n = sqrt(lambda_n / m) = OMEGA_MAX
        #   => k0 = m * OMEGA_MAX^2 / n^alpha
        alpha_val = alpha if alpha is not None else 2.0  # class default
        k0 = mass_val * OMEGA_MAX_POINT_MASS**2 / (n ** alpha_val)
        gravity_diag_val = resolve_point_mass_gravity_diag(
            n,
            gravity_enabled=gravity_enabled,
            gravity_diag=gravity_diag,
            stiffness_scalar=k0,
            gravity_margin=gravity_margin,
        )
        
        r4_u_val = _coerce_nonnegative_diag_values(r4_u, 1, "point_mass_r4_u")
        modal_kwargs = dict(
            k0=k0,
            gravity_diag=gravity_diag_val,
            gravity_type=gravity_type,
            r4_u=r4_u_val,
        )
        if Q_modes is not None:
            modal_kwargs["Q"] = Q_modes
        if B_modal is not None:
            modal_kwargs["B"] = B_modal
        if alpha is not None:
            modal_kwargs["alpha"] = alpha
        
        system = point_mass_cubic_drag_1du(**modal_kwargs, **common_kwargs)
        class_name = "point_mass_cubic_drag_1du"
        
        print(f"  Using {class_name}: dx={dx}, du={fixed_du}, n={n}"
              f", k0={k0:.4f}, alpha={alpha_val:.2f}"
              f", omega_max={np.sqrt(system.lambdas[-1]/mass_val):.2f}")
    elif fixed_du == 0:
        # Fully actuated: du=n
        r4_u_val = _coerce_nonnegative_diag_values(r4_u, n, "point_mass_r4_u")
        gravity_diag_val = resolve_point_mass_gravity_diag(
            n,
            gravity_enabled=gravity_enabled,
            gravity_diag=gravity_diag,
            stiffness_scalar=k_val,
            gravity_margin=gravity_margin,
        )
        system = point_mass_cubic_drag(
            m_u=n, B=None, k=k_val, gravity_diag=gravity_diag_val,
            gravity_type=gravity_type, r4_u=r4_u_val, **common_kwargs
        )
    else:
        raise ValueError(f"Unsupported fixed_du={fixed_du}. Allowed values are 0 (fully actuated) or 1.")

    if getattr(system, "has_gravity", False):
        eigvals = np.linalg.eigvalsh(system.linearized_stiffness_minus_gravity())
        print(
            "  Gravity term enabled: "
            f"type={system.gravity_type}, "
            f"G diag in [{system.gravity_diag.min():.4f}, {system.gravity_diag.max():.4f}], "
            f"min eig(K-G)={eigvals[0]:.4f}"
        )
        if eigvals[0] >= 0.0:
            print("  [WARN] K-G has no negative eigenvalue; the origin is not destabilized by this G.")
    print(f"  Point-mass integrator: {system.integrator}")
    if system.linear_damping > 0.0:
        print(f"  Linear damping enabled: d={system.linear_damping:.4g}")
    if getattr(system, "has_quartic_cost", False):
        print(
            "  Quartic point-mass cost enabled: "
            f"q4_p in [{system.q4_p.min():.4g}, {system.q4_p.max():.4g}], "
            f"q4_v in [{system.q4_v.min():.4g}, {system.q4_v.max():.4g}], "
            f"r4_u in [{system.r4_u.min():.4g}, {system.r4_u.max():.4g}]"
        )
    
    return system, C_cost




def create_nonlinear_system(system_name, dx, N=0, M=0, mass=None, k=None, c=None,
                            fixed_du=0, Q_modes=None, B_modal=None, alpha=None,
                            point_mass_gravity=False, point_mass_gravity_diag=None,
                            point_mass_gravity_margin=0.1,
                            point_mass_gravity_type="linear",
                            point_mass_integrator="rk4",
                            point_mass_linear_damping=0.0,
                            point_mass_q4_p=0.0, point_mass_q4_v=0.0,
                            point_mass_r4_u=0.0):
    """Factory for nonlinear benchmark systems used by this experiment script."""
    system_name = normalize_system_name(system_name)
    if system_name == "point_mass":
        return create_point_mass_system(
            dx, N=N, M=M, mass=mass, k=k, c=c, fixed_du=fixed_du,
            Q_modes=Q_modes, B_modal=B_modal, alpha=alpha,
            gravity_enabled=point_mass_gravity,
            gravity_diag=point_mass_gravity_diag,
            gravity_margin=point_mass_gravity_margin,
            gravity_type=point_mass_gravity_type,
            integrator=point_mass_integrator,
            linear_damping=point_mass_linear_damping,
            q4_p=point_mass_q4_p,
            q4_v=point_mass_q4_v,
            r4_u=point_mass_r4_u,
        )
    raise ValueError(f"Unsupported normalized system '{system_name}'")


def get_sampling_bounds(system, sampling_bounds=None):
    """Return (x1/p bounds, x2/v bounds, u bounds) for system sample generation."""
    return P_BOUNDS_POINT_MASS, V_BOUNDS_POINT_MASS, U_BOUNDS_POINT_MASS


def generate_dataset(system, N, seed=0, sampling_bounds=None):
    """
    Generate random (x,u) and compute x_next using system.step().
    Returns x, u, x_plus, u_plus (N samples).
    """
    np.random.seed(seed)
    p_bounds, v_bounds, u_bounds = get_sampling_bounds(system, sampling_bounds)
    x, u, x_plus, u_plus = system.generate_samples(
        p_bounds,
        v_bounds,
        u_bounds,
        n_samples=N
    )
    return x, u, x_plus, u_plus

def auxiliary_samples(system, M, seed=0, sampling_bounds=None):
    """Auxiliary pool for building P_y"""
    np.random.seed(seed + 12345)
    p_bounds, v_bounds, u_bounds = get_sampling_bounds(system, sampling_bounds)
    x_aux, u_aux = system.generate_samples_auxiliary(
        p_bounds,
        v_bounds,
        u_bounds,
        n_samples=M
    )
    return x_aux, u_aux

def _build_sym_constraint_matrix(P_z, P_z_next, gamma):
    """
    Build the symmetric constraint matrix A_sym and upper-triangle index map.
    
    Since M_i = p_i p_i^T - γ p^+_i p^+_i^T is symmetric, only the upper
    triangle of Q matters.  We parameterise Q by its upper triangle
    s ∈ R^{d(d+1)/2} and build A_sym such that  trace(Q @ M_i) = A_sym[i,:] @ s.
    
    For diagonal entries (j, j):   coefficient = M_i[j, j]
    For off-diagonal entries (j, k), j<k: coefficient = 2 * M_i[j, k]
    (because Q_{jk} + Q_{kj} contributes twice, and we set Q_{jk} = Q_{kj} = s_idx).
    
    Args:
        P_z: (N, d) polynomial features
        P_z_next: (N, d) polynomial features of next state
        gamma: discount factor
        
    Returns:
        A_sym: (N, d_sym) constraint matrix where d_sym = d(d+1)/2
        d: feature dimension
        d_sym: number of symmetric variables
    """
    N, d = P_z.shape
    d_sym = d * (d + 1) // 2
    
    # Build full M matrices: M_i[j,k] = P_z[i,j]*P_z[i,k] - γ*P_z_next[i,j]*P_z_next[i,k]
    # Shape: (N, d, d)
    M_full = (P_z[:, :, None] * P_z[:, None, :]) \
           - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])
    
    # Extract upper-triangle indices
    rows_idx, cols_idx = np.triu_indices(d)
    
    # A_sym[:, k] = M_full[:, rows_idx[k], cols_idx[k]]  (diagonal)
    #             or 2 * M_full[:, rows_idx[k], cols_idx[k]]  (off-diagonal)
    A_sym = M_full[:, rows_idx, cols_idx]  # (N, d_sym)
    
    # Scale off-diagonal columns by 2
    off_diag_mask = rows_idx != cols_idx
    A_sym[:, off_diag_mask] *= 2.0
    
    return A_sym, d, d_sym


def _sym_vec(mat_d, d):
    """Convert a (d, d) symmetric matrix to its upper-triangle vector representation."""
    rows_idx, cols_idx = np.triu_indices(d)
    v = mat_d[rows_idx, cols_idx].copy()
    off_diag = rows_idx != cols_idx
    v[off_diag] *= 2.0  # account for both (j,k) and (k,j) contributions
    return v


def _sym_to_full(s_vec, d):
    """Reconstruct a (d, d) symmetric matrix from its upper-triangle vector."""
    Q = np.zeros((d, d))
    rows_idx, cols_idx = np.triu_indices(d)
    Q[rows_idx, cols_idx] = s_vec
    Q[cols_idx, rows_idx] = s_vec  # mirror to lower triangle
    return Q


def solve_moment_matching_Q(P_z, P_z_next, P_y, L_xu, gamma, N, M, seed,
                             A_sym=None, d=None, d_sym=None):
    """
    Two-stage approach for point mass systems (features on [x,u]).
    
    Args:
        A_sym: Precomputed symmetric constraint matrix (N, d_sym). If None, built internally.
        d, d_sym: Feature dimension and symmetric variable count (required if A_sym is provided).
    
    Returns: (status_stage2, status_stage1, Q_learned, C_val, E_Q_learned, mu)
    """
    if d is None:
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
    P_y_weighted = P_y * np.sqrt(mu)[:, None]          # (M, d)
    C_val = P_y_weighted.T @ P_y_weighted               # (d, d)
    
    # -------------------------
    # Stage 2 (LP):  maximize trace(C_val @ Q)
    # s.t. forall i:  trace(Q @ M_i) <= ℓ_i
    #
    # Symmetric reduction: Q is parameterised by its upper triangle s ∈ R^{d_sym}.
    # A_sym @ s <= ℓ,  maximise c_sym^T s.
    # -------------------------
    if A_sym is None:
        A_sym, d, d_sym = _build_sym_constraint_matrix(P_z, P_z_next, gamma)
    
    c_sym = _sym_vec(C_val, d)
    
    s = cp.Variable(d_sym)
    cons = [A_sym @ s <= L_xu]
    obj = cp.Maximize(c_sym @ s)
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
    
    if status2 in ("optimal", "optimal_inaccurate"):
        Q_learned = _sym_to_full(s.value, d)
        E_Q_learned = prob_lp.value
        return status2, status1, Q_learned, C_val, E_Q_learned, mu
    else:
        return status2, status1, None, None, None, None

def solve_identity_Q(P_z, P_z_next, L_xu, gamma, N, seed, dx, du,
                     A_sym=None, d=None, d_sym=None):
    """
    Baseline: maximize trace(Q) s.t. trace(Q @ M_i) <= ℓ_i  for all i.
    
    Symmetric reduction: Q parameterised by upper triangle s ∈ R^{d_sym}.
    
    Args:
        A_sym: Precomputed symmetric constraint matrix (N, d_sym). If None, built internally.
        d, d_sym: Feature dimension and symmetric variable count (required if A_sym is provided).
    """
    if A_sym is None:
        A_sym, d, d_sym = _build_sym_constraint_matrix(P_z, P_z_next, gamma)
    
    # Objective: maximise trace(Q) — only diagonal entries of I contribute
    c_sym = _sym_vec(np.eye(d), d)
    
    s = cp.Variable(d_sym)
    cons = [A_sym @ s <= L_xu]
    objective = cp.Maximize(c_sym @ s)
    prob = cp.Problem(objective, cons)
    
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        return prob.status
    except Exception as e:
        print(f"    MOSEK failed in Identity LP: {e}")
        return "solver_error"

def gaussian_relevance_matrix_stateonly_degree2(dx, du=1, include_bias=False):
    """
    Build M = E[p(z) p(z)^T] for z = (x, u) ~ N(0, I),
    with scalar u and

        p(z) = [p'(x), u]^T

    where p'(x) contains all monomials in x of degree <= 2.

    Assumed ordering:
        include_bias=False:
            [x_1,...,x_dx, x_1^2,...,x_dx^2, x_1 x_2, x_1 x_3, ..., x_{dx-1} x_dx, u]

        include_bias=True:
            [1, x_1,...,x_dx, x_1^2,...,x_dx^2, x_1 x_2, ..., x_{dx-1} x_dx, u]

    Returns:
        M: (d, d) symmetric relevance matrix
    """
    if du != 1:
        raise ValueError(f"This helper assumes scalar u (du=1), got du={du}.")

    n_lin = dx
    n_sq = dx
    n_cross = dx * (dx - 1) // 2

    if include_bias:
        d = 1 + n_lin + n_sq + n_cross + 1
        M = np.zeros((d, d))

        i0 = 0
        i_lin = slice(1, 1 + n_lin)
        i_sq = slice(i_lin.stop, i_lin.stop + n_sq)
        i_cross = slice(i_sq.stop, i_sq.stop + n_cross)
        i_u = d - 1

        # E[1 * 1]
        M[i0, i0] = 1.0

        # E[1 * x_i^2] = 1
        M[i0, i_sq] = 1.0
        M[i_sq, i0] = 1.0

        # E[x x^T] = I
        M[i_lin, i_lin] = np.eye(n_lin)

        # E[s s^T], s_i = x_i^2:
        # diag = 3, off-diag = 1
        M[i_sq, i_sq] = np.ones((n_sq, n_sq)) + 2.0 * np.eye(n_sq)

        # E[r r^T] = I for r = [x_i x_j]_{i<j}
        if n_cross > 0:
            M[i_cross, i_cross] = np.eye(n_cross)

        # E[u^2] = 1
        M[i_u, i_u] = 1.0

        return M

    else:
        d = n_lin + n_sq + n_cross + 1
        M = np.zeros((d, d))

        i_lin = slice(0, n_lin)
        i_sq = slice(i_lin.stop, i_lin.stop + n_sq)
        i_cross = slice(i_sq.stop, i_sq.stop + n_cross)
        i_u = d - 1

        # E[x x^T] = I
        M[i_lin, i_lin] = np.eye(n_lin)

        # E[s s^T], s_i = x_i^2:
        # diag = 3, off-diag = 1
        M[i_sq, i_sq] = np.ones((n_sq, n_sq)) + 2.0 * np.eye(n_sq)

        # E[r r^T] = I
        if n_cross > 0:
            M[i_cross, i_cross] = np.eye(n_cross)

        # E[u^2] = 1
        M[i_u, i_u] = 1.0

        return M

def solve_gaussian_Q(P_z, P_z_next, L_xu, gamma, N, seed, dx, du,
                     A_sym=None, d=None, d_sym=None, include_bias=False):
    """
    Gaussian-relevance baseline:
        maximize trace(M @ Q)
        s.t.     trace(Q @ M_i) <= l_i  for all i

    where
        M = E[p(z)p(z)^T],   z ~ N(0, I),
    and p(z) = [p'(x), u]^T with p'(x) containing all state monomials of degree <= 2.

    Assumes scalar u and degree-2 StateOnlyPolynomialFeatures-style ordering.
    """
    if A_sym is None:
        A_sym, d, d_sym = _build_sym_constraint_matrix(P_z, P_z_next, gamma)

    M_gauss = gaussian_relevance_matrix_stateonly_degree2(
        dx=dx, du=du, include_bias=include_bias
    )

    if M_gauss.shape != (d, d):
        raise ValueError(
            f"Gaussian relevance matrix shape mismatch: M has shape {M_gauss.shape}, "
            f"but feature dimension is d={d}. "
            f"Check include_bias / feature ordering / polynomial degree."
        )

    c_sym = _sym_vec(M_gauss, d)

    s = cp.Variable(d_sym)
    cons = [A_sym @ s <= L_xu]
    objective = cp.Maximize(c_sym @ s)
    prob = cp.Problem(objective, cons)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        return prob.status
    except Exception as e:
        print(f"    MOSEK failed in Gaussian LP: {e}")
        return "solver_error"

# -------------------------
# Monte Carlo Policy Evaluation
# -------------------------
def extract_greedy_gain(Q_learned, dx, du):
    """
    Extract the greedy policy gain from a learned Q-matrix.
    
    For StateOnlyPolynomialFeatures, φ(x,u) = [φ_x(x), u] where φ_x(x) are
    polynomial features of x (dimension n_phi).  The Q-matrix has shape
    (n_phi + du, n_phi + du).
    
    Q(x,u) = φ(x,u)^T M φ(x,u)
            = φ_x^T M_xx φ_x + 2 φ_x^T M_xu u + u^T M_uu u
    
    Minimising over u:
        u*(x) = -M_uu^{-1} M_ux φ_x(x)        (a NONLINEAR policy when degree > 1)
    
    Args:
        Q_learned: (d, d) learned Q-matrix where d = n_phi + du
        dx: state dimension
        du: input dimension
        
    Returns:
        K: (du, n_phi) positive gain matrix.  Policy convention: u*(x) = -K @ φ_x(x).
           (Matches LQR convention: compute_lqr_gain returns K with u = -K @ x.)
           n_phi = d - du = Q_learned.shape[0] - du.
           Returns None if M_uu is singular.
    """
    d = Q_learned.shape[0]
    n_phi = d - du  # number of state polynomial features
    
    # Symmetrise
    Q_sym = 0.5 * (Q_learned + Q_learned.T)
    
    # Partition: last du rows/cols correspond to u
    M_xu = Q_sym[:n_phi, n_phi:]   # (n_phi, du)
    M_uu = Q_sym[n_phi:, n_phi:]   # (du, du)
    
    try:
        K = np.linalg.solve(M_uu, M_xu.T)
        if not np.isfinite(K).all():
            return None
        return K
    except np.linalg.LinAlgError:
        return None


def compute_lqr_gain(system, gamma):
    """
    Compute the LQR gain for the linearised system.
    
    Args:
        system: point_mass_cubic_drag or point_mass_cubic_drag_1du instance
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



    """
    Generic Monte Carlo policy evaluation for small nonlinear systems that expose
    .cost() and either .vectorized_step() or .step().
    """
    rng = np.random.RandomState(seed + 55555)
    dx = system.N_x
    du = K.shape[0]

    state_bounds = np.asarray(getattr(system, "state_bounds", [(-1.0, 1.0)] * dx), dtype=float)
    if state_bounds.shape == (2,) and dx == 1:
        state_bounds = state_bounds.reshape(1, 2)
    if state_bounds.shape != (dx, 2):
        raise ValueError(f"state_bounds must have shape ({dx}, 2), got {state_bounds.shape}")

    if u_lo is None or u_hi is None:
        u_bounds = getattr(system, "u_bounds", (-np.inf, np.inf))
        if u_lo is None:
            u_lo = u_bounds[0]
        if u_hi is None:
            u_hi = u_bounds[1]

    X = np.empty((n_rollouts, dx))
    for j in range(dx):
        X[:, j] = rng.uniform(state_bounds[j, 0], state_bounds[j, 1], size=n_rollouts)

    costs = np.zeros(n_rollouts)
    blown = np.zeros(n_rollouts, dtype=bool)
    discount = np.power(gamma, np.arange(T))

    n_rec = min(int(record_trajectories), n_rollouts)
    if n_rec > 0:
        X_hist = np.empty((n_rec, T + 1, dx))
        U_hist = np.empty((n_rec, T, du))
        X_hist[:, 0, :] = X[:n_rec]

    for t in range(T):
        active = ~blown
        U = np.zeros((n_rollouts, du))
        step_cost = np.zeros(n_rollouts)

        if np.any(active):
            X_active = X[active]
            if poly_x is not None:
                Phi_X = poly_x.transform(X_active)
                U_active = -(Phi_X @ K.T)
            else:
                U_active = -(X_active @ K.T)
            U_active = np.clip(U_active, u_lo, u_hi)
            U[active] = U_active
            step_cost[active] = system.cost(X_active, U_active)

        if n_rec > 0:
            U_hist[:, t, :] = U[:n_rec]

        costs += discount[t] * step_cost

        X_new = X.copy()
        if hasattr(system, "vectorized_step"):
            X_new[active] = system.vectorized_step(X[active], U[active])
        else:
            X_new[active] = np.asarray([system.step(x, u) for x, u in zip(X[active], U[active])])

        new_blown = active & ((~np.isfinite(X_new).all(axis=1)) | np.any(np.abs(X_new) > 1e6, axis=1))
        just_blown = new_blown & ~blown
        if np.any(just_blown):
            remaining_discount = discount[t:].sum()
            costs[just_blown] += remaining_discount * step_cost[just_blown]
        blown |= new_blown

        X = X_new

        if n_rec > 0:
            X_hist[:, t + 1, :] = X[:n_rec]

        if blown.all():
            if n_rec > 0:
                for s in range(t + 2, T + 1):
                    X_hist[:, s, :] = X[:n_rec]
                for s in range(t + 1, T):
                    U_hist[:, s, :] = 0.0
            break

    trajectories = None
    if n_rec > 0:
        trajectories = {
            "X": X_hist,
            "U": U_hist,
            "costs": costs[:n_rec].copy(),
        }

    return float(np.mean(costs)), float(np.std(costs)), trajectories


def mc_policy_cost(system, K, gamma, n_rollouts=200, T=20000, seed=0,
                   u_lo=U_BOUNDS_POINT_MASS[0], u_hi=U_BOUNDS_POINT_MASS[1],
                   record_trajectories=0, poly_x=None):
    """
    Monte Carlo estimate of the discounted cost under a policy derived from gain K.
    
    If poly_x is None (degree-1 / LQR case):
        u = -K @ x              (linear policy, K is du × dx)
    If poly_x is provided (degree > 1):
        u = -K @ φ_x(x)         (nonlinear policy, K is du × n_phi)
    
    All rollouts are batched: states are (n_rollouts, dx) and updated
    simultaneously using vectorised forward-Euler dynamics.
    
    Args:
        system: dynamical system with .cost() and attributes (m, c, delta_t, B, Q, R, K if 2du)
        K: gain matrix — (du, dx) for linear or (du, n_phi) for polynomial
        gamma: discount factor
        n_rollouts: number of trajectories
        T: trajectory horizon
        seed: random seed for initial states
        u_lo, u_hi: control bounds for clipping
        record_trajectories: number of rollouts to record full state/control histories for
                             (0 = no recording). The first `record_trajectories` rollouts are recorded.
        poly_x: fitted PolynomialFeatures transformer for states (from poly.poly_x).
                If None, the policy is linear: u = -K @ x.
                If provided, the policy is: u = -K @ poly_x.transform(x).
        
    Returns:
        mean_cost: average discounted cost across rollouts
        std_cost: std of discounted cost across rollouts
        trajectories: dict with keys "X" (n_rec, T+1, dx) and "U" (n_rec, T, du),
                      or None if record_trajectories == 0
    """


    rng = np.random.RandomState(seed + 55555)
    dx = system.N_x
    du = K.shape[0]
    n_dof = dx // 2
    
    # Draw random initial states: (n_rollouts, dx)
    X = np.zeros((n_rollouts, dx))
    X[:, :n_dof] = rng.uniform(P_BOUNDS_POINT_MASS_TEST[0], P_BOUNDS_POINT_MASS_TEST[1], size=(n_rollouts, n_dof))
    X[:, n_dof:] = rng.uniform(V_BOUNDS_POINT_MASS_TEST[0], V_BOUNDS_POINT_MASS_TEST[1], size=(n_rollouts, n_dof))
    
    costs = np.zeros(n_rollouts)
    blown = np.zeros(n_rollouts, dtype=bool)
    discount = np.power(gamma, np.arange(T))
    
    # Optional trajectory recording
    n_rec = min(int(record_trajectories), n_rollouts)
    if n_rec > 0:
        X_hist = np.empty((n_rec, T + 1, dx))
        U_hist = np.empty((n_rec, T, du))
        X_hist[:, 0, :] = X[:n_rec]
    
    for t in range(T):
        # Policy: u = -K @ φ(x) where φ is either identity or polynomial
        if poly_x is not None:
            # Nonlinear policy: φ_x(x) includes quadratic (and higher) terms
            Phi_X = poly_x.transform(X)    # (n_rollouts, n_phi)
            U = -(Phi_X @ K.T)             # (n_rollouts, du)
        else:
            # Linear policy: u = -K @ x
            U = -(X @ K.T)
        U = np.clip(U, u_lo, u_hi)
        
        if n_rec > 0:
            U_hist[:, t, :] = U[:n_rec]
        
        # Stage cost: defer to the system so optional quartic terms stay consistent.
        step_cost = system.cost(X, U)
        
        # Accumulate (only for non-blown trajectories)
        costs += discount[t] * np.where(blown, 0.0, step_cost)
        
        # Batched integration step. This respects the system's selected
        # integrator, which matters near weakly damped stable equilibria.
        X_new = system.vectorized_step(X, U)
        
        new_blown = np.any(np.abs(X_new) > 1e6, axis=1)
        just_blown = new_blown & ~blown
        if np.any(just_blown):
            remaining_discount = discount[t:].sum()
            costs[just_blown] += remaining_discount * step_cost[just_blown]
        blown |= new_blown
        
        X = X_new
        
        if n_rec > 0:
            X_hist[:, t + 1, :] = X[:n_rec]
        
        # Early exit if all trajectories have blown up
        if blown.all():
            # Fill remaining history with last state if recording
            if n_rec > 0:
                for s in range(t + 2, T + 1):
                    X_hist[:, s, :] = X[:n_rec]
                for s in range(t + 1, T):
                    U_hist[:, s, :] = 0.0
            break
    
    trajectories = None
    if n_rec > 0:
        trajectories = {
            "X": X_hist,
            "U": U_hist,
            "costs": costs[:n_rec].copy(),  # discounted cost per recorded rollout
        }
    
    return float(np.mean(costs)), float(np.std(costs)), trajectories


def plot_policy_trajectories(traj_mm, traj_zero, dt, dx, du, save_path=None,
                              n_show=200, mm_cost=None, zero_cost=None):
    """
    Plot MM and zero-input trajectories on the same axes.
    
    Layout: 1 row with two columns — positions on the left, controls on the right.
    All plotted trajectories use a muted colour palette with light transparency
    for better paper-ready appearance.
    
    Args:
        traj_mm:   dict with "X" (n_rec, T+1, dx) and "U" (n_rec, T, du)
        traj_zero: dict with "X" (n_rec, T+1, dx) and "U" (n_rec, T, du)
        dt: time step
        dx: state dimension
        du: input dimension
        save_path: if not None, save figure to this path
        n_show: number of rollouts to overlay (default 200)
        mm_cost: tuple (mean, std) of MM policy cost, or None
        zero_cost: tuple (mean, std) of zero-input policy cost, or None
    """
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["lines.linewidth"] = 0.8

    T_plus_1 = traj_mm["X"].shape[1]
    T_ctrl = traj_mm["U"].shape[1]
    t_state = np.arange(T_plus_1) * dt
    t_ctrl = np.arange(T_ctrl) * dt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)

    lw = 0.7
    alpha = 0.18
    color_mm = "#0072B2"      # paper-friendly blue
    color_zero = "#D55E00"    # paper-friendly red

    X_mm = traj_mm["X"][:n_show]
    U_mm = traj_mm["U"][:n_show]
    X_zero = traj_zero["X"][:n_show]
    U_zero = traj_zero["U"][:n_show]
    n_pos = dx // 2 if dx % 2 == 0 else dx

    ax_s, ax_u = axes

    # --- Left: Positions ---
    for r in range(X_zero.shape[0]):
        for j in range(n_pos):
            ax_s.plot(t_state, X_zero[r, :, j], color=color_zero,
                      alpha=alpha, linewidth=lw,
                      label="Uncontrolled" if (r == 0 and j == 0) else None)
    for r in range(X_mm.shape[0]):
        for j in range(n_pos):
            ax_s.plot(t_state, X_mm[r, :, j], color=color_mm,
                      alpha=alpha, linewidth=lw,
                      label="MM control" if (r == 0 and j == 0) else None)
    ax_s.set_title("State trajectories", fontsize=13, fontweight="semibold")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("Position")
    ax_s.grid(True, alpha=0.25, linestyle="--")

    # --- Right: Controls ---
    for r in range(U_zero.shape[0]):
        for j in range(du):
            ax_u.plot(t_ctrl, U_zero[r, :, j], color=color_zero,
                      alpha=alpha, linewidth=lw,
                      label="Uncontrolled" if (r == 0 and j == 0) else None)
    for r in range(U_mm.shape[0]):
        for j in range(du):
            ax_u.plot(t_ctrl, U_mm[r, :, j], color=color_mm,
                      alpha=alpha, linewidth=lw,
                      label="MM control" if (r == 0 and j == 0) else None)
    ax_u.set_title("Control trajectories", fontsize=13, fontweight="semibold")
    ax_u.set_xlabel("Time (s)")
    ax_u.set_ylabel("Control")
    ax_u.grid(True, alpha=0.25, linestyle="--")

    # Single legend for both axes
    handles, labels = ax_s.get_legend_handles_labels()
    if handles:
        ax_s.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(f"Trajectory comparison (dx={dx})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"    [plot] Trajectory figure saved to {save_path}")
    plt.close(fig)


def evaluate_Q_quality(Q_learned, system, dx, du, gamma, seed,
                       plot_trajectories_dir=None, poly=None, eval_horizon=20000):
    """
    Evaluate the quality of a learned Q-matrix via Monte Carlo policy evaluation.
    
    Extracts the greedy policy from Q_learned, rolls out trajectories, and
    compares against the LQR baseline (from the linearised system).
    
    For degree > 1, the MM policy is NONLINEAR:
        u*(x) = -M_uu^{-1} M_ux φ_x(x)
    where φ_x(x) are the polynomial features of x. The poly transformer
    must be passed so that mc_policy_cost can evaluate this nonlinear policy.
    
    Args:
        Q_learned: (d, d) learned Q-matrix where d = n_phi + du
        system: dynamical system instance
        dx: state dimension
        du: input dimension
        gamma: discount factor
        seed: random seed
        plot_trajectories_dir: if not None, save trajectory comparison plots
                               to this directory (filename auto-generated from dx/seed)
        poly: fitted StateOnlyPolynomialFeatures instance (or None for degree 1).
              When provided, its .poly_x attribute is used to compute φ_x(x).
        
    Returns:
        dict with keys:
            mm_cost_mean, mm_cost_std: MC cost of the MM greedy policy
            lqr_cost_mean, lqr_cost_std: MC cost of the LQR policy
            cost_norm_diff: (J_MM - J_LQR) / J_LQR
            K_mm_norm: Frobenius norm of MM gain
            K_lqr_norm: Frobenius norm of LQR gain
            Q_uu_cond: condition number of Q_uu block
        or None if the greedy policy cannot be extracted
    """
    K_mm = extract_greedy_gain(Q_learned, dx, du)
    if K_mm is None:
        print("    [eval] Cannot extract greedy gain (Q_uu singular)")
        return None
    
    # Check Q_uu conditioning (last du rows/cols are control)
    d = Q_learned.shape[0]
    n_phi = d - du
    Q_sym = 0.5 * (Q_learned + Q_learned.T)
    Q_uu = Q_sym[n_phi:, n_phi:]
    Q_uu_cond = float(np.linalg.cond(Q_uu))
    
    # State polynomial feature transformer for the nonlinear MM policy
    poly_x = poly.poly_x if poly is not None else None
    
    # LQR baseline
    try:
        K_lqr = compute_lqr_gain(system, gamma)
    except Exception as e:
        print(f"    [eval] LQR gain computation failed: {e}")
        K_lqr = np.zeros((du, dx))  # fallback: zero control
    
    # Zero-input baseline
    K_zero = np.zeros((du, dx))
    
    # Decide whether to record trajectories (record all rollouts for plotting)
    n_rec = 200 if plot_trajectories_dir is not None else 0
    
    # MC rollouts
    mm_mean, mm_std, traj_mm = mc_policy_cost(system, K_mm, gamma, seed=seed,
                                               T=eval_horizon,
                                               record_trajectories=n_rec,
                                               poly_x=poly_x)
    lqr_mean, lqr_std, traj_lqr = mc_policy_cost(system, K_lqr, gamma, seed=seed,
                                                   T=eval_horizon,
                                                   record_trajectories=n_rec)
    zero_mean, zero_std, traj_zero = mc_policy_cost(system, K_zero, gamma, seed=seed,
                                                     T=eval_horizon,
                                                     record_trajectories=n_rec)
    
    # Normalized difference: (J_MM - J_LQR) / J_LQR
    norm_diff = (mm_mean - lqr_mean) / lqr_mean if lqr_mean > 0 else float('inf')
    
    print(f"    [eval] MM cost: {mm_mean:.2f} ± {mm_std:.2f}  |  "
          f"LQR cost: {lqr_mean:.2f} ± {lqr_std:.2f}  |  "
          f"Zero cost: {zero_mean:.2f} ± {zero_std:.2f}  |  "
          f"norm_diff: {norm_diff:.4f}")
    
    # Plot trajectory comparison: MM vs zero-input
    if plot_trajectories_dir is not None and traj_mm is not None and traj_zero is not None:
        import os
        os.makedirs(plot_trajectories_dir, exist_ok=True)
        save_path = os.path.join(plot_trajectories_dir,
                                 f"trajectories_dx{dx}_seed{seed}.png")
        mm_traj_costs = traj_mm["costs"]
        zero_traj_costs = traj_zero["costs"]
        plot_policy_trajectories(traj_mm, traj_zero, system.delta_t, dx, du,
                                  save_path=save_path,
                                  mm_cost=(float(np.mean(mm_traj_costs)),
                                           float(np.std(mm_traj_costs))),
                                  zero_cost=(float(np.mean(zero_traj_costs)),
                                             float(np.std(zero_traj_costs))))
        
        # Save trajectory data as NPZ file (position and control only, no velocity)
        npz_path = os.path.join(plot_trajectories_dir,
                                f"trajectories_dx{dx}_seed{seed}_data.npz")
        T_state = traj_mm["X"].shape[1]
        T_ctrl = traj_mm["U"].shape[1]
        t_state = np.arange(T_state) * system.delta_t
        t_ctrl = np.arange(T_ctrl) * system.delta_t
        
        # Extract position components only (first dx//2 for point mass systems)
        n_pos = dx // 2
        P_mm = traj_mm["X"][:, :, :n_pos]   # (n_rec, T+1, n_pos)
        P_zero = traj_zero["X"][:, :, :n_pos]  # (n_rec, T+1, n_pos)
        
        np.savez_compressed(
            npz_path,
            P_mm=P_mm,
            U_mm=traj_mm["U"],
            P_zero=P_zero,
            U_zero=traj_zero["U"],
            t_state=t_state,
            t_ctrl=t_ctrl,
            dx=dx,
            du=du,
            seed=seed,
            mm_cost_mean=mm_mean,
            mm_cost_std=mm_std,
            zero_cost_mean=zero_mean,
            zero_cost_std=zero_std,
        )
        print(f"    [plot] Trajectory data saved to {npz_path}")
    
    return {
        "mm_cost_mean": mm_mean,
        "mm_cost_std": mm_std,
        "lqr_cost_mean": lqr_mean,
        "lqr_cost_std": lqr_std,
        "zero_cost_mean": zero_mean,
        "zero_cost_std": zero_std,
        "cost_norm_diff": norm_diff,
        "K_mm_norm": float(np.linalg.norm(K_mm, 'fro')),
        "K_lqr_norm": float(np.linalg.norm(K_lqr, 'fro')),
        "Q_uu_cond": Q_uu_cond,
    }


def run_one(seed, dx, N, gamma, M_offline, degree, exclude_u_squared=False,
            randomize_system=False, fixed_du=0, plot_trajectories_dir=None,
            system_name="point_mass", duffing_params=None, sampling_bounds=None,
            eval_horizon=20000, point_mass_gravity=False,
            point_mass_gravity_diag=None, point_mass_gravity_margin=0.1,
            point_mass_gravity_type="linear", point_mass_integrator="rk4",
            point_mass_linear_damping=0.0,
            point_mass_q4_p=0.0, point_mass_q4_v=0.0, point_mass_r4_u=0.0):
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
        fixed_du: 0 = fully actuated (du=n), 1 = du=1 (under-actuated)
        plot_trajectories_dir: if not None, save trajectory comparison plots to this directory
        point_mass_gravity: add +G h(p) to point-mass cubic-drag dynamics
        point_mass_gravity_type: one of "linear", "sin", "tanh", or "log"
        point_mass_integrator: "rk4" for trajectory-friendly integration or
            "euler" for the original forward-Euler discretization
        point_mass_linear_damping: optional viscous damping coefficient d in -d v
        point_mass_q4_p, point_mass_q4_v, point_mass_r4_u: optional quartic
            cost weights for point-mass position, velocity, and input.
    """
    system_name = normalize_system_name(system_name)
    timings = {}
    t_total_start = time.perf_counter()

    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Sample system parameters if randomization is enabled
    if randomize_system and system_name == "point_mass":
        mass, k, c = sample_system_params(seed)
        print(f"  Sampled params: m={mass:.3f}, k={k:.3f}, c={c:.3f}")
    else:
        mass, k, c = None, None, None
        if randomize_system and system_name != "point_mass":
            print(f"  randomize_system is only defined for point_mass; using fixed {system_name} parameters.")
    
    # Sample modal parameters for the fixed-du variants (Q_modes, B, alpha)
    n = dx // 2
    Q_modes, B_modal, alpha = None, None, None
    if randomize_system and system_name == "point_mass" and fixed_du == 1:
        Q_modes, B_modal, alpha = sample_modal_params(seed, n, du=fixed_du)
    
    # Create system for this dimension (with optional sampled parameters)
    t0 = time.perf_counter()
    system, C_cost = create_nonlinear_system(
        system_name, dx, N=0, M=0, mass=mass, k=k, c=c, fixed_du=fixed_du,
        Q_modes=Q_modes, B_modal=B_modal, alpha=alpha,
        point_mass_gravity=point_mass_gravity,
        point_mass_gravity_diag=point_mass_gravity_diag,
        point_mass_gravity_margin=point_mass_gravity_margin,
        point_mass_gravity_type=point_mass_gravity_type,
        point_mass_integrator=point_mass_integrator,
        point_mass_linear_damping=point_mass_linear_damping,
        point_mass_q4_p=point_mass_q4_p,
        point_mass_q4_v=point_mass_q4_v,
        point_mass_r4_u=point_mass_r4_u,
    )
    timings["system_creation"] = time.perf_counter() - t0
    
    # Generate dataset
    t0 = time.perf_counter()
    x, u, x_plus, u_plus = generate_dataset(system, N=N, seed=seed, sampling_bounds=sampling_bounds)
    timings["dataset_generation"] = time.perf_counter() - t0
    
    # Compute costs
    L_xu = system.cost(x, u)
    
    # Compute polynomial features once for both methods
    t0 = time.perf_counter()
    z = np.concatenate([x, u], axis=1)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)
    
    # Use StateOnlyPolynomialFeatures for point mass systems (consistent with nonlinear pipeline)
    if exclude_u_squared and degree > 1:
        poly = FilteredPolynomialFeatures(degree=degree, include_bias=False, dx=dx, du=system.N_u)
    else:
        # Use StateOnlyPolynomialFeatures for all cases (works for degree 1 and higher)
        poly = StateOnlyPolynomialFeatures(degree=degree, include_bias=False, dx=dx, du=system.N_u)
    
    Z_all = np.concatenate([z, z_plus], axis=0)
    P_all = poly.fit_transform(Z_all)
    P_z = P_all[:N]
    P_z_next = P_all[N:]
    
    # Auxiliary samples for moment matching
    x_aux, u_aux = auxiliary_samples(system, M_offline, seed=seed, sampling_bounds=sampling_bounds)
    y_aux = np.concatenate([x_aux, u_aux], axis=1)
    P_y = poly.transform(y_aux)
    timings["feature_computation"] = time.perf_counter() - t0
    
    # Precompute symmetric constraint matrix once for both LPs
    t0 = time.perf_counter()
    A_sym, d_feat, d_sym = _build_sym_constraint_matrix(P_z, P_z_next, gamma)
    timings["constraint_matrix"] = time.perf_counter() - t0
    
    # Solve both methods using the same polynomial features and constraint matrix
    t0 = time.perf_counter()
    status_m2, status_m1, Q_learned, C_val, E_Q_learned, mu = solve_moment_matching_Q(
        P_z, P_z_next, P_y, L_xu, gamma, N, M_offline, seed,
        A_sym=A_sym, d=d_feat, d_sym=d_sym)
    timings["moment_matching_lp"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # if system.N_u == 1 and degree == 4:
    #     status_id = solve_gaussian_Q(P_z, P_z_next, L_xu, gamma, N, seed, dx, system.N_u,
    #                              A_sym=A_sym, d=d_feat, d_sym=d_sym,
    #                              include_bias=False)
    #     comparison_baseline = "gaussian_relevance"
    # else:
    #     status_id = solve_identity_Q(P_z, P_z_next, L_xu, gamma, N, seed, dx, system.N_u,
    #                                  A_sym=A_sym, d=d_feat, d_sym=d_sym)
    #     comparison_baseline = "identity"
    status_id = solve_identity_Q(P_z, P_z_next, L_xu, gamma, N, seed, dx, system.N_u,
                                     A_sym=A_sym, d=d_feat, d_sym=d_sym)
    comparison_baseline = "identity"
    timings["identity_lp"] = time.perf_counter() - t0
    
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

    if system_name == "duffing":
        result.update(system.metadata())
        result.update({
            "polynomial_degree": degree,
            "feature_type": "state_only_polynomial",
            "data_seed": seed,
            "auxiliary_sample_config": {
                "M_offline": M_offline,
                "seed": seed + 12345,
            },
            "solver_status": {
                "moment_matching_stage1": status_m1,
                "moment_matching_stage2": status_m2,
                "comparison_baseline": status_id,
                "comparison_baseline_name": comparison_baseline,
            },
            "boundedness_status": {
                "moment_matching": is_bounded(status_m2),
                "comparison_baseline": is_bounded(status_id),
            },
            "evaluation_horizon": eval_horizon,
        })

    if system_name == "point_mass":
        result["gravity_enabled"] = bool(getattr(system, "has_gravity", False))
        result["gravity_type"] = getattr(system, "gravity_type", normalize_point_mass_gravity_type(point_mass_gravity_type))
        result["integrator"] = getattr(system, "integrator", point_mass_integrator)
        result["linear_damping"] = float(getattr(system, "linear_damping", 0.0))
        result.update({
            "quartic_cost_enabled": bool(getattr(system, "has_quartic_cost", False)),
            "q4_p": system.q4_p.tolist(),
            "q4_v": system.q4_v.tolist(),
            "r4_u": system.r4_u.tolist(),
        })
        if result["gravity_enabled"]:
            K_minus_G = system.linearized_stiffness_minus_gravity()
            eigvals = np.linalg.eigvalsh(K_minus_G)
            result.update({
                "gravity_diag": system.gravity_diag.tolist(),
                "min_eig_K_minus_G": float(eigvals[0]),
                "max_eig_K_minus_G": float(eigvals[-1]),
            })
    
    # Monte Carlo policy evaluation (only when MM LP is bounded)
    if is_bounded(status_m2) and Q_learned is not None:
        t0 = time.perf_counter()
        eval_result = evaluate_Q_quality(
            Q_learned, system, dx, system.N_u, gamma, seed,
            plot_trajectories_dir=plot_trajectories_dir,
            poly=poly,
            eval_horizon=eval_horizon,
        )
        timings["mc_policy_eval"] = time.perf_counter() - t0
        if eval_result is not None:
            result.update(eval_result)
    
    # Store sampled system parameters if randomization was used
    if randomize_system and system_name == "point_mass":
        result["mass"] = float(system.m)
        result["c"] = float(system.c)
        if fixed_du == 1:
            n = dx // 2
            alpha_val = alpha if alpha is not None else 2.0
            result["k0"] = float(system.m * OMEGA_MAX_POINT_MASS**2 / (n ** alpha_val))
            result["alpha"] = float(alpha_val)
        else:
            result["k"] = float(system.k)
    
    # Store timing metrics
    timings["total"] = time.perf_counter() - t_total_start
    result["timings"] = timings    
    return result

def sweep_over_dims(dims, seeds, N, gamma, M_offline, degree, exclude_u_squared=False,
                    randomize_system=False, fixed_du=0, plot_trajectories_dir=None,
                    system_name="point_mass", duffing_params=None, sampling_bounds=None,
                    eval_horizon=20000, point_mass_gravity=False,
                    point_mass_gravity_diag=None, point_mass_gravity_margin=0.1,
                    point_mass_gravity_type="linear", point_mass_integrator="rk4",
                    point_mass_linear_damping=0.0,
                    point_mass_q4_p=0.0, point_mass_q4_v=0.0, point_mass_r4_u=0.0):
    system_name = normalize_system_name(system_name)
    results = []
    for dx in dims:
        if system_name == "duffing" and dx % 2 != 0:
            print(f"[WARN] Skipping dx={dx} (must be even for Duffing family)")
            continue
        if system_name == "point_mass" and dx % 2 != 0:
            print(f"[WARN] Skipping dx={dx} (must be even for point mass system)")
            continue
        for s in seeds:
            print(f"Running {system_name} seed {s} for dx={dx}")
            try:
                results.append(run_one(
                    seed=int(s), dx=int(dx), N=N, gamma=gamma, M_offline=M_offline,
                    degree=degree, exclude_u_squared=exclude_u_squared,
                    randomize_system=randomize_system, fixed_du=fixed_du,
                    plot_trajectories_dir=plot_trajectories_dir,
                    system_name=system_name, duffing_params=duffing_params,
                    sampling_bounds=sampling_bounds, eval_horizon=eval_horizon,
                    point_mass_gravity=point_mass_gravity,
                    point_mass_gravity_diag=point_mass_gravity_diag,
                    point_mass_gravity_margin=point_mass_gravity_margin,
                    point_mass_gravity_type=point_mass_gravity_type,
                    point_mass_integrator=point_mass_integrator,
                    point_mass_linear_damping=point_mass_linear_damping,
                    point_mass_q4_p=point_mass_q4_p,
                    point_mass_q4_v=point_mass_q4_v,
                    point_mass_r4_u=point_mass_r4_u,
                ))
            except Exception as e:
                print(f"[WARN] Failed for dx={dx}, seed={s}: {e}. Skipping.")
                continue
    return results

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Boundedness vs state dimension (nonlinear systems).")
    parser.add_argument("--system", type=str, default="point_mass",
                        help="Nonlinear benchmark system: point_mass (default) or duffing.")
    parser.add_argument("--dims", type=str, default="2,4,6,8,10", help="Comma-separated list of state dimensions (e.g., '2,4,6,8,10'). Default: 2,4,6,8,10")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per dimension (default: 10)")
    parser.add_argument("--N", type=int, default=1000, help="Number of samples for dataset (default: 1000)")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor (default: from config)")
    parser.add_argument("--M_offline", type=int, default=500, help="Offline pool size (default: 500)")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial feature degree (default: 1)")
    parser.add_argument("--eval_horizon", type=int, default=20000, help="Evaluation horizon for closed-loop rollout cost (default: 20000)")
    parser.add_argument("--exclude_u_squared", action="store_true", help="Exclude u^2 terms from polynomial features (for degree 2)")
    parser.add_argument("--randomize_system", action="store_true", help="Sample m, k, c from LogNormal distributions per seed (default: use fixed values)")
    parser.add_argument("--fixed_du", type=int, default=1, help="Fixed control dimension: 0 = fully actuated (du=n), 1 = du=1. Default: 1.")
    parser.add_argument("--point_mass_gravity", action="store_true",
                        help="For the point-mass cubic-drag system, add the optional +G h(p) gravity term.")
    parser.add_argument("--point_mass_gravity_type", choices=("linear", "sin", "tanh", "log"), default="linear",
                        help="Point-mass gravity nonlinearity: linear uses p; sin uses sin(p); tanh uses tanh(p); log uses asinh(p). Default: linear.")
    parser.add_argument("--point_mass_gravity_diag", type=str, default=None,
                        help="Scalar or comma-separated diagonal entries for G. Supplying this also enables gravity.")
    parser.add_argument("--point_mass_gravity_margin", type=float, default=0.1,
                        help="Auto-select G as lambda_min(K)+margin when --point_mass_gravity is used without an explicit diagonal.")
    parser.add_argument("--point_mass_integrator", choices=("euler", "rk4"), default="rk4",
                        help="Point-mass integrator for generated samples and rollouts. Use rk4 to avoid forward-Euler energy injection near weakly damped equilibria. Default: rk4.")
    parser.add_argument("--point_mass_linear_damping", type=float, default=0.0,
                        help="Optional point-mass viscous damping coefficient d in -d v. This does not move equilibria, but makes stable equilibria linearly attracting. Default: 0.")
    parser.add_argument("--point_mass_q4_p", type=str, default="0.0",
                        help="Scalar or comma-separated weights for quartic position cost sum_i q4_p_i p_i^4. Default: 0.")
    parser.add_argument("--point_mass_q4_v", type=str, default="0.0",
                        help="Scalar or comma-separated weights for quartic velocity cost sum_i q4_v_i v_i^4. Default: 0.")
    parser.add_argument("--point_mass_r4_u", type=str, default="0.0",
                        help="Scalar or comma-separated weights for quartic input cost sum_j r4_u_j u_j^4. Default: 0.")
    parser.add_argument("--duffing_delta", type=float, default=0.2, help="Duffing damping delta (default: 0.2)")
    parser.add_argument("--duffing_alpha", type=float, default=1.0, help="Duffing linear stiffness alpha (default: 1.0)")
    parser.add_argument("--duffing_beta", type=float, default=1.0, help="Duffing cubic stiffness beta (default: 1.0)")
    parser.add_argument("--duffing_stiffness_growth", type=float, default=2.0, help="Duffing modal stiffness growth exponent for dx>2 (default: 2.0)")
    parser.add_argument("--duffing_cubic_coupling", choices=("diagonal", "radial", "modal"), default="diagonal",
                        help="Duffing cubic stiffness coupling: diagonal q_i^3, radial ||q||^2 q, or modal Q(Q^Tq)^3.")
    parser.add_argument("--duffing_dt", type=float, default=0.05, help="Duffing discretization step (default: 0.05)")
    parser.add_argument("--duffing_q1", type=float, default=1.0, help="Duffing x1^2 cost weight (default: 1.0)")
    parser.add_argument("--duffing_q2", type=float, default=0.1, help="Duffing x2^2 cost weight (default: 0.1)")
    parser.add_argument("--duffing_q4", type=float, default=0.05, help="Duffing x1^4 cost weight; set to 0 for quadratic cost (default: 0.05)")
    parser.add_argument("--duffing_r", type=float, default=0.01, help="Duffing control cost weight (default: 0.01)")
    parser.add_argument("--duffing_normalize_cost", action="store_true",
                        help="Divide Duffing state costs by the number of oscillators; control cost is unchanged.")
    parser.add_argument("--duffing_state_bounds", type=str, default="-2.5,2.5;-2.5,2.5",
                        help="Duffing state sampling bounds as 'x1_low,x1_high;x2_low,x2_high'.")
    parser.add_argument("--duffing_u_bounds", type=str, default="-5,5",
                        help="Duffing control sampling bounds as 'u_low,u_high'.")
    parser.add_argument("--plot_trajectories", action="store_true", help="Save trajectory comparison plots (MM vs LQR) for each bounded run")
    parser.add_argument("--out_json", type=str, default=None, help="Raw results JSON (if None, auto-generated with N)")
    parser.add_argument("--plot_dir", type=str, default=None, help="Plot filename (if None, auto-generated with N)")
    args = parser.parse_args()
    args.system = normalize_system_name(args.system)
    if args.fixed_du not in (0, 1):
        raise ValueError(f"--fixed_du must be 0 or 1 (got {args.fixed_du})")
    args.point_mass_gravity = args.point_mass_gravity or args.point_mass_gravity_diag is not None
    args.point_mass_gravity_type = normalize_point_mass_gravity_type(args.point_mass_gravity_type)
    
    # Parse dimensions
    if args.dims.strip():
        dims = [int(s) for s in args.dims.split(",") if s.strip()]
    else:
        dims = [2, 4, 6, 8, 10]  # Default: dx = 2, 4, 6, 8, 10
    
    # Generate seeds for each experiment
    seeds = list(range(0, args.seeds + 0))
    
    # Auto-generate filenames with concise experiment metadata.
    # Point-mass runs always use a fixed input dimension in the current shell
    # sweep, and "plots" is historical noise in these JSON names. Keep the
    # gravity choice and M_offline explicit because they change the experiment.
    point_mass_gravity_tag = (
        f"gravity_{args.point_mass_gravity_type}" if args.point_mass_gravity else "gravity_off"
    )
    point_mass_has_quartic_cost = (
        _has_positive_nonnegative_values(args.point_mass_q4_p, "point_mass_q4_p")
        or _has_positive_nonnegative_values(args.point_mass_q4_v, "point_mass_q4_v")
        or _has_positive_nonnegative_values(args.point_mass_r4_u, "point_mass_r4_u")
    )
    point_mass_cost_tag = "cost_quartic" if point_mass_has_quartic_cost else "cost_quadratic"
    point_mass_tag = (
        f"point_mass_N_{args.N}_M_{args.M_offline}_degree_{args.degree}_"
        f"{point_mass_gravity_tag}_{point_mass_cost_tag}_{args.point_mass_integrator}_"
        f"damp_{args.point_mass_linear_damping:g}"
    )
    if args.out_json is None and args.system == "point_mass":
        args.out_json = f"bounded_vs_dim_results_nonlinear_{point_mass_tag}.json"
    elif args.out_json is None:
        args.out_json = f"bounded_vs_dim_results_nonlinear_{args.system}_N_{args.N}_degree_{args.degree}.json"
    if args.plot_dir is None:
        if args.system == "point_mass":
            args.plot_dir = f"../figures/bounded_vs_dim_percentages_nonlinear_{point_mass_tag}.pdf"
        else:
            args.plot_dir = f"../figures/bounded_vs_dim_percentages_nonlinear_{args.system}_N_{args.N}_degree_{args.degree}.pdf"
    
    # Set up trajectory plotting directory if requested
    traj_dir = None
    if args.plot_trajectories:
        if args.system == "point_mass":
            traj_dir = f"../figures/trajectories_{point_mass_tag}"
        else:
            traj_dir = f"../figures/trajectories_{args.system}_N_{args.N}_degree_{args.degree}"

    duffing_params = {
        "delta": args.duffing_delta,
        "alpha": args.duffing_alpha,
        "beta": args.duffing_beta,
        "stiffness_growth": args.duffing_stiffness_growth,
        "cubic_coupling": args.duffing_cubic_coupling,
        "delta_t": args.duffing_dt,
        "q1": args.duffing_q1,
        "q2": args.duffing_q2,
        "q4": args.duffing_q4,
        "rho": args.duffing_r,
        "normalize_cost": args.duffing_normalize_cost,
        "gamma": args.gamma,
    }
    sampling_bounds = {
        "state_bounds": parse_bounds_pair(args.duffing_state_bounds, expected_dim=2),
        "u_bounds": parse_bounds_pair(args.duffing_u_bounds, expected_dim=1)[0],
    }
    
    results = sweep_over_dims(dims, seeds, N=args.N, gamma=args.gamma, M_offline=args.M_offline,
                              degree=args.degree, exclude_u_squared=args.exclude_u_squared,
                              randomize_system=args.randomize_system, fixed_du=args.fixed_du,
                              plot_trajectories_dir=traj_dir, system_name=args.system,
                              duffing_params=duffing_params, sampling_bounds=sampling_bounds,
                              eval_horizon=args.eval_horizon,
                              point_mass_gravity=args.point_mass_gravity,
                              point_mass_gravity_diag=args.point_mass_gravity_diag,
                              point_mass_gravity_margin=args.point_mass_gravity_margin,
                              point_mass_gravity_type=args.point_mass_gravity_type,
                              point_mass_integrator=args.point_mass_integrator,
                              point_mass_linear_damping=args.point_mass_linear_damping,
                              point_mass_q4_p=args.point_mass_q4_p,
                              point_mass_q4_v=args.point_mass_q4_v,
                              point_mass_r4_u=args.point_mass_r4_u)
    
    if len(results) == 0:
        print("No results (no valid dimensions or files). Exiting.")
        raise SystemExit(0)
    
    df = pd.DataFrame(results)
    
    # Boundedness aggregation
    agg = df.groupby("dx").agg(
        moment_bounded_mean=("moment_matching_bounded", "mean"),
        identity_bounded_mean=("identity_bounded", "mean"),
        n=("seed", "count"),
    ).reset_index()
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_mean"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_mean"]
    
    # MC policy cost aggregation (only for bounded MM runs)
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
    if args.system == "point_mass":
        agg_json_name = f"bounded_vs_dim_percentages_nonlinear_{point_mass_tag}.json"
    else:
        agg_json_name = f"bounded_vs_dim_percentages_nonlinear_{args.system}_N_{args.N}_degree_{args.degree}.json"
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
    
    # --- Timing summary ---
    timing_keys = ["system_creation", "dataset_generation", "feature_computation",
                   "constraint_matrix", "moment_matching_lp", "identity_lp", "mc_policy_eval", "total"]
    # Expand nested timings dict into flat columns
    for key in timing_keys:
        df[f"t_{key}"] = df["timings"].apply(lambda d: d.get(key, float("nan")) if isinstance(d, dict) else float("nan"))
    timing_agg = df.groupby("dx")[[f"t_{k}" for k in timing_keys]].mean().reset_index()
    print("\n=== Average Timing per (dx, seed) [seconds] ===")
    print(timing_agg.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    
    print("\nSaved:", args.out_json)
    print("Aggregated results saved:", agg_json_name)
