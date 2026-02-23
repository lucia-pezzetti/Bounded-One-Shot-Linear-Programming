
import json
import numpy as np
import cvxpy as cp
import sys
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve_discrete_are
from scipy import sparse
from feature_scaling import FilteredPolynomialFeatures
from dynamical_systems import dlqr

# Add parent directory to path to import from data module
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.load_systems import load_system, get_available_dimensions

# -------------------------
# Config (defaults; can be overridden by CLI)
# -------------------------
degree = 1           # polynomial feature degree
M_offline = 500      # auxiliary pool size for building P_y

# Sampling bounds for generating (x,u) datasets
# (Assumed symmetric for all dimensions)
x_bound = 3.0        # states sampled in [-x_bound, x_bound]
u_bound = 1.0        # inputs sampled in [-u_bound, u_bound]

du = 2               # fixed input dimension

def stage_cost(x: np.ndarray, u: np.ndarray, C: np.ndarray, rho):
    """
    Quadratic cost: l(x,u) = x^T L_x x + u^T L_u u where L = diag(C.T @ C, 10^{-4})
    x: (N, dx), u: (N, du) -> returns (N,)
    """
    dx = x.shape[1]
    du = u.shape[1]
    
    # State cost: x^T (C.T @ C) x = ||C x||^2
    xLx = np.einsum("ni,ij,nj->n", x, C.T @ C, x, optimize=True)
    # Control cost: u^T (10^{-4} * I) u = 10^{-4} * ||u||^2
    uLu = rho * np.einsum("ni,ij,nj->n", u, np.eye(du), u, optimize=True)
    return xLx + uLu

def generate_dataset(A, B, N, dx, du, seed=0):
    """
    Generate random (x,u) and compute x_next = A x + B u (no noise).
    Returns x, u, x_plus, u_plus (N samples).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-x_bound, x_bound, size=(N, dx))
    u = rng.uniform(-u_bound, u_bound, size=(N, du))
    x_plus = (A @ x.T + B @ u.T).T
    # Use u_plus = u (consistent dimensionality for feature construction)
    u_plus = rng.uniform(-u_bound, u_bound, size=(N, du))
    return x, u, x_plus, u_plus

def auxiliary_samples(M, dx, du, seed=0):
    """Auxiliary pool for building P_y"""
    rng = np.random.default_rng(seed + 12345)
    x_aux = rng.uniform(-x_bound, x_bound, size=(M, dx))
    u_aux = rng.uniform(-u_bound, u_bound, size=(M, du))
    return x_aux, u_aux


def compute_optimal_Q(A, B, C_cost, gamma, rho):
    """
    Compute the optimal Q matrix for the LQR system using the discrete algebraic Riccati equation.
    
    Args:
        A: State transition matrix (dx, dx)
        B: Control input matrix (dx, du) 
        C_cost: Cost matrix for states (dx, dx)
        gamma: Discount factor
        
    Returns:
        Q_optimal: Optimal Q matrix in feature space (d, d)
    """
    dx = A.shape[0]
    du = B.shape[1]
    
    # Define cost matrices for LQR: L = diag(C.T @ C, rho)
    Q_lqr = C_cost.T @ C_cost         # State cost matrix: C.T @ C
    R_lqr = rho * np.eye(du)  # Control cost matrix: rho * I_{d_u}
    
    # Solve discrete algebraic Riccati equation
    # Scale A and R for discount factor
    Ad = np.sqrt(gamma) * A
    Rd = R_lqr / gamma
    
    P = solve_discrete_are(Ad, B, Q_lqr, Rd)
    
    # Compute optimal Q-function components
    Qxx = Q_lqr + gamma * (A.T @ P @ A)
    Quu = R_lqr + gamma * (B.T @ P @ B)
    Qxu = gamma * (A.T @ P @ B)
    
    # Build optimal Q matrix in state-action space
    Q_optimal_sa = np.block([
        [Qxx,    Qxu],
        [Qxu.T,  Quu]
    ])

    return Q_optimal_sa


def extract_mm_policy_analytical_degree1(Q_learned, dx, du, u_bounds):
    """
    Extract moment matching policy analytically for degree 1 polynomial features.
    
    For degree 1, features are φ = [x1, x2, ..., x_dx, u1, u2, ..., u_du]
    Q-function: φ^T Q φ = x^T Q_xx x + 2x^T Q_xu u + u^T Q_uu u
    Optimal u: ∂/∂u = 0 → 2Q_xu^T x + 2Q_uu u = 0 → u = -Q_uu^{-1} Q_xu^T x
    
    Args:
        Q_learned: Learned Q matrix (dx+du, dx+du) in feature space
        dx: State dimension
        du: Input dimension
        u_bounds: Tuple (u_min, u_max) for action clipping
        
    Returns:
        Policy function that takes state x and returns optimal action u
    """
    # Partition Q matrix
    Q_xx = Q_learned[:dx, :dx]          # State-state interactions
    Q_xu = Q_learned[:dx, dx:]          # State-action interactions
    Q_uu = Q_learned[dx:, dx:]          # Action-action interactions
    
    # Pre-compute Q_uu inverse
    Q_uu_inv = np.linalg.inv(Q_uu)
    
    # Pre-compute gain matrix: K = -Q_uu^{-1} Q_xu^T
    K_mm = -Q_uu_inv @ Q_xu.T
    
    def mm_policy(x):
        """
        Moment matching policy: u = K_mm @ x
        
        Args:
            x: State vector, shape (dx,) or (n, dx)
            
        Returns:
            u: Action vector, shape (du,) or (n, du)
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Compute optimal action: u = K_mm @ x^T
        u = (K_mm @ x.T).T  # Shape: (n, du)
        
        # Clip to bounds
        u = np.clip(u, u_bounds[0], u_bounds[1])
        
        # Return format: scalar for single state/single control, array otherwise
        if u.shape[0] == 1 and u.shape[1] == 1:
            return float(u[0, 0])
        elif u.shape[0] == 1:
            return u[0]
        elif u.shape[1] == 1:
            return u.flatten()
        else:
            return u
    
    return mm_policy


def compare_policy_costs(A, B, C_cost, rho, gamma, Q_learned, poly, dx, du, n_test_states=100, horizon=1000, seed=0):
    """
    Compare online costs of moment matching policy vs LQR policy.
    
    Args:
        A, B, C_cost: System matrices
        rho: Control cost weight
        gamma: Discount factor
        Q_learned: Learned Q matrix from moment matching (in feature space)
        poly: PolynomialFeatures object for feature transformation
        dx: State dimension
        du: Input dimension
        n_test_states: Number of test initial states
        horizon: Simulation horizon
        seed: Random seed for test states
        
    Returns:
        dict: Comparison metrics including normalized cost difference
    """
    if Q_learned is None:
        return {"error": "Q_learned is None"}

    # Create LQR system
    system = dlqr(A, B, C_cost, rho, gamma)
    
    # Extract LQR policy directly (reimplemented from PolicyExtractor)
    P_lqr, K_lqr, q_lqr = system.optimal_solution()
    
    # Create LQR policy function
    def lqr_policy(x):
        """
        LQR policy: u = -K @ x, clipped to bounds.
        
        Args:
            x: State vector, shape (dx,) or (n, dx)
            
        Returns:
            u: Control action, shape (du,) or (n, du)
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Standard LQR control law: u = -K @ x
        u = -K_lqr @ x.T  # Shape: (du, n)
        u = u.T  # Shape: (n, du)
        # Clip to bounds (element-wise for vector controls)
        u = np.clip(u, -u_bound, u_bound)
        # Return format: scalar for single state/single control, array otherwise
        if u.shape[0] == 1 and u.shape[1] == 1:
            return float(u[0, 0])
        elif u.shape[0] == 1:
            return u[0]  # Shape: (du,)
        elif u.shape[1] == 1:
            return u.flatten()  # Shape: (n,)
        else:
            return u  # Shape: (n, du)
    
    # Extract moment matching policy analytically (degree 1)
    mm_policy = extract_mm_policy_analytical_degree1(Q_learned, dx, du, (-u_bound, u_bound))
    
    # Scale number of test states with dimension for better coverage
    n_test_states = 100 * int(dx)
    
    # Generate test initial states
    rng = np.random.default_rng(seed + 2000)
    test_states = rng.uniform(-x_bound, x_bound, size=(n_test_states, dx))
    
    # Simulate both policies and compute costs
    lqr_costs = []
    mm_costs = []
    lqr_success = []
    mm_success = []
    
    for i in range(n_test_states):
        x0 = test_states[i]
        
        # Simulate LQR policy
        x_current = x0.copy()
        lqr_cost = 0.0
        lqr_succ = True
        for step in range(horizon):
            u = lqr_policy(x_current)
            # Convert to numpy array and ensure proper shape
            u = np.asarray(u)
            if u.ndim == 0:
                u = u.reshape(1)  # Scalar -> (1,)
            elif u.ndim > 1:
                u = u.flatten()  # Flatten if needed
            
            # Check finiteness
            if not np.isfinite(u).all():
                lqr_succ = False
                break
            
            # Clip to bounds (element-wise for vector controls)
            u = np.clip(u, -u_bound, u_bound)
            
            # Ensure u has shape (du,) for system.step and (1, du) for stage_cost
            u_for_step = u if u.shape == (du,) else u.reshape(du)
            u_for_cost = u.reshape(1, -1) if u.ndim == 1 else u
            
            # Compute cost
            cost = stage_cost(x_current.reshape(1, -1), u_for_cost, C_cost, rho)[0]
            if not np.isfinite(cost):
                print(f"lqr_cost is not finite")
                lqr_succ = False
                break
            lqr_cost += (gamma ** step) * cost
            
            # Update state
            x_current = system.step(x_current, u_for_step)
            
            # Check for convergence (close to origin)
            if np.linalg.norm(x_current) < 1e-3:
                break
        
        lqr_costs.append(lqr_cost if lqr_succ else np.inf)
        lqr_success.append(lqr_succ)
    
    for i in range(n_test_states):
        x0 = test_states[i]
        
        # Simulate MM policy
        x_current = x0.copy()
        mm_cost = 0.0
        mm_succ = True
        for step in range(horizon):
            u = mm_policy(x_current)
            # Convert to numpy array and ensure proper shape
            u = np.asarray(u)
            if u.ndim == 0:
                u = u.reshape(1)  # Scalar -> (1,)
            elif u.ndim > 1:
                u = u.flatten()  # Flatten if needed
            
            # Check finiteness
            if not np.isfinite(u).all():
                mm_succ = False
                break
            
            # Clip to bounds (element-wise for vector controls)
            u = np.clip(u, -u_bound, u_bound)
            
            # Ensure u has shape (du,) for system.step and (1, du) for stage_cost
            u_for_step = u if u.shape == (du,) else u.reshape(du)
            u_for_cost = u.reshape(1, -1) if u.ndim == 1 else u
            
            # Compute cost
            cost = stage_cost(x_current.reshape(1, -1), u_for_cost, C_cost, rho)[0]
            if not np.isfinite(cost):
                print(f"mm_cost is not finite")
                mm_succ = False
                break
            mm_cost += (gamma ** step) * cost
            
            # Update state
            x_current = system.step(x_current, u_for_step)
            
            # Check for convergence (close to origin)
            if np.linalg.norm(x_current) < 1e-3:
                break
        
        mm_costs.append(mm_cost if mm_succ else np.inf)
        mm_success.append(mm_succ)
    
    # Compute comparison metrics
    metrics = {}
    
    # Only compare cases where both policies succeeded
    both_success_indices = [i for i in range(n_test_states) if lqr_success[i] and mm_success[i]]
    
    if len(both_success_indices) > 0:
        lqr_costs_both = [lqr_costs[i] for i in both_success_indices]
        mm_costs_both = [mm_costs[i] for i in both_success_indices]
        
        lqr_costs_arr = np.array(lqr_costs_both)
        mm_costs_arr = np.array(mm_costs_both)
        
        # Normalized cost difference: (MM_cost - LQR_cost) / LQR_cost
        cost_diff = mm_costs_arr - lqr_costs_arr
        normalized_diff = cost_diff / (lqr_costs_arr + 1e-8)
        # print(f"normalized_diff: {normalized_diff}")
        
        metrics["policy_cost_diff_mean"] = np.mean(cost_diff)
        metrics["policy_cost_diff_std"] = np.std(cost_diff)
        metrics["policy_cost_normalized_diff_mean"] = np.mean(normalized_diff)
        metrics["policy_cost_normalized_diff_std"] = np.std(normalized_diff)
        metrics["policy_cost_lqr_mean"] = np.mean(lqr_costs_arr)
        metrics["policy_cost_mm_mean"] = np.mean(mm_costs_arr)
        metrics["policy_cost_n_comparisons"] = len(lqr_costs_both)
    else:
        # Store NaN for metrics when no successful comparisons
        # This ensures the columns exist in the dataframe for aggregation
        metrics["policy_cost_diff_mean"] = np.nan
        metrics["policy_cost_diff_std"] = np.nan
        metrics["policy_cost_normalized_diff_mean"] = np.nan
        metrics["policy_cost_normalized_diff_std"] = np.nan
        metrics["policy_cost_lqr_mean"] = np.nan
        metrics["policy_cost_mm_mean"] = np.nan
        metrics["policy_cost_n_comparisons"] = 0
    
    metrics["policy_cost_lqr_success_rate"] = np.mean(lqr_success) if len(lqr_success) > 0 else 0.0
    metrics["policy_cost_mm_success_rate"] = np.mean(mm_success) if len(mm_success) > 0 else 0.0
    metrics["policy_cost_n_test_states"] = n_test_states
    
    # -------------------------
    # Value function comparison
    # V(x) = min_u Q(x, u)
    # For LQR: V_lqr(x) = x^T P_lqr x
    # For MM (degree 1): V_mm(x) = x^T (Q_xx - Q_xu Q_uu^{-1} Q_xu^T) x
    # -------------------------
    # Extract blocks from Q_learned for degree 1: features are [x1, ..., x_dx, u1, ..., u_du]
    Q_xx = Q_learned[:dx, :dx]
    Q_xu = Q_learned[:dx, dx:]
    Q_ux = Q_learned[dx:, :dx]
    Q_uu = Q_learned[dx:, dx:]
    
    # Compute MM value function matrix: P_mm = Q_xx - Q_xu Q_uu^{-1} Q_xu^T
    Q_uu_inv = np.linalg.inv(Q_uu)
    
    P_mm = Q_xx - Q_xu @ Q_uu_inv @ Q_xu.T
    
    # Evaluate value functions at all test states
    V_lqr_values = np.array([test_states[i] @ P_lqr @ test_states[i] for i in range(n_test_states)])
    V_mm_values = np.array([test_states[i] @ P_mm @ test_states[i] for i in range(n_test_states)])
    
    # Compute statistics
    V_lqr_mean = np.mean(V_lqr_values)
    V_mm_mean = np.mean(V_mm_values)
    
    # Absolute difference
    V_diff = V_mm_values - V_lqr_values
    V_diff_mean = np.mean(V_diff)
    V_diff_std = np.std(V_diff)
    
    # Normalized difference: (V_mm - V_lqr) / V_lqr
    V_normalized_diff = V_diff / (V_lqr_values + 1e-8)
    if V_lqr_values.any() < 1:
        print(f"V_lqr_values: {V_lqr_values}")
        print(f"V_mm_values: {V_mm_values}")
        print(f"V_diff: {V_diff}")
        print(f"V_normalized_diff: {V_normalized_diff}")
        raise Exception("V_lqr_values is less than 1")
    # print(f"V_normalized_diff: {V_normalized_diff}")
    V_normalized_diff_mean = np.mean(V_normalized_diff)
    V_normalized_diff_std = np.std(V_normalized_diff)
    
    # Store value function metrics
    metrics["value_func_lqr_mean"] = V_lqr_mean
    metrics["value_func_mm_mean"] = V_mm_mean
    metrics["value_func_diff_mean"] = V_diff_mean
    metrics["value_func_diff_std"] = V_diff_std
    metrics["value_func_normalized_diff_mean"] = V_normalized_diff_mean
    metrics["value_func_normalized_diff_std"] = V_normalized_diff_std
    
    # Also store the matrix norm difference between P_lqr and P_mm
    P_diff_frob = np.linalg.norm(P_mm - P_lqr, 'fro')
    P_diff_normalized = P_diff_frob / (np.linalg.norm(P_lqr, 'fro') + 1e-8)
    metrics["value_matrix_diff_frob"] = P_diff_frob
    metrics["value_matrix_diff_normalized"] = P_diff_normalized
    
    # -------------------------
    # K matrix comparison (gain matrices)
    # LQR: u = -K_lqr @ x
    # MM:  u = K_mm @ x where K_mm = -Q_uu^{-1} @ Q_xu^T
    # -------------------------
    # Compute K_mm from Q_learned
    K_mm = -Q_uu_inv @ Q_xu.T
    
    # Compute relative difference between K matrices
    # Note: LQR uses u = -K_lqr @ x, so we compare K_mm with -K_lqr
    K_diff = K_mm - (-K_lqr)  # K_mm vs -K_lqr (both have same sign convention for u = K @ x)
    K_diff_frob = np.linalg.norm(K_diff, 'fro')
    K_lqr_frob = np.linalg.norm(K_lqr, 'fro')
    K_diff_normalized = K_diff_frob / (K_lqr_frob + 1e-8)
    
    metrics["K_diff_frob"] = K_diff_frob
    metrics["K_diff_normalized"] = K_diff_normalized
    metrics["K_lqr_frob"] = K_lqr_frob
    metrics["K_mm_frob"] = np.linalg.norm(K_mm, 'fro')
    
    # -------------------------
    # Closed-loop stability check for MM policy
    # For discrete-time: stable iff all eigenvalues of (A + B @ K_mm) have |λ| < 1
    # -------------------------
    A_cl_mm = A + B @ K_mm  # Closed-loop system matrix with MM policy
    eigs_mm = np.linalg.eigvals(A_cl_mm)
    spectral_radius_mm = np.max(np.abs(eigs_mm))
    is_stable_mm = spectral_radius_mm < 1.0
    
    # Also check LQR stability (should always be stable)
    A_cl_lqr = A - B @ K_lqr  # Note: LQR uses u = -K_lqr @ x
    eigs_lqr = np.linalg.eigvals(A_cl_lqr)
    spectral_radius_lqr = np.max(np.abs(eigs_lqr))
    is_stable_lqr = spectral_radius_lqr < 1.0
    
    metrics["mm_spectral_radius"] = spectral_radius_mm
    metrics["mm_is_stable"] = is_stable_mm
    metrics["lqr_spectral_radius"] = spectral_radius_lqr
    metrics["lqr_is_stable"] = is_stable_lqr
    
    # Save complete K matrices (convert to nested lists for JSON serialization)
    # Note: LQR effective gain is -K_lqr, MM effective gain is K_mm
    metrics["K_lqr_matrix"] = K_lqr.tolist()  # Shape: (du, dx)
    metrics["K_mm_matrix"] = K_mm.tolist()    # Shape: (du, dx)
    
    # Also save the closed-loop eigenvalues for debugging
    metrics["mm_eigenvalues_real"] = np.real(eigs_mm).tolist()
    metrics["mm_eigenvalues_imag"] = np.imag(eigs_mm).tolist()
    metrics["lqr_eigenvalues_real"] = np.real(eigs_lqr).tolist()
    metrics["lqr_eigenvalues_imag"] = np.imag(eigs_lqr).tolist()
    
    if not is_stable_mm:
        print(f"    WARNING: MM closed-loop is UNSTABLE (spectral radius = {spectral_radius_mm:.4f})")
        # Print the eigenvalues for debugging
        print(f"    MM eigenvalues magnitudes: {np.abs(eigs_mm)}")
    
    return metrics


def compare_Q_matrices(Q_learned, Q_optimal, test_samples=None, poly=None, dx=None, du=None, C_val=None):
    """
    Compare the learned Q matrix with the optimal Q matrix using trace difference and Q-function evaluation.
    
    Args:
        Q_learned: Learned Q matrix from moment matching (in scaled feature space)
        Q_optimal: Optimal Q matrix from LQR theory (in original state-action space)
        test_samples: Optional test samples (x, u) for Q-function evaluation
        poly: PolynomialFeatures object for feature transformation
        dx: State dimension
        du: Input dimension
        C_val: Covariance matrix from MM stage 1 (C_val = Σ mu_i * y_i y_i^T)
    Returns:
        dict: Comparison metrics
    """
    if Q_learned is None or Q_optimal is None:
        return {"error": "One or both Q matrices are None"}
    
    metrics = {}
    
    # Trace difference - the matrices are in the same feature space since degree=1
    trace_diff_abs = np.trace(Q_learned) - np.trace(Q_optimal)
    
    # Normalize trace error by the scale of the optimal Q matrix entries
    # Better options for trace-specific normalization:
    # 1. Trace of optimal Q: |trace(Q_learned) - trace(Q_optimal)| / |trace(Q_optimal)|
    # 2. Mean absolute value of optimal Q: |trace_diff| / mean(|Q_optimal|)
    # 3. Max absolute value of optimal Q: |trace_diff| / max(|Q_optimal|)
    # 4. Frobenius norm (current): |trace_diff| / ||Q_optimal||_F
    
    # Option 1: Normalize by trace of optimal Q (most direct for trace error)
    trace_optimal = np.trace(Q_optimal)
    if abs(trace_optimal) > 1e-10:  # Avoid division by zero
        metrics["trace_diff"] = trace_diff_abs / trace_optimal
        metrics["trace_diff_abs"] = trace_diff_abs  # Keep absolute value for reference
    else:
        # Fallback: normalize by mean absolute value of matrix entries
        mean_abs_optimal = np.mean(Q_optimal)
        if mean_abs_optimal > 1e-10:
            metrics["trace_diff"] = trace_diff_abs / mean_abs_optimal
            metrics["trace_diff_abs"] = trace_diff_abs
        else:
            metrics["trace_diff"] = trace_diff_abs  # Final fallback to absolute
            metrics["trace_diff_abs"] = trace_diff_abs
    
    # Weighted trace comparison using C_val from MM LP (the direction of optimization)
    # This compares trace(C_val @ Q_learned) vs trace(C_val @ Q_optimal)
    if C_val is not None:
        weighted_trace_learned = np.trace(C_val @ Q_learned)
        weighted_trace_optimal = np.trace(C_val @ Q_optimal)
        weighted_trace_diff_abs = weighted_trace_learned - weighted_trace_optimal
        
        # Normalize by the optimal weighted trace
        if abs(weighted_trace_optimal) > 1e-10:
            metrics["weighted_trace_diff"] = weighted_trace_diff_abs / weighted_trace_optimal
            metrics["weighted_trace_diff_abs"] = weighted_trace_diff_abs
        else:
            metrics["weighted_trace_diff"] = weighted_trace_diff_abs
            metrics["weighted_trace_diff_abs"] = weighted_trace_diff_abs
        
        metrics["weighted_trace_learned"] = weighted_trace_learned
        metrics["weighted_trace_optimal"] = weighted_trace_optimal

    # Q-function evaluation on test samples
    if test_samples is not None and poly is not None:
        x_test, u_test = test_samples
        
        # Transform test samples to polynomial features
        z_test = np.concatenate([x_test, u_test], axis=1)
        phi_test = poly.transform(z_test)
        
        # Evaluate learned Q-function (in scaled feature space)
        Q_learned_values = np.array([phi_test[i] @ Q_learned @ phi_test[i] 
                                   for i in range(len(phi_test))])
        
        # Evaluate optimal Q-function (in original state-action space)
        Q_optimal_values = np.array([z_test[i] @ Q_optimal @ z_test[i] 
                                   for i in range(len(z_test))])
        
        # Compute comparison metrics
        metrics["q_value_diff_mean"] = np.mean(Q_learned_values - Q_optimal_values)
        metrics["q_value_diff_std"] = np.std(Q_learned_values - Q_optimal_values)
        metrics["q_value_diff_max"] = np.max(Q_learned_values - Q_optimal_values)
        metrics["q_value_correlation"] = np.corrcoef(Q_learned_values, Q_optimal_values)[0, 1]
        
        # Relative error metrics
        metrics["q_value_rel_error_mean"] = np.mean((Q_learned_values - Q_optimal_values) / 
                                                   (Q_optimal_values + 1e-8))
        metrics["q_value_rel_error_std"] = np.std((Q_learned_values - Q_optimal_values) / 
                                                 (Q_optimal_values + 1e-8))
    
    return metrics


def solve_moment_matching_Q(P_z, P_z_next, P_y, L_xu, gamma, N, M, seed):
    """
    Two-stage approach for general linear systems (features on [x,u]).
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
        cp.sum(lambda_var) == 1
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

    # Compute C_val
    mu = mu_var.value
    # C_val = P_y^T Diag(mu) P_y (reuse the same efficient row-scaling form)
    # C_val = (P_y.T @ (mu[:, None] * P_y))
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

    # Objective: sum_i mu_i * <Q, y_i y_i^T> (same as lqr_optimized.py)
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
        Q_learned = 0.5 * (Q_learned + Q_learned.T)
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
    
    Returns:
        status: Solver status
    """
    d = P_z.shape[1]

    Q_id = cp.Variable((d, d)) 

    # Individual constraints for better numerical stability
    cons = []
    for i in range(N):
        p = P_z[i]
        pn = P_z_next[i]
        Mi = np.outer(p, p) - gamma * np.outer(pn, pn)
        cons.append(cp.sum(cp.multiply(Q_id, Mi)) <= L_xu[i])
    
    # Create block diagonal matrix C = diag(I_dx, 0.8 * I_du)
    # For linear features, first dx features are states, next du features are controls
    C_matrix = np.zeros((d, d))
    C_matrix[:dx, :dx] = np.eye(dx)  # Identity for state dimensions
    C_matrix[dx:, dx:] = 0.8 * np.eye(du)  # 0.8 * Identity for input dimensions

    objective = cp.Maximize(cp.trace(cp.Constant(C_matrix) @ Q_id))
    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        return prob.status
    except Exception as e:
        print(f"    MOSEK failed in Identity LP: {e}")
        return "solver_error"


def run_one(seed, dx, A, B, C_cost, N, gamma, M_offline, degree, rho, exclude_u_squared=False):
    """Generate a dataset and solve both LPs; return boundedness flags and Q quality comparison."""
    # Set seed for reproducibility
    np.random.seed(seed)
    
    x, u, x_plus, u_plus = generate_dataset(A, B, N=N, dx=dx, du=B.shape[1], seed=0)

    # Compute polynomial features once for both methods
    z = np.concatenate([x, u], axis=1)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)
    
    # Use filtered polynomial features if exclude_u_squared is True
    if exclude_u_squared:
        du = B.shape[1]
        poly = FilteredPolynomialFeatures(degree=degree, include_bias=False, dx=dx, du=du, exclude_u_squared=True)
    else:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    Z_all = np.concatenate([z, z_plus], axis=0)  # Stack vertically: (2N, dx+du)
    P_all = poly.fit_transform(Z_all)
    P_z = P_all[:N]       # First N rows
    P_z_next = P_all[N:]  # Last N rows
    L_xu = stage_cost(x, u, C_cost, rho)
    
    # Auxiliary samples for moment matching
    x_aux, u_aux = auxiliary_samples(M_offline, dx, B.shape[1], seed=0)
    y_aux = np.concatenate([x_aux, u_aux], axis=1)
    P_y = poly.transform(y_aux)
    
    # Solve both methods using the same polynomial features
    # Moment matching LP
    status_m2, status_m1, Q_learned, C_val, E_Q_learned, mu= solve_moment_matching_Q(
        P_z, P_z_next, P_y, L_xu, gamma, N, M_offline, seed)
    
    # Identity LP
    status_id = solve_identity_Q(P_z, P_z_next, L_xu, gamma, N, seed, dx, B.shape[1])

    def is_bounded(status):
        return status in ("optimal", "optimal_inaccurate")

    result = {
        "seed": seed,
        "dx": dx,
        "moment_matching_status": status_m2,
        "moment_matching_bounded": is_bounded(status_m2),
        "moment_stage1_status": status_m1,
        "identity_status": status_id,
        "identity_bounded": is_bounded(status_id),
    }
    
    # Add Q quality comparison if moment matching was successful
    if is_bounded(status_m2) and Q_learned is not None and C_val is not None and E_Q_learned is not None:
        # Compute optimal Q matrix (in original state-action space)
        Q_optimal_orig = compute_optimal_Q(A, B, C_cost, gamma, rho)
        
        if Q_optimal_orig is not None:
            # Generate test samples for Q-function evaluation
            n_test = 1000  # Number of test samples
            x_test, u_test = auxiliary_samples(n_test, dx, B.shape[1], seed=seed + 1000)
            test_samples = (x_test, u_test)
            
            # Compare Q matrices using trace difference and Q-function evaluation
            # Pass C_val for weighted trace comparison (using the MM LP direction)
            q_metrics = compare_Q_matrices(Q_learned, Q_optimal_orig, test_samples, poly, dx=dx, du=B.shape[1], C_val=C_val)
            
            # Add metrics to result
            for key, value in q_metrics.items():
                if isinstance(value, (int, float, np.number)):
                    result[f"{key}"] = float(value)
                else:
                    result[f"{key}"] = str(value)
            
            # Extract Q_uu components (action-action block) from both Q matrices
            du_local = B.shape[1]
            Q_uu_lqr = Q_optimal_orig[dx:, dx:]  # Shape: (du, du)
            Q_uu_mm = Q_learned[dx:, dx:]        # Shape: (du, du)
            
            # Save Q_uu matrices (convert to nested lists for JSON serialization)
            result["Q_uu_lqr_matrix"] = Q_uu_lqr.tolist()
            result["Q_uu_mm_matrix"] = Q_uu_mm.tolist()
            
            # Save condition numbers of Q_uu matrices (important for numerical stability)
            result["Q_uu_lqr_cond"] = float(np.linalg.cond(Q_uu_lqr))
            result["Q_uu_mm_cond"] = float(np.linalg.cond(Q_uu_mm))
            
            # Compare policy costs
            policy_metrics = compare_policy_costs(A, B, C_cost, rho, gamma, Q_learned, poly, dx, B.shape[1], 
                                                    n_test_states=100, horizon=1000, seed=seed + 3000)
            for key, value in policy_metrics.items():
                if isinstance(value, (int, float, np.number)):
                    result[f"{key}"] = float(value)
                else:
                    result[f"{key}"] = str(value)
        else:
            result["q_error"] = "Failed to compute optimal Q"
    else:
        result["q_error"] = "Moment matching failed or Q not available"
    
    return result

def sweep_over_dims(dims, seeds, N, gamma, M_offline, degree, rho, exclude_u_squared=False):
    """
    Sweep over dimensions and seeds, loading a different system for each seed.
    
    Each seed maps to a different system from the data/dx_{dx}_du_2_systems.json files.
    There are 50 pre-generated controllable systems per dimension.
    """
    results = []
    available_dims = get_available_dimensions()
    
    for dx in dims:
        # Check if systems are available for this dimension
        if dx not in available_dims:
            print(f"[WARN] No systems available for dx={dx}. Available: {available_dims}. Skipping.")
            continue
        
        for i, s in enumerate(seeds):
            # Use seed index to select different system (50 systems available per dimension)
            system_idx = (i) % 50  # Wrap around if more than 50 seeds
            
            try:
                A, B, C_cost = load_system(n=dx, idx=system_idx)
            except Exception as e:
                print(f"[WARN] Failed to load system for dx={dx}, idx={system_idx}: {e}. Skipping.")
                continue
            
            print(f"Running seed {s} for dx={dx} (system {system_idx})")
            results.append(run_one(seed=int(s), dx=int(dx), A=A, B=B, C_cost=C_cost, N=N, gamma=gamma, M_offline=M_offline, degree=degree, rho=rho, exclude_u_squared=exclude_u_squared))
    
    return results

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Boundedness vs state dimension (linear systems).")
    parser.add_argument("--dims", type=str, default="", help="Comma-separated list of state dimensions (e.g., '4,8,12'). If empty, use --dmin.. arguments.")
    parser.add_argument("--dmin", type=int, default=2, help="Minimum dx (used if --dims is empty)")
    parser.add_argument("--dmax", type=int, default=30, help="Maximum dx (used if --dims is empty)")
    parser.add_argument("--dstep", type=int, default=1, help="Step for dx (used if --dims is empty)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per dimension (default: 10)")
    parser.add_argument("--N", type=int, default=1000, help="Number of samples for dataset (default: 200)")
    from config import GAMMA
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor (default: from config)")
    parser.add_argument("--M_offline", type=int, default=250, help="Offline pool size (default: 500)")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial feature degree (default: 1)")
    parser.add_argument("--rho", type=float, default=0.1, help="Control cost weight (default: 0.1)")
    parser.add_argument("--exclude_u_squared", action="store_true", help="Exclude u^2 terms from polynomial features (for degree 2)")
    parser.add_argument("--out_json", type=str, default=None, help="Raw results JSON (if None, auto-generated with N)")
    parser.add_argument("--plot_dir", type=str, default=None, help="Plot filename (if None, auto-generated with N)")
    parser.add_argument("--boundedness_threshold", type=float, default=50.0, help="Minimum boundedness percentage to include in trace plot (default: 50.0)")
    args = parser.parse_args()

    if args.dims.strip():
        dims = [int(s) for s in args.dims.split(",") if s.strip()]
    else:
        dims = list(range(args.dmin, args.dmax + 1, args.dstep))
    
    # Generate seeds for each experiment
    # Each seed will use a different system (system_idx = seed_index % 50)
    seeds = list(range(21, args.seeds + 21))
    
    # Print available dimensions for reference
    print(f"Available dimensions with pre-generated systems: {get_available_dimensions()}")

    # Auto-generate filenames with N if not provided
    if args.out_json is None:
        args.out_json = f"bounded_vs_dim_results_N_{args.N}_rho_{args.rho}.json"
    if args.plot_dir is None:
        args.plot_dir = f"../figures/bounded_vs_dim_percentages_N_{args.N}_rho_{args.rho}.pdf"

    results = sweep_over_dims(dims, seeds, N=args.N, gamma=args.gamma, M_offline=args.M_offline, degree=degree, rho=args.rho, exclude_u_squared=args.exclude_u_squared)

    if len(results) == 0:
        print("No results (no valid dimensions or files). Exiting.")
        raise SystemExit(0)

    df = pd.DataFrame(results)
    agg = df.groupby("dx").agg(
        moment_bounded_mean=("moment_matching_bounded", "mean"),
        identity_bounded_mean=("identity_bounded", "mean"),
        n=("seed", "count"),
    ).reset_index()

    # Convert to percentages
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_mean"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_mean"]

    # Add Q quality metrics aggregation
    q_metrics = ["trace_diff", "trace_diff_abs", "weighted_trace_diff", "weighted_trace_diff_abs",
                 "q_value_diff_mean", "q_value_diff_max", 
                 "q_value_correlation", "q_value_rel_error_mean"]
    
    for metric in q_metrics:
        if metric in df.columns:
            # Only aggregate non-null values
            valid_data = df[df[metric].notna()]
            if len(valid_data) > 0:
                # Use groupby with proper aggregation
                metric_agg = valid_data.groupby("dx")[metric].agg(['mean', 'std', 'min', 'max']).reset_index()
                # Rename columns to match expected format
                metric_agg = metric_agg.rename(columns={'mean': f'{metric}_mean', 'std': f'{metric}_std', 'min': f'{metric}_min', 'max': f'{metric}_max'})
                # Merge with main aggregation
                agg = agg.merge(metric_agg, on='dx', how='left')
    
    # Add policy cost metrics aggregation
    policy_cost_metrics = ["policy_cost_normalized_diff_mean", "policy_cost_diff_mean", 
                          "policy_cost_lqr_mean", "policy_cost_mm_mean"]
    
    # Add value function metrics aggregation
    value_func_metrics = ["value_func_normalized_diff_mean", "value_func_diff_mean",
                          "value_func_lqr_mean", "value_func_mm_mean",
                          "value_matrix_diff_normalized",
                          "K_diff_normalized", "K_diff_frob"]
    
    # Debug: Check what policy cost columns exist
    policy_cost_cols = [col for col in df.columns if col.startswith("policy_cost")]
    if len(policy_cost_cols) > 0:
        print(f"\nPolicy cost columns found in dataframe: {policy_cost_cols}")
        # Check for error columns
        error_cols = [col for col in policy_cost_cols if "error" in col]
        if len(error_cols) > 0:
            print(f"Policy cost error columns: {error_cols}")
            for col in error_cols:
                error_counts = df[col].value_counts()
                print(f"  {col} value counts:\n{error_counts}")
    else:
        print("\nWARNING: No policy cost columns found in dataframe!")
        print("Available columns:", list(df.columns))
    
    for metric in policy_cost_metrics:
        if metric in df.columns:
            # Filter out non-numeric values (like error strings)
            valid_data = df[df[metric].notna()].copy()
            # Convert to numeric, coercing errors to NaN
            valid_data[metric] = pd.to_numeric(valid_data[metric], errors='coerce')
            valid_data = valid_data[valid_data[metric].notna()]
            
            if len(valid_data) > 0:
                # Use groupby with proper aggregation
                metric_agg = valid_data.groupby("dx")[metric].agg(['mean', 'std', 'min', 'max']).reset_index()
                # Rename columns to match expected format
                metric_agg = metric_agg.rename(columns={'mean': f'{metric}_mean', 'std': f'{metric}_std', 'min': f'{metric}_min', 'max': f'{metric}_max'})
                # Merge with main aggregation
                agg = agg.merge(metric_agg, on='dx', how='left')
                print(f"Aggregated {metric}: {len(valid_data)} valid entries across {len(metric_agg)} dimensions")
            else:
                print(f"WARNING: No valid numeric data for {metric}")
        else:
            print(f"WARNING: Metric {metric} not found in dataframe columns")
    
    # Aggregate value function metrics
    for metric in value_func_metrics:
        if metric in df.columns:
            # Filter out non-numeric values (like error strings)
            valid_data = df[df[metric].notna()].copy()
            # Convert to numeric, coercing errors to NaN
            valid_data[metric] = pd.to_numeric(valid_data[metric], errors='coerce')
            valid_data = valid_data[valid_data[metric].notna()]
            
            if len(valid_data) > 0:
                # Use groupby with proper aggregation
                metric_agg = valid_data.groupby("dx")[metric].agg(['mean', 'std', 'min', 'max']).reset_index()
                # Rename columns to match expected format
                metric_agg = metric_agg.rename(columns={'mean': f'{metric}_mean', 'std': f'{metric}_std', 'min': f'{metric}_min', 'max': f'{metric}_max'})
                # Merge with main aggregation
                agg = agg.merge(metric_agg, on='dx', how='left')
                print(f"Aggregated {metric}: {len(valid_data)} valid entries across {len(metric_agg)} dimensions")
            else:
                print(f"WARNING: No valid numeric data for {metric}")

    # Add individual seed values as lists for policy cost and value function metrics
    key_metrics_to_list = ["policy_cost_normalized_diff_mean", "value_func_normalized_diff_mean"]
    for metric in key_metrics_to_list:
        if metric in df.columns:
            # Get all individual values per dimension as a list
            values_by_dx = df.groupby("dx")[metric].apply(
                lambda x: [v for v in pd.to_numeric(x, errors='coerce').dropna().tolist()]
            ).reset_index()
            values_by_dx.columns = ["dx", f"{metric}_all_seeds"]
            agg = agg.merge(values_by_dx, on='dx', how='left')
            print(f"Added individual seed values for {metric}")

    # Save raw + aggregated
    df.to_json(args.out_json, orient="records", indent=2)
    # Generate aggregated filename with N
    agg_json_name = f"bounded_vs_dim_percentages_N_{args.N}_rho_{args.rho}.json"
    agg.to_json(agg_json_name, orient="records", indent=2)
    
    # Print compact table
    cols = ["dx", "n",
            "moment_bounded_pct",
            "identity_bounded_pct"]
    print(agg[cols].to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    
    # Print Q quality summary
    print("\nQ Quality Comparison Summary:")
    print("=" * 50)
    
    # Define metric names for better readability
    metric_names = {
        "trace_diff": "Relative trace error (|Δtrace|/|trace_optimal|)",
        "trace_diff_abs": "Absolute trace difference",
        "weighted_trace_diff": "Relative weighted trace error (using C from MM LP)",
        "weighted_trace_diff_abs": "Absolute weighted trace difference",
        "q_value_diff_mean": "Q-value difference (mean)",
        "q_value_diff_max": "Q-value difference (max)",
        "q_value_correlation": "Q-value correlation",
        "q_value_rel_error_mean": "Q-value relative error (mean)"
    }
    
    for metric in q_metrics:
        if f"{metric}_mean" in agg.columns:
            print(f"{metric}_mean in agg.columns")
            # Only compute statistics for dimensions where we have data
            valid_data = agg[f"{metric}_mean"].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                min_val = agg[f"{metric}_min"].dropna().min() if f"{metric}_min" in agg.columns else 0
                max_val = agg[f"{metric}_max"].dropna().max() if f"{metric}_max" in agg.columns else 0
                metric_name = metric_names.get(metric, metric)
                print(f"{metric_name}: {mean_val:.4f} (range: {min_val:.4f} - {max_val:.4f}) (over {len(valid_data)} dimensions)")
            else:
                metric_name = metric_names.get(metric, metric)
                print(f"{metric_name}: No successful comparisons")
    
    # ===========================================
    # FIGURE 1: Boundedness + Q matrix trace difference
    # ===========================================
    # Font sizes for plots
    LABEL_SIZE = 20
    TITLE_SIZE = 20
    LEGEND_SIZE = 18
    TICK_SIZE = 16
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Boundedness percentages
    axes1[0].plot(agg["dx"], agg["moment_bounded_pct"],
                    marker="o", label="Moment matching", color="blue")
    axes1[0].plot(agg["dx"], agg["identity_bounded_pct"],
                    marker="s", label="Identity covariance", color="red")
    axes1[0].set_xlabel("State dimension", fontsize=LABEL_SIZE)
    axes1[0].set_ylabel("Bounded LPs (%)", fontsize=LABEL_SIZE)
    axes1[0].set_title("Percentage of bounded LP problems vs state dimension", fontsize=TITLE_SIZE)
    axes1[0].legend(fontsize=LEGEND_SIZE)
    axes1[0].tick_params(axis='both', labelsize=TICK_SIZE)
    axes1[0].grid(True)
    
    # Plot 2: Trace difference (normalized and filtered by boundedness threshold)
    if "trace_diff_mean" in agg.columns:
        # Filter dimensions based on boundedness threshold
        threshold = args.boundedness_threshold
        filtered_agg = agg[agg["moment_bounded_pct"] >= threshold].copy()
        
        if len(filtered_agg) > 0:
            # Plot the mean line
            axes1[1].plot(filtered_agg["dx"], filtered_agg["trace_diff_mean"], 
                           marker="o", color="blue", label="Mean")
            # Add shaded region for min-max range
            if "trace_diff_min" in filtered_agg.columns and "trace_diff_max" in filtered_agg.columns:
                axes1[1].fill_between(filtered_agg["dx"], filtered_agg["trace_diff_min"], filtered_agg["trace_diff_max"], 
                                       alpha=0.3, color="blue", label="Min-Max range")
            
            axes1[1].set_xlabel("State dimension", fontsize=LABEL_SIZE)
            axes1[1].set_ylabel("Relative trace error", fontsize=LABEL_SIZE)
            axes1[1].set_title(f"Q matrix trace difference vs dimension", fontsize=TITLE_SIZE)
            axes1[1].tick_params(axis='both', labelsize=TICK_SIZE)
            axes1[1].grid(True)
            axes1[1].legend(fontsize=LEGEND_SIZE)
            
            print(f"Trace plot shows {len(filtered_agg)} dimensions with ≥{threshold}% boundedness")
        else:
            axes1[1].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes1[1].transAxes, fontsize=LABEL_SIZE)
            axes1[1].set_title(f"Q matrix trace difference vs dimension", fontsize=TITLE_SIZE)
            print(f"No dimensions found with ≥{threshold}% boundedness")
    else:
        axes1[1].text(0.5, 0.5, "No trace difference data available", 
                       ha="center", va="center", transform=axes1[1].transAxes, fontsize=LABEL_SIZE)
        axes1[1].set_title("Q matrix trace difference vs dimension", fontsize=TITLE_SIZE)
    
    fig1.tight_layout()
    # Generate filename for figure 1
    fig1_path = args.plot_dir.replace(".pdf", "_boundedness.pdf")
    fig1.savefig(fig1_path, dpi=150)
    print(f"Saved Figure 1 (Boundedness + Trace): {fig1_path}")
    plt.close(fig1)
    
    # ===========================================
    # FIGURE 2: Policy cost + Value function comparison
    # ===========================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Policy cost comparison (normalized difference)
    if "policy_cost_normalized_diff_mean_mean" in agg.columns:
        # Filter dimensions based on boundedness threshold
        threshold = args.boundedness_threshold
        filtered_agg = agg[agg["moment_bounded_pct"] >= threshold].copy()
        
        if len(filtered_agg) > 0:
            # Plot the mean line
            axes2[0].plot(filtered_agg["dx"], filtered_agg["policy_cost_normalized_diff_mean_mean"], 
                           marker="o", color="green", label="Mean normalized difference")
            # Add shaded region for min-max range
            if "policy_cost_normalized_diff_mean_min" in filtered_agg.columns and "policy_cost_normalized_diff_mean_max" in filtered_agg.columns:
                axes2[0].fill_between(filtered_agg["dx"], filtered_agg["policy_cost_normalized_diff_mean_min"], 
                                       filtered_agg["policy_cost_normalized_diff_mean_max"], 
                                       alpha=0.3, color="green", label="Min-Max range")
            
            # Add horizontal line at y=0 for reference
            axes2[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            axes2[0].set_xlabel("State dimension", fontsize=LABEL_SIZE)
            axes2[0].set_ylabel("Normalized cost difference", fontsize=LABEL_SIZE)
            axes2[0].set_title(f"Policy cost comparison vs dimension", fontsize=TITLE_SIZE)
            axes2[0].tick_params(axis='both', labelsize=TICK_SIZE)
            axes2[0].grid(True)
            axes2[0].legend(fontsize=LEGEND_SIZE)
            
            print(f"Policy cost plot shows {len(filtered_agg)} dimensions with ≥{threshold}% boundedness")
        else:
            axes2[0].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes2[0].transAxes, fontsize=LABEL_SIZE)
            axes2[0].set_title(f"Policy cost comparison vs dimension", fontsize=TITLE_SIZE)
            print(f"No dimensions found with ≥{threshold}% boundedness for policy cost plot")
    else:
        axes2[0].text(0.5, 0.5, "No policy cost data available", 
                       ha="center", va="center", transform=axes2[0].transAxes, fontsize=LABEL_SIZE)
        axes2[0].set_title("Policy cost comparison vs dimension", fontsize=TITLE_SIZE)
    
    # Plot 2: Value function comparison (normalized difference)
    if "value_func_normalized_diff_mean_mean" in agg.columns:
        # Filter dimensions based on boundedness threshold
        threshold = args.boundedness_threshold
        filtered_agg = agg[agg["moment_bounded_pct"] >= threshold].copy()
        
        if len(filtered_agg) > 0:
            # Plot the mean line
            axes2[1].plot(filtered_agg["dx"], filtered_agg["value_func_normalized_diff_mean_mean"], 
                           marker="o", color="purple", label="Mean normalized difference")
            # Add shaded region for min-max range
            if "value_func_normalized_diff_mean_min" in filtered_agg.columns and "value_func_normalized_diff_mean_max" in filtered_agg.columns:
                axes2[1].fill_between(filtered_agg["dx"], filtered_agg["value_func_normalized_diff_mean_min"], 
                                       filtered_agg["value_func_normalized_diff_mean_max"], 
                                       alpha=0.3, color="purple", label="Min-Max range")
            
            # Add horizontal line at y=0 for reference
            axes2[1].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            axes2[1].set_xlabel("State dimension", fontsize=LABEL_SIZE)
            axes2[1].set_ylabel("Normalized V difference", fontsize=LABEL_SIZE)
            axes2[1].set_title(f"Value function comparison vs dimension", fontsize=TITLE_SIZE)
            axes2[1].tick_params(axis='both', labelsize=TICK_SIZE)
            axes2[1].grid(True)
            axes2[1].legend(fontsize=LEGEND_SIZE)
            
            # Print value function summary
            V_diff_mean = filtered_agg["value_func_normalized_diff_mean_mean"].mean()
            print(f"\nValue function comparison summary:")
            print(f"  Average normalized V difference: {V_diff_mean:.4f}")
            print(f"  Value function plot shows {len(filtered_agg)} dimensions with ≥{threshold}% boundedness")
        else:
            axes2[1].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes2[1].transAxes, fontsize=LABEL_SIZE)
            axes2[1].set_title(f"Value function comparison vs dimension", fontsize=TITLE_SIZE)
            print(f"No dimensions found with ≥{threshold}% boundedness for value function plot")
    else:
        axes2[1].text(0.5, 0.5, "No value function data available", 
                       ha="center", va="center", transform=axes2[1].transAxes, fontsize=LABEL_SIZE)
        axes2[1].set_title("Value function comparison vs dimension", fontsize=TITLE_SIZE)
    
    fig2.tight_layout()
    # Generate filename for figure 2
    fig2_path = args.plot_dir.replace(".pdf", "_policy_value.pdf")
    fig2.savefig(fig2_path, dpi=150)
    print(f"Saved Figure 2 (Policy + Value): {fig2_path}")
    plt.close(fig2)
    
    # ===========================================
    # FIGURE 3: Policy cost + Value function comparison (with std bands)
    # ===========================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Policy cost comparison with std bands
    if "policy_cost_normalized_diff_mean_mean" in agg.columns:
        # Filter dimensions based on boundedness threshold
        threshold = args.boundedness_threshold
        filtered_agg = agg[agg["moment_bounded_pct"] >= threshold].copy()
        
        if len(filtered_agg) > 0:
            mean_vals = filtered_agg["policy_cost_normalized_diff_mean_mean"]
            # Plot the mean line
            axes3[0].plot(filtered_agg["dx"], mean_vals, 
                           marker="o", color="green", label="Mean")
            # Add shaded region for mean ± std
            if "policy_cost_normalized_diff_mean_std" in filtered_agg.columns:
                std_vals = filtered_agg["policy_cost_normalized_diff_mean_std"]
                axes3[0].fill_between(filtered_agg["dx"], 
                                       mean_vals - std_vals, 
                                       mean_vals + std_vals, 
                                       alpha=0.3, color="green", label="±1 Std Dev")
            
            # Add horizontal line at y=0 for reference
            axes3[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            axes3[0].set_xlabel("State dimension", fontsize=LABEL_SIZE)
            axes3[0].set_ylabel("Normalized cost difference", fontsize=LABEL_SIZE)
            axes3[0].set_title(f"Policy cost comparison vs dimension", fontsize=TITLE_SIZE)
            axes3[0].tick_params(axis='both', labelsize=TICK_SIZE)
            axes3[0].grid(True)
            axes3[0].legend(fontsize=LEGEND_SIZE)
        else:
            axes3[0].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes3[0].transAxes, fontsize=LABEL_SIZE)
            axes3[0].set_title(f"Policy cost comparison vs dimension", fontsize=TITLE_SIZE)
    else:
        axes3[0].text(0.5, 0.5, "No policy cost data available", 
                       ha="center", va="center", transform=axes3[0].transAxes, fontsize=LABEL_SIZE)
        axes3[0].set_title("Policy cost comparison vs dimension", fontsize=TITLE_SIZE)
    
    # Plot 2: Value function comparison with std bands
    if "value_func_normalized_diff_mean_mean" in agg.columns:
        # Filter dimensions based on boundedness threshold
        threshold = args.boundedness_threshold
        filtered_agg = agg[agg["moment_bounded_pct"] >= threshold].copy()
        
        if len(filtered_agg) > 0:
            mean_vals = filtered_agg["value_func_normalized_diff_mean_mean"]
            # Plot the mean line
            axes3[1].plot(filtered_agg["dx"], mean_vals, 
                           marker="o", color="purple", label="Mean")
            # Add shaded region for mean ± std
            if "value_func_normalized_diff_mean_std" in filtered_agg.columns:
                std_vals = filtered_agg["value_func_normalized_diff_mean_std"]
                axes3[1].fill_between(filtered_agg["dx"], 
                                       mean_vals - std_vals, 
                                       mean_vals + std_vals, 
                                       alpha=0.3, color="purple", label="±1 Std Dev")
            
            # Add horizontal line at y=0 for reference
            axes3[1].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            axes3[1].set_xlabel("State dimension", fontsize=LABEL_SIZE)
            axes3[1].set_ylabel("Normalized V difference", fontsize=LABEL_SIZE)
            axes3[1].set_title(f"Value function comparison vs dimension", fontsize=TITLE_SIZE)
            axes3[1].tick_params(axis='both', labelsize=TICK_SIZE)
            axes3[1].grid(True)
            axes3[1].legend(fontsize=LEGEND_SIZE)
        else:
            axes3[1].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes3[1].transAxes, fontsize=LABEL_SIZE)
            axes3[1].set_title(f"Value function comparison vs dimension", fontsize=TITLE_SIZE)
    else:
        axes3[1].text(0.5, 0.5, "No value function data available", 
                       ha="center", va="center", transform=axes3[1].transAxes, fontsize=LABEL_SIZE)
        axes3[1].set_title("Value function comparison vs dimension", fontsize=TITLE_SIZE)
    
    fig3.tight_layout()
    # Generate filename for figure 3
    fig3_path = args.plot_dir.replace(".pdf", "_policy_value_std.pdf")
    fig3.savefig(fig3_path, dpi=150)
    print(f"Saved Figure 3 (Policy + Value with std bands): {fig3_path}")
    plt.close(fig3)
    
    # ===========================================
    # FIGURE 4: K matrix difference + Weighted trace difference
    # ===========================================
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: K matrix relative difference (||K_mm - K_lqr||_F / ||K_lqr||_F)
    if "K_diff_normalized_mean" in agg.columns:
        # Filter dimensions based on boundedness threshold
        threshold = args.boundedness_threshold
        filtered_agg = agg[agg["moment_bounded_pct"] >= threshold].copy()
        
        if len(filtered_agg) > 0:
            mean_vals = filtered_agg["K_diff_normalized_mean"]
            # Plot the mean line
            axes4[0].plot(filtered_agg["dx"], mean_vals, 
                           marker="o", color="darkorange", label="Mean")
            # Add shaded region for mean ± std
            if "K_diff_normalized_std" in filtered_agg.columns:
                std_vals = filtered_agg["K_diff_normalized_std"]
                axes4[0].fill_between(filtered_agg["dx"], 
                                       mean_vals - std_vals, 
                                       mean_vals + std_vals, 
                                       alpha=0.3, color="darkorange", label="±1 Std Dev")
            
            # Add horizontal line at y=0 for reference
            axes4[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            axes4[0].set_xlabel("State dimension", fontsize=LABEL_SIZE)
            axes4[0].set_ylabel("Relative K difference", fontsize=LABEL_SIZE)
            axes4[0].set_title(f"Gain matrix K: ||K_mm - K_lqr||/||K_lqr||", fontsize=TITLE_SIZE)
            axes4[0].tick_params(axis='both', labelsize=TICK_SIZE)
            axes4[0].grid(True)
            axes4[0].legend(fontsize=LEGEND_SIZE)
            
            print(f"K matrix plot shows {len(filtered_agg)} dimensions with ≥{threshold}% boundedness")
        else:
            axes4[0].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes4[0].transAxes, fontsize=LABEL_SIZE)
            axes4[0].set_title(f"Gain matrix K comparison", fontsize=TITLE_SIZE)
    else:
        axes4[0].text(0.5, 0.5, "No K matrix data available", 
                       ha="center", va="center", transform=axes4[0].transAxes, fontsize=LABEL_SIZE)
        axes4[0].set_title("Gain matrix K comparison", fontsize=TITLE_SIZE)
    
    # Plot 2: Weighted trace difference (using C_val from MM LP direction)
    if "weighted_trace_diff_mean" in agg.columns:
        # Filter dimensions based on boundedness threshold
        threshold = args.boundedness_threshold
        filtered_agg = agg[agg["moment_bounded_pct"] >= threshold].copy()
        
        if len(filtered_agg) > 0:
            mean_vals = filtered_agg["weighted_trace_diff_mean"]
            # Plot the mean line
            axes4[1].plot(filtered_agg["dx"], mean_vals, 
                           marker="o", color="teal", label="Mean")
            # Add shaded region for mean ± std
            if "weighted_trace_diff_std" in filtered_agg.columns:
                std_vals = filtered_agg["weighted_trace_diff_std"]
                axes4[1].fill_between(filtered_agg["dx"], 
                                       mean_vals - std_vals, 
                                       mean_vals + std_vals, 
                                       alpha=0.3, color="teal", label="±1 Std Dev")
            
            # Add horizontal line at y=0 for reference
            axes4[1].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            axes4[1].set_xlabel("State dimension", fontsize=LABEL_SIZE)
            axes4[1].set_ylabel("Relative weighted trace diff", fontsize=LABEL_SIZE)
            axes4[1].set_title(f"Weighted trace: tr(C·Q_mm) vs tr(C·Q_lqr)", fontsize=TITLE_SIZE)
            axes4[1].tick_params(axis='both', labelsize=TICK_SIZE)
            axes4[1].grid(True)
            axes4[1].legend(fontsize=LEGEND_SIZE)
            
            print(f"Weighted trace plot shows {len(filtered_agg)} dimensions with ≥{threshold}% boundedness")
        else:
            axes4[1].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes4[1].transAxes, fontsize=LABEL_SIZE)
            axes4[1].set_title(f"Weighted trace comparison", fontsize=TITLE_SIZE)
    else:
        axes4[1].text(0.5, 0.5, "No weighted trace data available", 
                       ha="center", va="center", transform=axes4[1].transAxes, fontsize=LABEL_SIZE)
        axes4[1].set_title("Weighted trace comparison", fontsize=TITLE_SIZE)
    
    fig4.tight_layout()
    # Generate filename for figure 4
    fig4_path = args.plot_dir.replace(".pdf", "_K_weighted_trace.pdf")
    fig4.savefig(fig4_path, dpi=150)
    print(f"Saved Figure 4 (K matrix + Weighted trace): {fig4_path}")
    plt.close(fig4)
    
    # # Plot 3: Q-value difference
    # q_diff_col = "q_value_diff_mean_mean" if "q_value_diff_mean_mean" in agg.columns else None
    # if q_diff_col:
    #     # Plot the mean line
    #     axes[1, 0].plot(agg["dx"], agg[q_diff_col], 
    #                    marker="o", color="red", label="Mean")
    #     # Add shaded region for min-max range
    #     if "q_value_diff_mean_min" in agg.columns and "q_value_diff_mean_max" in agg.columns:
    #         axes[1, 0].fill_between(agg["dx"], agg["q_value_diff_mean_min"], agg["q_value_diff_mean_max"], 
    #                                alpha=0.3, color="red", label="Min-Max range")
    #     axes[1, 0].set_xlabel("State dimension (dx)")
    #     axes[1, 0].set_ylabel("Q-value difference (mean)")
    #     axes[1, 0].set_title("Q-function value difference vs dimension")
    #     axes[1, 0].grid(True)
    #     axes[1, 0].legend()
    # else:
    #     axes[1, 0].text(0.5, 0.5, "No Q-value difference data available", 
    #                    ha="center", va="center", transform=axes[1, 0].transAxes)
    #     axes[1, 0].set_title("Q-function value difference vs dimension")
    
    # # Plot 4: Q-value correlation
    # q_corr_col = "q_value_correlation_mean" if "q_value_correlation_mean" in agg.columns else None
    # if q_corr_col:
    #     # Plot the mean line
    #     axes[1, 1].plot(agg["dx"], agg[q_corr_col], 
    #                    marker="o", color="green", label="Mean")
    #     # Add shaded region for min-max range
    #     if "q_value_correlation_min" in agg.columns and "q_value_correlation_max" in agg.columns:
    #         # Ensure bounds stay within [0, 1] for correlation
    #         upper_bound = np.clip(agg["q_value_correlation_max"], 0, 1)
    #         lower_bound = np.clip(agg["q_value_correlation_min"], 0, 1)
    #         axes[1, 1].fill_between(agg["dx"], lower_bound, upper_bound, 
    #                                alpha=0.3, color="green", label="Min-Max range")
    #     axes[1, 1].set_xlabel("State dimension (dx)")
    #     axes[1, 1].set_ylabel("Q-value correlation")
    #     axes[1, 1].set_title("Q-function value correlation vs dimension")
    #     axes[1, 1].grid(True)
    #     axes[1, 1].set_ylim([0, 1])  # Correlation is between 0 and 1
    #     axes[1, 1].legend()
    # else:
    #     axes[1, 1].text(0.5, 0.5, "No Q-value correlation data available", 
    #                    ha="center", va="center", transform=axes[1, 1].transAxes)
    #     axes[1, 1].set_title("Q-function value correlation vs dimension")
    
    print(f"\nSaved: {args.out_json}")
    print(f"Figures saved: {fig1_path}, {fig2_path}, {fig3_path}, {fig4_path}")