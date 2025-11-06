
import json
import numpy as np
import cvxpy as cp
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve_discrete_are
from scipy import sparse
from policy_extraction import PolicyExtractor

# -------------------------
# Config (defaults; can be overridden by CLI)
# -------------------------
degree = 2           # polynomial feature degree
M_offline = 500      # auxiliary pool size for building P_y

# Sampling bounds for generating (x,u) datasets
# (Assumed symmetric for all dimensions)
x_bound = 3.0        # states sampled in [-x_bound, x_bound]
u_bound = 1.0        # inputs sampled in [-u_bound, u_bound]

du = 2               # fixed input dimension

def load_system_json(data_dir: Path, dx: int):
    """Load A, B, C from ../data/dx_%d.json"""
    fname = data_dir / f"dx_{dx}.json"
    with open(fname, "r") as f:
        data = json.load(f)
    A = np.array(data["A"], dtype=float)
    B = np.array(data["B"], dtype=float)
    C = np.array(data["C"], dtype=float)
    # Basic checks
    if A.shape != (dx, dx):
        raise ValueError(f"A shape mismatch for dx={dx}: got {A.shape}, expected {(dx, dx)}")
    if B.shape[0] != dx or B.shape[1] != du:
        raise ValueError(f"B shape mismatch for dx={dx}, du={du}: got {B.shape}, expected {(dx, du)}")
    if C.shape != (dx, dx):
        raise ValueError(f"C shape mismatch for dx={dx}: got {C.shape}, expected {(dx, dx)}")
    return A, B, C

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
    
    try:
        P = solve_discrete_are(Ad, B, Q_lqr, Rd)
    except Exception as e:
        print(f"Warning: Failed to solve discrete ARE: {e}")
        return None
    
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


def compare_Q_matrices(Q_learned, Q_optimal, test_samples=None, poly=None):
    """
    Compare the learned Q matrix with the optimal Q matrix using trace difference and Q-function evaluation.
    
    Args:
        Q_learned: Learned Q matrix from moment matching (in scaled feature space)
        Q_optimal: Optimal Q matrix from LQR theory (in original state-action space)
        test_samples: Optional test samples (x, u) for Q-function evaluation
        poly: PolynomialFeatures object for feature transformation
    Returns:
        dict: Comparison metrics
    """
    if Q_learned is None or Q_optimal is None:
        return {"error": "One or both Q matrices are None"}
    
    metrics = {}
    
    # Trace difference - the matrices are in the same feature space since degree=1
    trace_diff_abs = np.abs(np.trace(Q_learned) - np.trace(Q_optimal))
    
    # Normalize trace error by the scale of the optimal Q matrix entries
    # Better options for trace-specific normalization:
    # 1. Trace of optimal Q: |trace(Q_learned) - trace(Q_optimal)| / |trace(Q_optimal)|
    # 2. Mean absolute value of optimal Q: |trace_diff| / mean(|Q_optimal|)
    # 3. Max absolute value of optimal Q: |trace_diff| / max(|Q_optimal|)
    # 4. Frobenius norm (current): |trace_diff| / ||Q_optimal||_F
    
    # Option 1: Normalize by trace of optimal Q (most direct for trace error)
    trace_optimal = np.trace(Q_optimal)
    if abs(trace_optimal) > 1e-10:  # Avoid division by zero
        metrics["trace_diff"] = trace_diff_abs / abs(trace_optimal)
        metrics["trace_diff_abs"] = trace_diff_abs  # Keep absolute value for reference
    else:
        # Fallback: normalize by mean absolute value of matrix entries
        mean_abs_optimal = np.mean(np.abs(Q_optimal))
        if mean_abs_optimal > 1e-10:
            metrics["trace_diff"] = trace_diff_abs / mean_abs_optimal
            metrics["trace_diff_abs"] = trace_diff_abs
        else:
            metrics["trace_diff"] = trace_diff_abs  # Final fallback to absolute
            metrics["trace_diff_abs"] = trace_diff_abs
    
    print(f"Trace difference (normalized): {metrics['trace_diff']:.6f}")
    print(f"Trace difference (absolute): {metrics['trace_diff_abs']:.6f}")
    print(f"Optimal Q matrix trace: {np.trace(Q_optimal)}")
    print(f"Learned Q matrix trace: {np.trace(Q_learned)}")
    print(f"Trace difference (learned Q - optimal Q): {np.trace(Q_learned) - np.trace(Q_optimal)}")

    # Q-function evaluation on test samples
    print(f"Q-function evaluation conditions: test_samples={test_samples is not None}, poly={poly is not None}")
    if test_samples is not None and poly is not None:
        try:
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
            metrics["q_value_diff_mean"] = np.mean(np.abs(Q_learned_values - Q_optimal_values))
            print(f"Q-value difference (mean): {metrics['q_value_diff_mean']}")
            metrics["q_value_diff_std"] = np.std(np.abs(Q_learned_values - Q_optimal_values))
            print(f"Q-value difference (std): {metrics['q_value_diff_std']}")
            metrics["q_value_diff_max"] = np.max(np.abs(Q_learned_values - Q_optimal_values))
            print(f"Q-value difference (max): {metrics['q_value_diff_max']}")
            metrics["q_value_correlation"] = np.corrcoef(Q_learned_values, Q_optimal_values)[0, 1]
            print(f"Q-value correlation: {metrics['q_value_correlation']}")
            
            # Relative error metrics
            metrics["q_value_rel_error_mean"] = np.mean(np.abs(Q_learned_values - Q_optimal_values) / 
                                                       (np.abs(Q_optimal_values) + 1e-8))
            metrics["q_value_rel_error_std"] = np.std(np.abs(Q_learned_values - Q_optimal_values) / 
                                                     (np.abs(Q_optimal_values) + 1e-8))
            
            # Check value function positivity for learned Q matrix
            print("\n=== Checking Value Function Positivity ===")
            extractor = PolicyExtractor()
            positivity_results = extractor.check_value_function_positivity(
                Q_learned, poly, n_samples=1000, verbose=True
            )
            
            # Add positivity results to metrics
            metrics["value_function_all_positive"] = positivity_results["all_positive"]
            metrics["value_function_positive_ratio"] = positivity_results["positive_ratio"]
            metrics["value_function_min_value"] = positivity_results["min_value"]
            metrics["value_function_max_value"] = positivity_results["max_value"]
            metrics["value_function_mean_value"] = positivity_results["mean_value"]
            metrics["value_function_n_samples"] = positivity_results["n_samples"]
            
        except Exception as e:
            print(f"Error in Q-function evaluation: {e}")
            metrics["q_value_error"] = str(e)
    else:
        print("Q-function evaluation skipped due to missing parameters")
    
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
        # cp.sum(mu_var) == 1,
        cp.sum(lambda_var) == 1,
    ]
    
    # Objective: make C ≈ I without materializing dense arrays
    I_d = cp.Constant(sparse.eye(d, format="csr"))
    C_approx = sum_PyPyT
    objective = cp.Minimize(cp.norm(C_approx - I_d, "fro"))

    prob = cp.Problem(objective, constraints)
    try:
        mosek_params = {
            "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-9,
            "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-9,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-9,
            "MSK_IPAR_INTPNT_BASIS": 1
        }
        prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                prob.solve(verbose=False)
    status1 = prob.status

    if status1 not in ("optimal", "optimal_inaccurate"):
        return "failed_stage1", status1, None, None, None, None

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
    Q    = cp.Variable((d, d))

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
    try:
        mosek_params = {
            "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-9,
            "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-9,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-9,
            "MSK_IPAR_INTPNT_BASIS": 1
        }
        prob_lp.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)
    except Exception:
        try:
            prob_lp.solve(solver=cp.SCS, verbose=False)
        except Exception:
            try:
                prob_lp.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                prob_lp.solve(verbose=False)
    status2 = prob_lp.status
    
    # Return the learned Q matrix, polynomial features, and optimization values if successful
    if status2 in ("optimal", "optimal_inaccurate"):
        Q_learned = Q.value
        # Q_learned_mat = Q_learned.reshape((d, d), order="C")
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
        cons.append(cp.sum(cp.multiply(Q_id, Mi)) <= L_xu[i])
    
    # Create block diagonal matrix C = diag(I_dx, 0.8 * I_du)
    # For linear features, first dx features are states, next du features are controls
    C_matrix = np.zeros((d, d))
    C_matrix[:dx, :dx] = np.eye(dx)  # Identity for state dimensions
    C_matrix[dx:, dx:] = 0.8 * np.eye(du)  # 0.8 * Identity for input dimensions
    
    # constraints = [F_mat @ Q_id_vec <= cp.Constant(L_xu), cp.trace(Sigma_const @ Q_id) == 1,      ]

    objective = cp.Maximize(cp.trace(cp.Constant(C_matrix) @ Q_id))
    prob = cp.Problem(objective, cons)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                prob.solve(verbose=False)

    return prob.status


def run_one(seed, dx, A, B, C_cost, N, gamma, M_offline, degree, rho):
    """Generate a dataset and solve both LPs; return boundedness flags and Q quality comparison."""
    # Set seed for reproducibility
    np.random.seed(seed)

    x, u, x_plus, u_plus = generate_dataset(A, B, N=N, dx=dx, du=B.shape[1], seed=seed)

    # Compute polynomial features once for both methods
    z = np.concatenate([x, u], axis=1)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    Z_all = np.concatenate([z, z_plus], axis=0)
    P_all = poly.fit_transform(Z_all)
    P_z = P_all[:N]
    P_z_next = P_all[N:]
    L_xu = stage_cost(x, u, C_cost, rho)
    
    # Auxiliary samples for moment matching
    x_aux, u_aux = auxiliary_samples(M_offline, dx, B.shape[1], seed=seed)
    y_aux = np.concatenate([x_aux, u_aux], axis=1)
    P_y = poly.transform(y_aux)
    
    # Solve both methods using the same polynomial features
    status_m2, status_m1, Q_learned, C_val, E_Q_learned, mu = solve_moment_matching_Q(P_z, P_z_next, P_y, L_xu, gamma, N, M_offline, seed)
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
        try:
            # Compute optimal Q matrix (in original state-action space)
            Q_optimal_orig = compute_optimal_Q(A, B, C_cost, gamma, rho)
            
            if Q_optimal_orig is not None:
                # Generate test samples for Q-function evaluation
                n_test = 1000  # Number of test samples
                x_test, u_test = auxiliary_samples(n_test, dx, B.shape[1], seed=seed + 1000)
                test_samples = (x_test, u_test)
                
                # Compare Q matrices using trace difference and Q-function evaluation
                q_metrics = compare_Q_matrices(Q_learned, Q_optimal_orig, test_samples, poly)
                
                # Add metrics to result
                for key, value in q_metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        result[f"{key}"] = float(value)
                    else:
                        result[f"{key}"] = str(value)
            else:
                result["q_error"] = "Failed to compute optimal Q"
        except Exception as e:
            result["q_error"] = f"Q comparison failed: {str(e)}"
    else:
        result["q_error"] = "Moment matching failed or Q not available"
    
    return result

def sweep_over_dims(dims, data_dir: Path, seeds, N, gamma, M_offline, degree, rho):
    results = []
    for dx in dims:
        path = data_dir / f"dx_{dx}.json"
        if not path.exists():
            print(f"[WARN] Missing file: {path}. Skipping dx={dx}.")
            continue
        try:
            A, B, C_cost = load_system_json(data_dir, dx)
        except Exception as e:
            print(f"[WARN] Failed to load/validate for dx={dx}: {e}. Skipping.")
            continue          
        for s in seeds:
            print(f"Running seed {s} for dx={dx}")
            results.append(run_one(seed=int(s), dx=int(dx), A=A, B=B, C_cost=C_cost, N=N, gamma=gamma, M_offline=M_offline, degree=degree, rho=rho))
    return results

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Boundedness vs state dimension (linear systems).")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory containing dx_%n.json files")
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
    parser.add_argument("--rho", type=float, default=0.001, help="Control cost weight (default: 0.1)")
    parser.add_argument("--out_json", type=str, default="bounded_vs_dim_results_2.json", help="Raw results JSON")
    parser.add_argument("--plot_dir", type=str, default="../figures/bounded_vs_dim_percentages_2.pdf", help="Plot filename")
    parser.add_argument("--boundedness_threshold", type=float, default=50.0, help="Minimum boundedness percentage to include in trace plot (default: 50.0)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.dims.strip():
        dims = [int(s) for s in args.dims.split(",") if s.strip()]
    else:
        dims = list(range(args.dmin, args.dmax + 1, args.dstep))
    
    # Generate seeds for each experiment
    # Start seeds from 0: 10, 11, ..., args.seeds + 9
    seeds = list(range(21, args.seeds + 21))

    results = sweep_over_dims(dims, data_dir, seeds, N=args.N, gamma=args.gamma, M_offline=args.M_offline, degree=args.degree, rho=args.rho)

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
    q_metrics = ["trace_diff", "trace_diff_abs", "q_value_diff_mean", "q_value_diff_max", 
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

    # Save raw + aggregated
    df.to_json(args.out_json, orient="records", indent=2)
    agg.to_json("bounded_vs_dim_percentages.json", orient="records", indent=2)
    
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
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Boundedness percentages
    axes[0].plot(agg["dx"], agg["moment_bounded_pct"],
                    marker="o", label="Moment matching", color="blue")
    axes[0].plot(agg["dx"], agg["identity_bounded_pct"],
                    marker="s", label="Identity covariance", color="red")
    axes[0].set_xlabel("State dimension")
    axes[0].set_ylabel("Bounded LPs (%)")
    axes[0].set_title("Percentage of bounded LP problems vs state dimension")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Trace difference (normalized and filtered by boundedness threshold)
    if "trace_diff_mean" in agg.columns:
        # Filter dimensions based on boundedness threshold
        threshold = args.boundedness_threshold
        filtered_agg = agg[agg["moment_bounded_pct"] >= threshold].copy()
        
        if len(filtered_agg) > 0:
            # Plot the mean line
            axes[1].plot(filtered_agg["dx"], filtered_agg["trace_diff_mean"], 
                           marker="o", color="blue", label="Mean")
            # Add shaded region for min-max range
            if "trace_diff_min" in filtered_agg.columns and "trace_diff_max" in filtered_agg.columns:
                axes[1].fill_between(filtered_agg["dx"], filtered_agg["trace_diff_min"], filtered_agg["trace_diff_max"], 
                                       alpha=0.3, color="blue", label="Min-Max range")
            
            axes[1].set_xlabel("State dimension")
            axes[1].set_ylabel("Relative trace error")
            axes[1].set_title(f"Q matrix trace difference vs dimension")
            axes[1].grid(True)
            axes[1].legend()
            
            print(f"Trace plot shows {len(filtered_agg)} dimensions with ≥{threshold}% boundedness")
        else:
            axes[1].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes[0, 1].transAxes)
            axes[1].set_title(f"Q matrix trace difference vs dimension\n(≥{threshold}% boundedness)")
            print(f"No dimensions found with ≥{threshold}% boundedness")
    else:
        axes[1].text(0.5, 0.5, "No trace difference data available", 
                       ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[1].set_title("Q matrix trace difference vs dimension")
    
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
    
    plt.tight_layout()
    plt.savefig(args.plot_dir, dpi=150)
    print("Saved:", args.out_json, "Figures saved in", args.plot_dir)