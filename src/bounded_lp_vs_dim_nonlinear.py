import json
import numpy as np
import cvxpy as cp
from pathlib import Path
from scipy.linalg import solve_discrete_are
from scipy import sparse
from polynomial_features import FilteredPolynomialFeatures, StateOnlyPolynomialFeatures
from dynamical_systems_polished import point_mass_cubic_drag
from config import (
    GAMMA, M_POINT_MASS_2D, K_POINT_MASS_2D, C_POINT_MASS_2D, DT_POINT_MASS_2D,
    RHO_POINT_MASS_2D, C_P_POINT_MASS, C_V_POINT_MASS,
    P_BOUNDS_POINT_MASS, V_BOUNDS_POINT_MASS, U_BOUNDS_POINT_MASS
)

# -------------------------
# Config (defaults; can be overridden by CLI)
# -------------------------
degree = 2           # polynomial feature degree
M_offline = 500      # auxiliary pool size for building P_y

def create_point_mass_system(dx, N=0, M=0):
    """
    Create a point_mass_cubic_drag system for given state dimension.
    
    Args:
        dx: State dimension (must be even: dx = 2n where n is position/velocity dimension)
        N: Number of samples (for compatibility)
        M: Auxiliary pool size (for compatibility)
    
    Returns:
        system: point_mass_cubic_drag instance
        C_cost: Cost matrix (dx, dx)
    """
    if dx % 2 != 0:
        raise ValueError(f"dx must be even for point mass system (dx={dx}). State is [p, v] where p and v are n-dimensional, so dx = 2n.")
    
    n = dx // 2  # Position/velocity dimension
    m_u = n      # Control dimension (fully actuated)
    
    # Build cost matrix C using config variables
    C_cost = np.diag(np.sqrt(np.concatenate([
        np.full(n, C_P_POINT_MASS),  # Position weights
        np.full(n, C_V_POINT_MASS)   # Velocity weights
    ])))
    
    system = point_mass_cubic_drag(
        n=n,
        m_u=m_u,
        B=None,  # Identity matrix (fully actuated)
        mass=M_POINT_MASS_2D,
        k=K_POINT_MASS_2D,
        c=C_POINT_MASS_2D,
        delta_t=DT_POINT_MASS_2D,
        C=C_cost,
        rho=RHO_POINT_MASS_2D,
        gamma=GAMMA,
        N=N,
        M=M
    )
    
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

def compute_optimal_Q(system, C_cost, gamma, rho):
    """
    Compute the optimal Q matrix for the point mass system by linearizing and solving LQR.
    
    Args:
        system: point_mass_cubic_drag instance
        C_cost: Cost matrix for states (dx, dx)
        gamma: Discount factor
        rho: Control cost weight
        
    Returns:
        Q_optimal: Optimal Q matrix in state-action space (dx+du, dx+du)
    """
    # Linearize the system around the origin
    A_d, B_d = system.linearized_system(use_backward_euler=False)
    
    dx = A_d.shape[0]
    du = B_d.shape[1]
    
    # Define cost matrices for LQR
    Q_lqr = C_cost.T @ C_cost  # State cost matrix: C.T @ C
    R_lqr = rho * np.eye(du)   # Control cost matrix: rho * I_{d_u}
    
    # Solve discrete algebraic Riccati equation
    # Scale A and R for discount factor
    Ad = np.sqrt(gamma) * A_d
    Rd = R_lqr / gamma
    
    try:
        P = solve_discrete_are(Ad, B_d, Q_lqr, Rd)
    except Exception as e:
        print(f"Warning: Failed to solve discrete ARE: {e}")
        return None
    
    # Compute optimal Q-function components
    Qxx = Q_lqr + gamma * (A_d.T @ P @ A_d)
    Quu = R_lqr + gamma * (B_d.T @ P @ B_d)
    Qxu = gamma * (A_d.T @ P @ B_d)
    
    # Build optimal Q matrix in state-action space
    Q_optimal_sa = np.block([
        [Qxx,    Qxu],
        [Qxu.T,  Quu]
    ])
    
    return Q_optimal_sa

def compare_Q_matrices(Q_learned, Q_optimal, test_samples=None, poly=None, dx=None, du=None):
    """
    Compare the learned Q matrix with the optimal Q matrix using trace difference and Q-function evaluation.
    
    Args:
        Q_learned: Learned Q matrix from moment matching (in feature space)
        Q_optimal: Optimal Q matrix from LQR theory (in original state-action space)
        test_samples: Optional test samples (x, u) for Q-function evaluation
        poly: PolynomialFeatures object for feature transformation
        dx: State dimension
        du: Input dimension
    Returns:
        dict: Comparison metrics
    """
    if Q_learned is None or Q_optimal is None:
        return {"error": "One or both Q matrices are None"}
    
    metrics = {}
    
    # Trace difference - the matrices are in the same feature space since degree=1
    trace_diff_abs = np.abs(np.trace(Q_learned) - np.trace(Q_optimal))
    
    # Normalize trace error by the scale of the optimal Q matrix entries
    trace_optimal = np.trace(Q_optimal)
    if abs(trace_optimal) > 1e-10:  # Avoid division by zero
        metrics["trace_diff"] = trace_diff_abs / abs(trace_optimal)
        metrics["trace_diff_abs"] = trace_diff_abs
    else:
        # Fallback: normalize by mean absolute value of matrix entries
        mean_abs_optimal = np.mean(np.abs(Q_optimal))
        if mean_abs_optimal > 1e-10:
            metrics["trace_diff"] = trace_diff_abs / mean_abs_optimal
            metrics["trace_diff_abs"] = trace_diff_abs
        else:
            metrics["trace_diff"] = trace_diff_abs
            metrics["trace_diff_abs"] = trace_diff_abs
    
    # Q-function evaluation on test samples
    if test_samples is not None and poly is not None:
        try:
            x_test, u_test = test_samples
            
            # Transform test samples to polynomial features
            z_test = np.concatenate([x_test, u_test], axis=1)
            phi_test = poly.transform(z_test)
            
            # Evaluate learned Q-function (in feature space)
            Q_learned_values = np.array([phi_test[i] @ Q_learned @ phi_test[i] 
                                       for i in range(len(phi_test))])
            
            # Evaluate optimal Q-function (in original state-action space)
            Q_optimal_values = np.array([z_test[i] @ Q_optimal @ z_test[i] 
                                       for i in range(len(z_test))])
            
            # Compute comparison metrics
            metrics["q_value_diff_mean"] = np.mean(np.abs(Q_learned_values - Q_optimal_values))
            metrics["q_value_diff_std"] = np.std(np.abs(Q_learned_values - Q_optimal_values))
            metrics["q_value_diff_max"] = np.max(np.abs(Q_learned_values - Q_optimal_values))
            metrics["q_value_correlation"] = np.corrcoef(Q_learned_values, Q_optimal_values)[0, 1]
            
            # Relative error metrics
            metrics["q_value_rel_error_mean"] = np.mean(np.abs(Q_learned_values - Q_optimal_values) / 
                                                       (np.abs(Q_optimal_values) + 1e-8))
            metrics["q_value_rel_error_std"] = np.std(np.abs(Q_learned_values - Q_optimal_values) / 
                                                     (np.abs(Q_optimal_values) + 1e-8))
            
        except Exception as e:
            print(f"Error in Q-function evaluation: {e}")
            metrics["q_value_error"] = str(e)
    
    return metrics

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
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                prob.solve(verbose=False)
    
    return prob.status

def run_one(seed, dx, N, gamma, M_offline, degree, rho, exclude_u_squared=False):
    """Generate a dataset and solve both LPs; return boundedness flags and Q quality comparison."""
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create system for this dimension
    system, C_cost = create_point_mass_system(dx, N=0, M=0)
    
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
        "moment_matching_status": status_m2,
        "moment_matching_bounded": is_bounded(status_m2),
        "moment_stage1_status": status_m1,
        "identity_status": status_id,
        "identity_bounded": is_bounded(status_id),
    }
    
    # Add Q quality comparison if moment matching was successful
    if is_bounded(status_m2) and Q_learned is not None and C_val is not None and E_Q_learned is not None:
        try:
            # Compute optimal Q matrix by linearizing the system
            Q_optimal_orig = compute_optimal_Q(system, C_cost, gamma, rho)
            
            if Q_optimal_orig is not None:
                # Generate test samples for Q-function evaluation
                n_test = 1000
                x_test, u_test = auxiliary_samples(system, n_test, seed=seed + 1000)
                test_samples = (x_test, u_test)
                
                # Compare Q matrices using trace difference and Q-function evaluation
                q_metrics = compare_Q_matrices(Q_learned, Q_optimal_orig, test_samples, poly, dx=dx, du=system.N_u)
                
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

def sweep_over_dims(dims, seeds, N, gamma, M_offline, degree, rho, exclude_u_squared=False):
    results = []
    for dx in dims:
        if dx % 2 != 0:
            print(f"[WARN] Skipping dx={dx} (must be even for point mass system)")
            continue
        for s in seeds:
            print(f"Running seed {s} for dx={dx}")
            try:
                results.append(run_one(seed=int(s), dx=int(dx), N=N, gamma=gamma, M_offline=M_offline, degree=degree, rho=rho, exclude_u_squared=exclude_u_squared))
            except Exception as e:
                print(f"[WARN] Failed for dx={dx}, seed={s}: {e}. Skipping.")
                continue
    return results

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Boundedness vs state dimension (nonlinear point mass systems).")
    parser.add_argument("--dims", type=str, default="2,4,6,8,10", help="Comma-separated list of state dimensions (e.g., '2,4,6,8,10'). Default: 2,4,6,8,10")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per dimension (default: 10)")
    parser.add_argument("--N", type=int, default=1000, help="Number of samples for dataset (default: 1000)")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor (default: from config)")
    parser.add_argument("--M_offline", type=int, default=500, help="Offline pool size (default: 500)")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial feature degree (default: 1)")
    parser.add_argument("--rho", type=float, default=RHO_POINT_MASS_2D, help="Control cost weight (default: from config)")
    parser.add_argument("--exclude_u_squared", action="store_true", help="Exclude u^2 terms from polynomial features (for degree 2)")
    parser.add_argument("--out_json", type=str, default=None, help="Raw results JSON (if None, auto-generated with N)")
    parser.add_argument("--plot_dir", type=str, default=None, help="Plot filename (if None, auto-generated with N)")
    parser.add_argument("--boundedness_threshold", type=float, default=50.0, help="Minimum boundedness percentage to include in trace plot (default: 50.0)")
    args = parser.parse_args()
    
    # Parse dimensions
    if args.dims.strip():
        dims = [int(s) for s in args.dims.split(",") if s.strip()]
    else:
        dims = [2, 4, 6, 8, 10]  # Default: dx = 2, 4, 6, 8, 10
    
    # Generate seeds for each experiment
    seeds = list(range(21, args.seeds + 21))
    
    # Auto-generate filenames with N if not provided
    if args.out_json is None:
        args.out_json = f"bounded_vs_dim_results_nonlinear_N_{args.N}.json"
    if args.plot_dir is None:
        args.plot_dir = f"../figures/bounded_vs_dim_percentages_nonlinear_N_{args.N}.pdf"
    
    results = sweep_over_dims(dims, seeds, N=args.N, gamma=args.gamma, M_offline=args.M_offline, degree=args.degree, rho=args.rho, exclude_u_squared=args.exclude_u_squared)
    
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
    # Generate aggregated filename with N
    agg_json_name = f"bounded_vs_dim_percentages_nonlinear_N_{args.N}.json"
    agg.to_json(agg_json_name, orient="records", indent=2)
    
    # Print compact table
    cols = ["dx", "moment_bounded_pct", "identity_bounded_pct"]
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
    axes[0].set_xlabel("State dimension (dx)")
    axes[0].set_ylabel("Bounded LPs (%)")
    axes[0].set_title("Percentage of bounded LP problems vs state dimension\n(Point Mass with Cubic Drag)")
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
            
            axes[1].set_xlabel("State dimension (dx)")
            axes[1].set_ylabel("Relative trace error")
            axes[1].set_title(f"Q matrix trace difference vs dimension\n(≥{threshold}% boundedness)")
            axes[1].grid(True)
            axes[1].legend()
            
            print(f"Trace plot shows {len(filtered_agg)} dimensions with ≥{threshold}% boundedness")
        else:
            axes[1].text(0.5, 0.5, f"No dimensions with ≥{threshold}% boundedness", 
                           ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title(f"Q matrix trace difference vs dimension\n(≥{threshold}% boundedness)")
            print(f"No dimensions found with ≥{threshold}% boundedness")
    else:
        axes[1].text(0.5, 0.5, "No trace difference data available", 
                       ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Q matrix trace difference vs dimension")
    
    plt.tight_layout()
    plt.savefig(args.plot_dir, dpi=150)
    print("Saved:", args.out_json)
    print("Aggregated results saved:", agg_json_name)
    print("Figures saved in", args.plot_dir)
