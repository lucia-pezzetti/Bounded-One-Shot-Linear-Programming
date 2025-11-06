import numpy as np
import json
from sklearn.preprocessing import PolynomialFeatures
from utils import RobustBlockScaleOnlyScaler
from feature_scaling import PolynomialFeatureScaler, create_robust_polynomial_features
from dynamical_systems import dlqr
from policy_extraction import PolicyExtractor
from moment_matching import solve_moment_matching_Q, solve_identity_Q
from bounded_lp_vs_dim_linear import compute_optimal_Q
from config import (
    GAMMA, DEGREE, DEFAULT_M_OFFLINE, DEFAULT_REGULARIZATION
)

# -------------------------
# Config - Using centralized configuration
# -------------------------
gamma = GAMMA  # Using centralized discount factor
degree = DEGREE

# LQR system parameters - will be loaded from data files
def load_lqr_system(dx):
    """Load LQR system matrices from data files"""
    data_file = f"data/dx_{dx}_du_{1}.json"
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    A = np.array(data['A'])
    B = np.array(data['B'])
    C = np.array(data['C'])
    
    return A, B, C

# State/action sampling bounds for LQR systems
def get_lqr_bounds(dx, du):
    """Get appropriate bounds for LQR system based on dimensions"""
    # Use reasonable bounds for LQR systems
    x_bounds = (-3.0, 3.0)  # State bounds
    u_bounds = (-1.0, 1.0)  # Control bounds
    
    return x_bounds, u_bounds

def run_one(seed, N, dx, du, M_offline=DEFAULT_M_OFFLINE, use_scaling=True, regularization=DEFAULT_REGULARIZATION, 
                    data_pools=None, test_states=None, poly_scaler=None, return_mu=False, fixed_mu=None):
    """Generate data with given seed and N, then solve both LPs and return boundedness flags."""
    # Load LQR system matrices
    A, B, C = load_lqr_system(dx)
    
    # Get bounds for this system
    x_bounds, u_bounds = get_lqr_bounds(dx, du)
    
    # Build system and data
    system = dlqr(A, B, C, rho=0.01, gamma=gamma, sigma=0.0)

    # Use pre-generated data pools if provided, otherwise generate new data
    if data_pools is not None:
        x, u, x_plus, u_plus = data_pools['x'][:N], data_pools['u'][:N], data_pools['x_plus'][:N], data_pools['u_plus'][:N]
        x_aux, u_aux = data_pools['x_aux'], data_pools['u_aux']
    else:
        x, u, x_plus, u_plus = system.generate_samples(
            x_bounds, u_bounds, n_samples=N
        )
        # Auxiliary sample for Py
        x_aux, u_aux = system.generate_samples_auxiliary(
            x_bounds, u_bounds, n_samples=M_offline
        )

    z = np.concatenate([x, u], axis=1)           # (N, dx+du)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)  # (N, dx+du)
    Z_all = np.concatenate([z, z_plus], axis=0)       # (2N, dx+du)
    
    # Compute costs (needed for both scaling and non-scaling cases)
    L_xu = system.cost(x, u)

    # Handle polynomial feature generation based on whether robust scaler is used
    if poly_scaler is not None:
        # Use robust polynomial feature scaler (for degree 2+)
        P_all = poly_scaler.transform(Z_all)
        
        P_z = P_all[:N]
        P_z_next = P_all[N:]
        
        y = np.concatenate([x_aux, u_aux], axis=1)  # (M, dx+du)
        P_y = poly_scaler.transform(y)
        
        # Use robust scaled features for moment matching
        Q_mm, mu, status_info = solve_moment_matching_Q(
            P_z, P_z_next, P_y, L_xu, N, M_offline, gamma,
            regularization=regularization, fixed_mu=fixed_mu
        )
    else:
        raise ValueError("Scaling method not supported")
    
    status_m2 = status_info["stage2_status"]
    status_m1 = status_info["stage1_status"]

    def is_bounded(status):
        return status in ("optimal", "optimal_inaccurate")

    # Extract policy using the analytical method
    if Q_mm is not None:
        # Create PolicyExtractor for LQR system
        extractor = PolicyExtractor(degree=degree, u_bounds=u_bounds)

        def extract_lqr_policy(system):
            P_lqr, K_lqr, q_lqr = system.optimal_solution()
            def lqr_policy(x):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                u = -K_lqr @ x.T  # Standard LQR control law: u = -K @ x
                # Handle clipping for multi-dimensional control
                u_clipped = np.clip(u.T, u_bounds[0], u_bounds[1])
                # Always return as 1D array
                return u_clipped.flatten()
            return lqr_policy
        
        # Extract LQR policy
        lqr_policy = extract_lqr_policy(system)
        
        # Extract moment matching policy
        if poly_scaler is not None:
            print("Extracting moment matching policy with polynomial scaler")
            # Use robust polynomial scaler for policy extraction
            mm_policy = extractor.extract_moment_matching_policy_analytical(
                system, Q_mm, poly_scaler
            )
        else:
            # Create a dummy policy that always returns 0 if moment matching failed
            print("Moment matching failed")
            lqr_policy = None
            mm_policy = None

    # Identity baseline
    status_id = solve_identity_Q(P_z, P_z_next, L_xu, N, gamma)

    # Generate test states (use provided test_states or generate new ones)
    if test_states is None:
        test_states, _ = system.generate_samples_auxiliary(
            x_bounds, u_bounds, n_samples=10
        )

    # Policy comparison
    if lqr_policy is not None and mm_policy is not None:
        results_policy = extractor.compare_policies(
            system, lqr_policy, mm_policy, test_states, N, horizon=500
        )
    else:
        print("Policy extraction failed")
        results_policy = {
            "lqr_success": 0.0,
            "mm_success": 0.0,
            "lqr_costs": np.inf,
            "mm_costs": np.inf
        }

    # Compute Q-matrix comparison metrics
    Q_optimal_orig = compute_optimal_Q(A, B, C, gamma, 0.01)
    
    # Initialize comparison metrics
    trace_diff = np.nan
    frobenius_diff = np.nan
    relative_trace_error = np.nan
    relative_frobenius_error = np.nan
    
    if Q_mm is not None and Q_optimal_orig is not None:
        try:
            # Transform optimal Q-matrix to polynomial feature space for comparison
            # Generate test samples to evaluate Q-functions
            n_test = 1000
            np.random.seed(seed + 1000)  # Different seed for test samples
            x_test = np.random.uniform(x_bounds[0], x_bounds[1], size=(n_test, dx))
            u_test = np.random.uniform(u_bounds[0], u_bounds[1], size=(n_test, du))
            test_samples = np.concatenate([x_test, u_test], axis=1)
            
            # Transform test samples to polynomial features
            if poly_scaler is not None:
                test_features = poly_scaler.transform(test_samples)
            else:
                raise ValueError("poly_scaler is required for Q-matrix comparison")
            
            # Evaluate Q-functions on test samples
            q_mm_values = np.array([test_features[i] @ Q_mm @ test_features[i].T for i in range(n_test)]).flatten()
            
            # Evaluate optimal Q-function on test samples
            q_optimal_values = np.array([test_samples[i] @ Q_optimal_orig @ test_samples[i].T for i in range(n_test)]).flatten()
            
            # Compute comparison metrics based on Q-function values
            trace_diff = np.mean(np.abs(q_mm_values - q_optimal_values))
            frobenius_diff = np.linalg.norm(q_mm_values - q_optimal_values) / np.sqrt(n_test)
            
            # Compute relative errors
            if np.mean(np.abs(q_optimal_values)) > 1e-10:
                relative_trace_error = trace_diff / np.mean(np.abs(q_optimal_values))
            if np.linalg.norm(q_optimal_values) > 1e-10:
                relative_frobenius_error = frobenius_diff / (np.linalg.norm(q_optimal_values) / np.sqrt(n_test))
            
            # Print comparison for debugging
            print(f"  Q-matrix comparison (N={N}, seed={seed}):")
            print(f"    Mean Q-value - MM: {np.mean(q_mm_values):.6f}, Optimal: {np.mean(q_optimal_values):.6f}")
            print(f"    Mean absolute difference: {trace_diff:.6f}")
            print(f"    RMS difference: {frobenius_diff:.6f}")
            print(f"    Relative mean error: {relative_trace_error:.6f}")
            print(f"    Relative RMS error: {relative_frobenius_error:.6f}")
            
        except Exception as e:
            print(f"  Error computing Q-matrix comparison: {e}")
            import traceback
            traceback.print_exc()

    result = {
        "seed": seed,
        "N": N,
        "dx": dx,
        "du": du,
        "moment_matching_status": status_m2,
        "moment_matching_bounded": is_bounded(status_m2),
        "moment_stage1_status": status_m1,
        "identity_status": status_id,
        "identity_bounded": is_bounded(status_id),
        "lqr_success": results_policy["lqr_success"],
        "lqr_costs": results_policy["lqr_costs"],
        "mm_success": results_policy["mm_success"],
        "mm_costs": results_policy["mm_costs"],
        "use_scaling": use_scaling,
        "regularization": regularization,
        "Q_mm": Q_mm,
        "Q_optimal": Q_optimal_orig,
        "trace_diff": trace_diff,
        "frobenius_diff": frobenius_diff,
        "relative_trace_error": relative_trace_error,
        "relative_frobenius_error": relative_frobenius_error,
        # Additional metrics from comparison
        "lqr_costs_std": results_policy.get("lqr_costs_std", 0.0),
        "mm_costs_std": results_policy.get("mm_costs_std", 0.0),
        "lqr_costs_median": results_policy.get("lqr_costs_median", 0.0),
        "mm_costs_median": results_policy.get("mm_costs_median", 0.0),
    }
    
    if return_mu:
        return result, mu
    else:
        return result

def sweep(dx_values=[2, 5, 10, 20], N_values=range(50, 501, 50), seeds=range(10), M_offline=DEFAULT_M_OFFLINE, 
                  use_scaling=True, regularization=DEFAULT_REGULARIZATION, fix_mu_from_first_n=False):
    results = []
    
    # Find the maximum N to generate data pools
    N_max = max(N_values)
    
    for dx in dx_values:
        print(f"Processing dx={dx}...")
        
        # Load LQR system for this dimension
        A, B, C = load_lqr_system(dx)
        du = B.shape[1]  # Get control dimension from B matrix
        
        for s in seeds:
            seed = int(s)
            print(f"  Processing seed {seed}...")
            
            # Generate data pools once per seed for nested datasets
            np.random.seed(seed)
            
            # Get bounds for this system
            x_bounds, u_bounds = get_lqr_bounds(dx, du)
            
            system_temp = dlqr(A, B, C, rho=0.01, gamma=gamma, sigma=0.0)
            
            # Generate large pool of training data
            x_pool, u_pool, x_plus_pool, u_plus_pool = system_temp.generate_samples(
                x_bounds, u_bounds, n_samples=N_max
            )
            
            # Generate auxiliary data (fixed for all N)
            x_aux, u_aux = system_temp.generate_samples_auxiliary(
                x_bounds, u_bounds, n_samples=M_offline
            )
            
            # Generate test states (fixed for all N)
            test_states, _ = system_temp.generate_samples_auxiliary(
                x_bounds, u_bounds, n_samples=30
            )
            
            # Create data pools dictionary
            data_pools = {
                'x': x_pool,
                'u': u_pool, 
                'x_plus': x_plus_pool,
                'u_plus': u_plus_pool,
                'x_aux': x_aux,
                'u_aux': u_aux
            }

            # Fit polynomial features ONCE on the full dataset
            z_pool = np.concatenate([x_pool, u_pool], axis=1)
            z_plus_pool = np.concatenate([x_plus_pool, u_plus_pool], axis=1)
            Z_pool = np.concatenate([z_pool, z_plus_pool], axis=0)
            
            # Use robust polynomial feature scaler
            if use_scaling:
                # Use robust polynomial feature scaler with separate scaling for states and actions
                # For degree 1, use standard scaling (with centering) to ensure position is centered around 0
                # For degree 2+, use robust scaling for better numerical stability
                scaling_method = 'standard' if degree == 1 else 'robust'
                
                poly_scaler = PolynomialFeatureScaler(
                    degree=degree, 
                    scaling_method=scaling_method,
                    dx=dx,
                    du=du
                )
                P_pool = poly_scaler.fit_transform(Z_pool)
                print(f"    Fitted {scaling_method} polynomial scaler (degree={degree}, dx={dx}, du={du}) on {len(Z_pool)} samples for seed {seed}")
            else:
                poly_scaler = PolynomialFeatureScaler(
                    degree=degree, 
                    scaling_method='none'
                )
                P_pool = poly_scaler.fit_transform(Z_pool)
                print(f"    Fitted polynomial scaler (degree={degree}) on {len(Z_pool)} samples for seed {seed}")
            
            print(f"    Generated data pools: {len(x_pool)} training samples, {len(x_aux)} auxiliary samples, {len(test_states)} test states")
            
            # Run experiments for each N using nested datasets
            mu_fixed = None
            for i, N in enumerate(N_values):
                if i == 0 and fix_mu_from_first_n:
                    # For the first N, compute mu normally and store it
                    print(f"    Computing mu for first N={N}, seed={seed}")
                    result = run_one(
                        seed=seed, N=int(N), dx=dx, du=du, M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler,
                        return_mu=True
                    )
                    if result is not None and len(result) == 2:
                        result_dict, mu = result
                        mu_fixed = mu
                        print(f"    Fixed mu computed with shape {mu_fixed.shape}")
                        results.append(result_dict)
                    else:
                        print(f"    Failed to compute mu for first N, proceeding without fixed mu")
                        results.append(run_one(
                            seed=seed, N=int(N), dx=dx, du=du, M_offline=M_offline, 
                            use_scaling=use_scaling, regularization=regularization,
                            data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler
                        ))
                else:
                    # For subsequent N values, use the fixed mu
                    if mu_fixed is not None and fix_mu_from_first_n:
                        print(f"    Using fixed mu for N={N}, seed={seed}")
                        results.append(run_one(
                            seed=seed, N=int(N), dx=dx, du=du, M_offline=M_offline, 
                            use_scaling=use_scaling, regularization=regularization,
                            data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler,
                            fixed_mu=mu_fixed
                        ))
                    else:
                        # Fallback to normal computation
                        results.append(run_one(
                            seed=seed, N=int(N), dx=dx, du=du, M_offline=M_offline, 
                            use_scaling=use_scaling, regularization=regularization,
                            data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler
                        ))
    
    return results

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Percentage boundedness of LPs vs N for LQR systems.")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds per N (default: 10)")
    parser.add_argument("--Nmin", type=int, default=500, help="Min N (default: 100)")
    parser.add_argument("--Nmax", type=int, default=5000, help="Max N (default: 2000)")
    parser.add_argument("--Nstep", type=int, default=250, help="Step for N (default: 100)")
    parser.add_argument("--dx_values", type=int, nargs='+', default=[4], help="State dimensions to test (default: [2, 5, 10, 20])")
    parser.add_argument("--out_json", type=str, default="../results/test_lqr_analytical_noscaling_deg1_1e-1_mu_sum_1000M.json", help="Where to save raw results JSON")
    parser.add_argument("--plot_png", type=str, default="../figures/test_lqr_analytical_noscaling_deg1_1e-1_mu_sum_1000M.pdf", help="Where to save the plot")
    parser.add_argument("--plot_policy", type=str, default="../figures/test_lqr_policy_analytical_noscaling_deg1_1e-2_mu_sum_1000M.pdf", help="Where to save the policy comparison plot")
    parser.add_argument("--M_offline", type=int, default=500, help="Offline pool size (default: 500)")
    parser.add_argument("--use_scaling", action="store_true", default=False, help="Use feature scaling (default: False)")
    parser.add_argument("--regularization", type=float, default=1e-4, help="Regularization parameter (default: 1e-4)")
    parser.add_argument("--fix_mu", action="store_true", default=False, help="Fix mu from first N and reuse for all subsequent N values")
    
    args = parser.parse_args()
    N_values = list(range(args.Nmin, args.Nmax + 1, args.Nstep))
    seeds = list(range(args.seeds))
    # seeds = [s+6 for s in range(args.seeds)]

    try:
        results = sweep(
            dx_values=args.dx_values, N_values=N_values, seeds=seeds, M_offline=args.M_offline,
            use_scaling=args.use_scaling, regularization=args.regularization,
            fix_mu_from_first_n=args.fix_mu
        )
    except Exception as e:
        # If the user's environment is missing dependencies (e.g., MOSEK, dynamical_systems),
        # we surface a helpful error message instead of crashing silently.
        import sys, traceback
        print("ERROR while running sweep:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

    # Aggregate
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Define custom aggregation functions that exclude np.inf values
    def mean_no_inf(series):
        finite_values = series[np.isfinite(series)]
        return finite_values.mean() if len(finite_values) > 0 else np.nan
    
    def min_no_inf(series):
        finite_values = series[np.isfinite(series)]
        return finite_values.min() if len(finite_values) > 0 else np.nan
    
    def max_no_inf(series):
        finite_values = series[np.isfinite(series)]
        return finite_values.max() if len(finite_values) > 0 else np.nan
    
    agg = df.groupby(["dx", "N"]).agg(
        moment_bounded_pct=("moment_matching_bounded", "mean"),
        moment_bounded_pct_min=("moment_matching_bounded", "min"),
        moment_bounded_pct_max=("moment_matching_bounded", "max"),
        identity_bounded_pct=("identity_bounded", "mean"),
        identity_bounded_pct_min=("identity_bounded", "min"),
        identity_bounded_pct_max=("identity_bounded", "max"),
        # Policy comparison metrics
        lqr_costs=("lqr_costs", mean_no_inf),
        mm_costs=("mm_costs", mean_no_inf),
        lqr_costs_min=("lqr_costs", min_no_inf),
        mm_costs_min=("mm_costs", min_no_inf),
        lqr_costs_max=("lqr_costs", max_no_inf),
        mm_costs_max=("mm_costs", max_no_inf),
        lqr_costs_std=("lqr_costs_std", mean_no_inf),
        mm_costs_std=("mm_costs_std", mean_no_inf),
        lqr_costs_median=("lqr_costs_median", mean_no_inf),
        mm_costs_median=("mm_costs_median", mean_no_inf),
        # Q-matrix comparison metrics
        trace_diff_mean=("trace_diff", mean_no_inf),
        trace_diff_min=("trace_diff", min_no_inf),
        trace_diff_max=("trace_diff", max_no_inf),
        frobenius_diff_mean=("frobenius_diff", mean_no_inf),
        frobenius_diff_min=("frobenius_diff", min_no_inf),
        frobenius_diff_max=("frobenius_diff", max_no_inf),
        relative_trace_error_mean=("relative_trace_error", mean_no_inf),
        relative_trace_error_min=("relative_trace_error", min_no_inf),
        relative_trace_error_max=("relative_trace_error", max_no_inf),
        relative_frobenius_error_mean=("relative_frobenius_error", mean_no_inf),
        relative_frobenius_error_min=("relative_frobenius_error", min_no_inf),
        relative_frobenius_error_max=("relative_frobenius_error", max_no_inf),
    ).reset_index()

    # Convert to percentages
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_pct"]
    agg["moment_bounded_pct_min"] = 100 * agg["moment_bounded_pct_min"]
    agg["moment_bounded_pct_max"] = 100 * agg["moment_bounded_pct_max"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_pct"]
    agg["identity_bounded_pct_min"] = 100 * agg["identity_bounded_pct_min"]
    agg["identity_bounded_pct_max"] = 100 * agg["identity_bounded_pct_max"]

    # Save raw + aggregated
    df.to_json(args.out_json, orient="records", indent=2)
    agg.to_json("bounded_lp_percentages_lqr.json", orient="records", indent=2)

    # Plot
    import matplotlib.pyplot as plt

    # Plot boundedness vs N for different dx values
    plt.figure(figsize=(12, 8))
    
    for dx in args.dx_values:
        dx_data = agg[agg['dx'] == dx]
        
        # Plot moment matching with shaded area
        plt.plot(dx_data["N"], dx_data["moment_bounded_pct"], 
                marker="o", label=f"Moment matching (dx={dx})", linewidth=2)
        plt.fill_between(dx_data["N"], dx_data["moment_bounded_pct_min"], dx_data["moment_bounded_pct_max"], 
                         alpha=0.3)
        
        # Plot identity covariance with shaded area
        plt.plot(dx_data["N"], dx_data["identity_bounded_pct"], 
                marker="s", label=f"Identity covariance (dx={dx})", linewidth=2)
        plt.fill_between(dx_data["N"], dx_data["identity_bounded_pct_min"], dx_data["identity_bounded_pct_max"], 
                         alpha=0.3)
    
    plt.xlabel("N (number of samples)")
    plt.ylabel("Bounded LPs (%)")
    plt.title("Percentage of bounded LP problems vs sample size for LQR systems")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_png, dpi=150)
    print("Saved:", args.out_json, "and", args.plot_png)

    # Plot Q-matrix trace difference comparison
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Trace difference
    plt.subplot(2, 2, 1)
    for dx in args.dx_values:
        dx_data = agg[agg['dx'] == dx]
        plt.plot(dx_data["N"], dx_data["trace_diff_mean"], 
                marker="o", label=f"dx={dx}", linewidth=2)
        plt.fill_between(dx_data["N"], dx_data["trace_diff_min"], dx_data["trace_diff_max"], 
                         alpha=0.3)
    plt.xlabel("N (number of samples)")
    plt.ylabel("Mean Absolute Q-value Difference")
    plt.title("Q-function Value Difference vs Sample Size")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Subplot 2: Relative trace error
    plt.subplot(2, 2, 2)
    for dx in args.dx_values:
        dx_data = agg[agg['dx'] == dx]
        plt.plot(dx_data["N"], dx_data["relative_trace_error_mean"], 
                marker="o", label=f"dx={dx}", linewidth=2)
        plt.fill_between(dx_data["N"], dx_data["relative_trace_error_min"], dx_data["relative_trace_error_max"], 
                         alpha=0.3)
    plt.xlabel("N (number of samples)")
    plt.ylabel("Relative Mean Q-value Error")
    plt.title("Relative Mean Q-value Error vs Sample Size")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Subplot 3: Frobenius norm difference
    plt.subplot(2, 2, 3)
    for dx in args.dx_values:
        dx_data = agg[agg['dx'] == dx]
        plt.plot(dx_data["N"], dx_data["frobenius_diff_mean"], 
                marker="o", label=f"dx={dx}", linewidth=2)
        plt.fill_between(dx_data["N"], dx_data["frobenius_diff_min"], dx_data["frobenius_diff_max"], 
                         alpha=0.3)
    plt.xlabel("N (number of samples)")
    plt.ylabel("RMS Q-value Difference")
    plt.title("RMS Q-value Difference vs Sample Size")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Subplot 4: Relative Frobenius error
    plt.subplot(2, 2, 4)
    for dx in args.dx_values:
        dx_data = agg[agg['dx'] == dx]
        plt.plot(dx_data["N"], dx_data["relative_frobenius_error_mean"], 
                marker="o", label=f"dx={dx}", linewidth=2)
        plt.fill_between(dx_data["N"], dx_data["relative_frobenius_error_min"], dx_data["relative_frobenius_error_max"], 
                         alpha=0.3)
    plt.xlabel("N (number of samples)")
    plt.ylabel("Relative RMS Q-value Error")
    plt.title("Relative RMS Q-value Error vs Sample Size")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(args.plot_policy, dpi=150)
    print("Saved Q-matrix comparison plot:", args.plot_policy)
    
    # Plot policy comparison with shaded areas
    plt.figure(figsize=(12, 8))
    
    for dx in args.dx_values:
        dx_data = agg[agg['dx'] == dx]
        
        # Plot LQR with shaded area
        plt.plot(dx_data["N"], dx_data["lqr_costs"], 
                marker="o", label=f"LQR (dx={dx})", linewidth=2)
        plt.fill_between(dx_data["N"], dx_data["lqr_costs_min"], dx_data["lqr_costs_max"], 
                         alpha=0.3)
        
        # Plot moment matching with shaded area
        plt.plot(dx_data["N"], dx_data["mm_costs"], 
                marker="s", label=f"Moment matching (dx={dx})", linewidth=2)
        plt.fill_between(dx_data["N"], dx_data["mm_costs_min"], dx_data["mm_costs_max"], 
                         alpha=0.3)
    
    plt.xlabel("N (number of samples)")
    plt.ylabel("Cost")
    plt.title("Policy comparison vs sample size for LQR systems")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_policy.replace("_policy", "_policy_comparison"), dpi=150)
    print("Saved policy comparison plot:", args.plot_policy.replace("_policy", "_policy_comparison"))
    
    # Print summary statistics
    print("\n=== Q-matrix Comparison Summary ===")
    for dx in args.dx_values:
        dx_data = agg[agg['dx'] == dx]
        print(f"\ndx={dx}:")
        print(f"  Average Q-value difference: {dx_data['trace_diff_mean'].mean():.6f}")
        print(f"  Average relative Q-value error: {dx_data['relative_trace_error_mean'].mean():.6f}")
        print(f"  Average RMS Q-value difference: {dx_data['frobenius_diff_mean'].mean():.6f}")
        print(f"  Average relative RMS Q-value error: {dx_data['relative_frobenius_error_mean'].mean():.6f}")
    
    print("\n=== Policy Comparison Summary ===")
    for dx in args.dx_values:
        dx_data = agg[agg['dx'] == dx]
        print(f"\ndx={dx}:")
        print(f"  Average LQR costs: {dx_data['lqr_costs'].mean():.6f}")
        print(f"  Average MM costs: {dx_data['mm_costs'].mean():.6f}")
        print(f"  Average LQR costs std: {dx_data['lqr_costs_std'].mean():.6f}")
        print(f"  Average MM costs std: {dx_data['mm_costs_std'].mean():.6f}")