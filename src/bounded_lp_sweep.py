from random import seed
import numpy as np
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
from utils import ScaleOnlyScaler
from dynamical_systems import cart_pole, dlqr
from policy_extraction import PolicyExtractor
from scipy import sparse
from moment_matching import solve_moment_matching_Q, solve_identity_Q
from config import (
    GAMMA, M_C, M_P, L, DT, C_CART_POLE, RHO_CART_POLE, DEGREE,
    X_BOUNDS, X_DOT_BOUNDS, THETA_BOUNDS, THETA_DOT_BOUNDS, U_BOUNDS,
    DEFAULT_M_OFFLINE, DEFAULT_REGULARIZATION
)

# -------------------------
# Config - Using centralized configuration
# -------------------------
M_c = M_C
M_p = M_P
l = L
dt = DT
gamma = GAMMA  # Using centralized discount factor
C = C_CART_POLE
rho = RHO_CART_POLE

degree = DEGREE

# State/action sampling bounds
x_bounds = X_BOUNDS
x_dot_bounds = X_DOT_BOUNDS
theta_bounds = THETA_BOUNDS
theta_dot_bounds = THETA_DOT_BOUNDS
u_bounds = U_BOUNDS

def run_one(seed, N, M_offline=DEFAULT_M_OFFLINE, use_scaling=True, regularization=DEFAULT_REGULARIZATION, 
                    data_pools=None, test_states=None, scaler=None, poly=None, return_mu=False, fixed_mu=None):
    """Generate data with given seed and N, then solve both LPs and return boundedness flags."""
    # Build system and data
    extractor = PolicyExtractor(degree=degree)
    system = cart_pole(M_c, M_p, l, dt, C, rho, gamma, N, M_offline)

    lqr_policy, _, _ = extractor.extract_lqr_policy(system)

    # Use pre-generated data pools if provided, otherwise generate new data
    if data_pools is not None:
        x, u, x_plus, u_plus = data_pools['x'][:N], data_pools['u'][:N], data_pools['x_plus'][:N], data_pools['u_plus'][:N]
        x_aux, u_aux = data_pools['x_aux'], data_pools['u_aux']
    else:
        x, u, x_plus, u_plus = system.generate_samples(
            x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=N
        )
        # Auxiliary sample for Py
        x_aux, u_aux = system.generate_samples_auxiliary(
            x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=M_offline
        )

    z = np.concatenate([x, u], axis=1)           # (N, dx+du)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)  # (N, dx+du)
    Z_all = np.concatenate([z, z_plus], axis=0)       # (2N, dx+du)
    
    # Compute costs (needed for both scaling and non-scaling cases)
    L_xu = system.cost(x, u)

    # Handle polynomial feature generation based on whether scaling is used
    if use_scaling:
        # Use provided scaler (fitted on full dataset)
        Z_all_scaled = scaler.transform(Z_all)
        
        # poly was fitted on scaled data, so use scaled data
        P_all = poly.transform(Z_all_scaled)
        
        P_z = P_all[:N]
        P_z_next = P_all[N:]
        
        y = np.concatenate([x_aux, u_aux], axis=1)  # (M, dx+du)
        y_scaled = scaler.transform(y)
        P_y = poly.transform(y_scaled)
        
        # Use scaled features for moment matching
        Q_mm, mu, status_info = solve_moment_matching_Q(
            P_z, P_z_next, P_y, L_xu, N, M_offline, gamma,
            regularization=regularization, fixed_mu=fixed_mu
        )

        L_xu = L_xu / scaler.scale_[-1]
    else:
        # poly was fitted on unscaled data, so use unscaled data
        P_all = poly.transform(Z_all)
        
        P_z = P_all[:N]
        P_z_next = P_all[N:]
        
        y = np.concatenate([x_aux, u_aux], axis=1)  # (M, dx+du)
        P_y = poly.transform(y)
        
        # Use original features for moment matching
        Q_mm, mu, status_info = solve_moment_matching_Q(
            P_z, P_z_next, P_y, L_xu, N, M_offline, gamma,
            regularization=regularization, fixed_mu=fixed_mu
        )
        # print(f"eigenvalues of Q_mm: {np.linalg.eigvals(Q_mm)}")
    
    status_m2 = status_info["stage2_status"]
    status_m1 = status_info["stage1_status"]

    # Extract policy using the analytical method
    if Q_mm is not None:
        # # Check value function positivity before extracting policy
        # print("\n=== Checking Value Function Positivity ===")
        positivity_results = extractor.check_value_function_positivity(
            Q_mm, poly, scaler, n_samples=1000, verbose=False
        )
        
        if use_scaling and scaler is not None:
            # print("Using analytical policy extraction with scaling")
            mm_policy = extractor.extract_moment_matching_policy_analytical(
                system, Q_mm, poly, scaler
            )
            # print("Using scipy optimization policy extraction with scaling")
            # mm_policy = extractor.extract_policy_scipy_optimization(
            #     system, Q_mm, poly, scaler
            # )
        else:
            # print("Using analytical policy extraction without scaling")
            mm_policy = extractor.extract_moment_matching_policy_analytical(
                system, Q_mm, poly
            )
            # print("Using scipy optimization policy extraction without scaling")
            # mm_policy = extractor.extract_policy_scipy_optimization(
            #     system, Q_mm, poly
            # )
    else:
        # Create a dummy policy that always returns 0 if moment matching failed
        print("Moment matching failed")
        mm_policy = None
        positivity_results = {
            "all_positive": False,
            "positive_ratio": 0.0,
            "min_value": np.inf,
            "max_value": -np.inf,
            "mean_value": np.nan,
            "n_samples": 0,
            "error": "Q_matrix is None"
        }

    # Identity baseline
    status_id = solve_identity_Q(P_z, P_z_next, L_xu, N, gamma)

    # Generate test states (use provided test_states or generate new ones)
    if test_states is None:
        test_states, _ = system.generate_samples_auxiliary(
            (-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5),
            extractor.u_bounds, n_samples=50
        )

    if mm_policy is not None:
        results_policy = extractor.compare_policies(
            system, lqr_policy, mm_policy, test_states, horizon=3000
        )
    else:
        print("Moment matching failed")
        results_policy = {
            "lqr_success": 0.0,
            "mm_success": 0.0,
            "lqr_costs": np.inf,
            "mm_costs": np.inf
        }

    # Interpret boundedness:
    # "optimal"/"optimal_inaccurate" -> bounded & feasible
    # "unbounded" -> unbounded
    # anything else (infeasible, error) -> treat as NOT bounded
    def is_bounded(status):
        return status in ("optimal", "optimal_inaccurate")

    print("N: ", N)
    print("Seed: ", seed)
    print("LQR success: ", results_policy["lqr_success"])
    print("MM success: ", results_policy["mm_success"])
    print("LQR costs: ", results_policy["lqr_costs"])
    print("MM costs: ", results_policy["mm_costs"])

    result = {
        "seed": seed,
        "N": N,
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
        # Additional metrics from comparison
        "lqr_costs_std": results_policy.get("lqr_costs_std", 0.0),
        "mm_costs_std": results_policy.get("mm_costs_std", 0.0),
        "lqr_costs_median": results_policy.get("lqr_costs_median", 0.0),
        "mm_costs_median": results_policy.get("mm_costs_median", 0.0),
        # Value function positivity check results
        "value_function_all_positive": positivity_results["all_positive"],
        "value_function_positive_ratio": positivity_results["positive_ratio"],
        "value_function_min_value": positivity_results["min_value"],
        "value_function_max_value": positivity_results["max_value"],
        "value_function_mean_value": positivity_results["mean_value"],
        "value_function_n_samples": positivity_results["n_samples"],
    }
    
    if return_mu:
        return result, mu
    else:
        return result

def sweep(N_values=range(50, 501, 50), seeds=range(10), M_offline=DEFAULT_M_OFFLINE, 
                  use_scaling=True, regularization=DEFAULT_REGULARIZATION, fix_mu_from_first_n=False):
    results = []
    
    # Find the maximum N to generate data pools
    N_max = max(N_values)
    
    for s in seeds:
        seed = int(s)
        print(f"Processing seed {seed}...")
        
        # Generate data pools once per seed for nested datasets
        np.random.seed(seed)
        extractor = PolicyExtractor(degree=degree)
        system_temp = cart_pole(M_c, M_p, l, dt, C, rho, gamma, N_max, M_offline)
        
        # Generate large pool of training data
        x_pool, u_pool, x_plus_pool, u_plus_pool = system_temp.generate_samples(
            x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=N_max
        )
        
        # Generate auxiliary data (fixed for all N)
        x_aux, u_aux = system_temp.generate_samples_auxiliary(
            x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=M_offline
        )
        
        # Generate test states (fixed for all N)
        test_states, _ = system_temp.generate_samples_auxiliary(
            (-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5),
            extractor.u_bounds, n_samples=50
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
        
        # Fit scaler on full dataset if using scaling
        scaler = None
        if use_scaling:
            # Create scale-only scaler (no centering, only division by std)
            scaler = ScaleOnlyScaler()
            scaler.fit(Z_pool)
            Z_pool_scaled = scaler.transform(Z_pool)
            print(f"  Fitted scale-only scaler on {len(Z_pool)} samples for seed {seed}")
        else:
            Z_pool_scaled = Z_pool
        
        # Fit polynomial features on the appropriately scaled data
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(Z_pool_scaled)  # Fit on scaled data if scaling is used
        
        print(f"  Generated data pools: {len(x_pool)} training samples, {len(x_aux)} auxiliary samples, {len(test_states)} test states")
        
        # Run experiments for each N using nested datasets
        mu_fixed = None
        for i, N in enumerate(N_values):
            if i == 0 and fix_mu_from_first_n:
                # For the first N, compute mu normally and store it
                print(f"Computing mu for first N={N}, seed={seed}")
                result = run_one(
                    seed=seed, N=int(N), M_offline=M_offline, 
                    use_scaling=use_scaling, regularization=regularization,
                    data_pools=data_pools, test_states=test_states, scaler=scaler, poly=poly,
                    return_mu=True
                )
                if result is not None and len(result) == 2:
                    result_dict, mu = result
                    mu_fixed = mu
                    print(f"Fixed mu computed with shape {mu_fixed.shape}")
                    results.append(result_dict)
                else:
                    print(f"Failed to compute mu for first N, proceeding without fixed mu")
                    results.append(run_one(
                        seed=seed, N=int(N), M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, scaler=scaler, poly=poly
                    ))
            else:
                # For subsequent N values, use the fixed mu
                if mu_fixed is not None and fix_mu_from_first_n:
                    print(f"Using fixed mu for N={N}, seed={seed}")
                    results.append(run_one(
                        seed=seed, N=int(N), M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, scaler=scaler, poly=poly,
                        fixed_mu=mu_fixed
                    ))
                else:
                    # Fallback to normal computation
                    results.append(run_one(
                        seed=seed, N=int(N), M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, scaler=scaler, poly=poly
                    ))
    
    return results

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Percentage boundedness of LPs vs N.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per N (default: 10)")
    parser.add_argument("--Nmin", type=int, default=250, help="Min N (default: 100)")
    parser.add_argument("--Nmax", type=int, default=5000, help="Max N (default: 2000)")
    parser.add_argument("--Nstep", type=int, default=250, help="Step for N (default: 100)")
    parser.add_argument("--out_json", type=str, default="../results/test_analytical_noscaling_deg1_1e-2_mu_sum_1000M_mix.json", help="Where to save raw results JSON")
    parser.add_argument("--plot_png", type=str, default="../figures/test_analytical_noscaling_deg1_1e-2_mu_sum_1000M_mix.pdf", help="Where to save the plot")
    parser.add_argument("--plot_policy", type=str, default="../figures/test_policy_analytical_noscaling_deg1_1e-2_mu_sum_1000M_mix.pdf", help="Where to save the policy comparison plot")
    parser.add_argument("--M_offline", type=int, default=1000, help="Offline pool size (default: 500)")
    parser.add_argument("--use_scaling", action="store_true", default=False, help="Use feature scaling (default: False)")
    parser.add_argument("--regularization", type=float, default=1e-4, help="Regularization parameter (default: 1e-4)")
    parser.add_argument("--fix_mu", action="store_true", default=False, help="Fix mu from first N and reuse for all subsequent N values")
    
    args = parser.parse_args()
    N_values = list(range(args.Nmin, args.Nmax + 1, args.Nstep))
    seeds = list(range(args.seeds))

    try:
        results = sweep(
            N_values=N_values, seeds=seeds, M_offline=args.M_offline,
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
    
    agg = df.groupby("N").agg(
        moment_bounded_pct=("moment_matching_bounded", "mean"),
        identity_bounded_pct=("identity_bounded", "mean"),
        moment_bounded_pct_min=("moment_matching_bounded", "min"),
        identity_bounded_pct_min=("identity_bounded", "min"),
        moment_bounded_pct_max=("moment_matching_bounded", "max"),
        identity_bounded_pct_max=("identity_bounded", "max"),
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
    ).reset_index()

    # Convert to percentages
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_pct"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_pct"]
    agg["moment_bounded_pct_min"] = 100 * agg["moment_bounded_pct_min"]
    agg["identity_bounded_pct_min"] = 100 * agg["identity_bounded_pct_min"]
    agg["moment_bounded_pct_max"] = 100 * agg["moment_bounded_pct_max"]
    agg["identity_bounded_pct_max"] = 100 * agg["identity_bounded_pct_max"]

    # Save raw + aggregated
    df.to_json(args.out_json, orient="records", indent=2)
    agg.to_json("bounded_lp_percentages.json", orient="records", indent=2)

    # Plot
    import matplotlib.pyplot as plt

    plt.figure()
    
    # Plot moment matching with shaded area
    plt.plot(agg["N"], agg["moment_bounded_pct"], marker="o", label="Moment matching", color="blue")
    plt.fill_between(agg["N"], agg["moment_bounded_pct_min"], agg["moment_bounded_pct_max"], 
                     alpha=0.3, color="blue")
    
    # Plot identity covariance with shaded area
    plt.plot(agg["N"], agg["identity_bounded_pct"], marker="s", label="Identity covariance", color="red")
    plt.fill_between(agg["N"], agg["identity_bounded_pct_min"], agg["identity_bounded_pct_max"], 
                     alpha=0.3, color="red")
    
    plt.xlabel("N (number of samples)")
    plt.ylabel("Bounded LPs (%)")
    plt.title("Percentage of bounded LP problems vs sample size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_png, dpi=150)
    print("Saved:", args.out_json, "and", args.plot_png)

    # Plot policy comparison with shaded areas
    plt.figure()
    
    # Plot LQR with shaded area
    plt.plot(agg["N"], agg["lqr_costs"], 
                marker="o", label="LQR", color="green")
    plt.fill_between(agg["N"], agg["lqr_costs_min"], agg["lqr_costs_max"], 
                     alpha=0.3, color="green")
    
    # Plot moment matching with shaded area
    plt.plot(agg["N"], agg["mm_costs"], 
                marker="s", label="Moment matching", color="orange")
    plt.fill_between(agg["N"], agg["mm_costs_min"], agg["mm_costs_max"], 
                     alpha=0.3, color="orange")
    
    plt.xlabel("N (number of samples)")
    plt.ylabel("Cost")
    plt.title("Policy comparison vs sample size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_policy, dpi=150)
    print("Saved policy comparison plot:", args.plot_policy)
