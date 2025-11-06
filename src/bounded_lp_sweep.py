import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from feature_scaling import PolynomialFeatureScaler
from dynamical_systems import cart_pole
from policy_extraction import PolicyExtractor
from moment_matching import solve_identity_Q
from moment_matching import solve_moment_matching_Q
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
                    data_pools=None, test_states=None, poly_scaler=None, return_mu=False, fixed_mu=None):
    """Generate data with given seed and N, then solve both LPs and return boundedness flags."""
    # Build system and data
    extractor = PolicyExtractor(
        degree=degree,
        M_c=M_c, M_p=M_p, l=l, dt=dt,
        C=C, rho=rho
    )
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
            # P_z, P_z_next, P_y, L_xu_scaled, N, M_offline, gamma,
            regularization=regularization, fixed_mu=fixed_mu
        )
    else:
        raise ValueError("Scaling method not supported")
        # print(f"eigenvalues of Q_mm: {np.linalg.eigvals(Q_mm)}")
    
    status_m2 = status_info["stage2_status"]
    status_m1 = status_info["stage1_status"]

    # Extract policy using the analytical method
    if Q_mm is not None:
        
        if poly_scaler is not None:
            # Use robust polynomial scaler for policy extraction
            # For robust polynomial scalers, we need to handle the scaling differently
            mm_policy = extractor.extract_moment_matching_policy_analytical(
                system, Q_mm, poly_scaler  # Pass the poly_scaler
            )
            # mm_policy = extractor.extract_policy_grid_search(
            #     system, Q_mm, poly_scaler=poly_scaler, n_grid_points=101
            # )
        else:
            raise ValueError("Scaling method not supported")

        # comparison_results = extractor.compare_policy_methods(system, Q_mm, poly, scaler, n_grid_points=200)
        # # Plot action differences
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 4))

        # plt.subplot(1, 2, 1)
        # plt.scatter(comparison_results['grid_actions'], comparison_results['analytical_actions'])
        # plt.plot([-10, 10], [-10, 10], 'r--', alpha=0.5)  # Perfect agreement line
        # plt.xlabel('Grid Search Actions')
        # plt.ylabel('Analytical Actions')
        # plt.title('Action Comparison')
        # plt.grid(True, alpha=0.3)

        # plt.subplot(1, 2, 2)
        # plt.scatter(comparison_results['grid_q_values'], comparison_results['analytical_q_values'])
        # plt.plot([0, 100], [0, 100], 'r--', alpha=0.5)  # Perfect agreement line
        # plt.xlabel('Grid Search Q-values')
        # plt.ylabel('Analytical Q-values')
        # plt.title('Q-value Comparison')
        # plt.grid(True, alpha=0.3)

        # plt.tight_layout()
        # plt.show()

    else:
        # Create a dummy policy that always returns 0 if moment matching failed
        print("Moment matching failed")
        mm_policy = None

    # Identity baseline
    status_id = solve_identity_Q(P_z, P_z_next, L_xu, N, gamma)

    # Generate test states (use provided test_states or generate new ones)
    if test_states is None:
        test_states, _ = system.generate_samples_auxiliary(
            (-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5),
            extractor.u_bounds, n_samples=10
        )

    # Optionally replace MM policy with a multivariate cubic surrogate in full state to stabilize degree-2 behavior
    cubic_metrics = None
    policy_for_eval = mm_policy
    # if mm_policy is not None and degree == 2:
    #     try:
    #         # Improved sampling strategy: mix uniform + focused sampling near equilibrium
    #         # More samples for better accuracy
    #         n_fit = 5000  # Increased from 2000
    #         n_fit_uniform = int(0.7 * n_fit)  # 70% uniform
    #         n_fit_focused = n_fit - n_fit_uniform  # 30% focused near equilibrium
            
    #         rng = np.random.default_rng(seed + 12345)
    #         X_fit = np.empty((n_fit, 4), dtype=float)
            
    #         # Uniform sampling (broader coverage)
    #         X_fit[:n_fit_uniform, 0] = rng.uniform(x_bounds[0], x_bounds[1], size=n_fit_uniform)
    #         X_fit[:n_fit_uniform, 1] = rng.uniform(x_dot_bounds[0], x_dot_bounds[1], size=n_fit_uniform)
    #         X_fit[:n_fit_uniform, 2] = rng.uniform(theta_bounds[0], theta_bounds[1], size=n_fit_uniform)
    #         X_fit[:n_fit_uniform, 3] = rng.uniform(theta_dot_bounds[0], theta_dot_bounds[1], size=n_fit_uniform)
            
    #         # Focused sampling near equilibrium (more important for control)
    #         focus_scale = 0.3  # Sample within 30% of range around equilibrium
    #         X_fit[n_fit_uniform:, 0] = rng.normal(0, (x_bounds[1] - x_bounds[0]) * focus_scale, size=n_fit_focused)
    #         X_fit[n_fit_uniform:, 1] = rng.normal(0, (x_dot_bounds[1] - x_dot_bounds[0]) * focus_scale, size=n_fit_focused)
    #         X_fit[n_fit_uniform:, 2] = rng.normal(0, (theta_bounds[1] - theta_bounds[0]) * focus_scale, size=n_fit_focused)
    #         X_fit[n_fit_uniform:, 3] = rng.normal(0, (theta_dot_bounds[1] - theta_dot_bounds[0]) * focus_scale, size=n_fit_focused)
            
    #         # Clip to bounds
    #         X_fit[:, 0] = np.clip(X_fit[:, 0], x_bounds[0], x_bounds[1])
    #         X_fit[:, 1] = np.clip(X_fit[:, 1], x_dot_bounds[0], x_dot_bounds[1])
    #         X_fit[:, 2] = np.clip(X_fit[:, 2], theta_bounds[0], theta_bounds[1])
    #         X_fit[:, 3] = np.clip(X_fit[:, 3], theta_dot_bounds[0], theta_dot_bounds[1])

    #         # Targets from MM policy
    #         u_fit = np.array([float(mm_policy(X_fit[i])) for i in range(n_fit)], dtype=float)

    #         # Optionally scale states before polynomial expansion if scaler is available
    #         X_basis = X_fit
    #         if poly_scaler is not None and hasattr(poly_scaler, 'scaler_x') and poly_scaler.scaler_x is not None:
    #             X_basis = poly_scaler.scaler_x.transform(X_fit)

    #         # Multivariate cubic features
    #         poly3 = PolynomialFeatures(3, include_bias=True)
    #         Phi = poly3.fit_transform(X_basis)

    #         # Improved regularization: cross-validate to find optimal lambda
    #         # Try multiple regularization values and pick best
    #         lambda_candidates = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    #         best_lam = 1e-6
    #         best_r2 = -np.inf
            
    #         # Use a subset for quick validation
    #         n_val_internal = min(500, n_fit // 4)
    #         val_indices = rng.choice(n_fit, size=n_val_internal, replace=False)
    #         train_indices = np.setdiff1d(np.arange(n_fit), val_indices)
            
    #         Phi_train = Phi[train_indices]
    #         u_train = u_fit[train_indices]
    #         Phi_val_internal = Phi[val_indices]
    #         u_val_internal = u_fit[val_indices]
            
    #         for lam in lambda_candidates:
    #             try:
    #                 A = Phi_train.T @ Phi_train + lam * np.eye(Phi.shape[1])
    #                 b = Phi_train.T @ u_train
    #                 coeffs_cv = np.linalg.solve(A, b)
                    
    #                 u_pred_cv = Phi_val_internal @ coeffs_cv
    #                 resid_cv = u_val_internal - u_pred_cv
    #                 ss_res_cv = np.sum(resid_cv**2)
    #                 ss_tot_cv = np.sum((u_val_internal - np.mean(u_val_internal))**2)
    #                 r2_cv = 1.0 - ss_res_cv / ss_tot_cv if ss_tot_cv > 0 else -np.inf
                    
    #                 if r2_cv > best_r2:
    #                     best_r2 = r2_cv
    #                     best_lam = lam
    #             except:
    #                 continue
            
    #         # Fit final model with best lambda on full training set
    #         lam = best_lam
    #         A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    #         b = Phi.T @ u_fit
    #         coeffs = np.linalg.solve(A, b)

    #         # Define surrogate policy over full state
    #         def cubic_policy_full(state):
    #             s = np.atleast_2d(state.astype(float)) if isinstance(state, np.ndarray) else np.atleast_2d(np.array(state, dtype=float))
    #             S = s
    #             if poly_scaler is not None and hasattr(poly_scaler, 'scaler_x') and poly_scaler.scaler_x is not None:
    #                 S = poly_scaler.scaler_x.transform(S)
    #             phi = poly3.transform(S)
    #             u = phi @ coeffs
    #             u = float(u.ravel()[0]) if u.size == 1 else u.ravel()
    #             return float(np.clip(u, u_bounds[0], u_bounds[1]))

    #         # Evaluate fit on a validation set
    #         n_val = 500
    #         X_val = np.empty((n_val, 4), dtype=float)
    #         X_val[:, 0] = rng.uniform(x_bounds[0], x_bounds[1], size=n_val)
    #         X_val[:, 1] = rng.uniform(x_dot_bounds[0], x_dot_bounds[1], size=n_val)
    #         X_val[:, 2] = rng.uniform(theta_bounds[0], theta_bounds[1], size=n_val)
    #         X_val[:, 3] = rng.uniform(theta_dot_bounds[0], theta_dot_bounds[1], size=n_val)
    #         u_true = np.array([float(mm_policy(X_val[i])) for i in range(n_val)], dtype=float)
    #         S_val = X_val if (poly_scaler is None or getattr(poly_scaler, 'scaler_x', None) is None) else poly_scaler.scaler_x.transform(X_val)
    #         Phi_val = poly3.transform(S_val)
    #         u_pred = Phi_val @ coeffs
    #         u_pred = u_pred.reshape(-1)
    #         resid = u_true - u_pred
    #         rmse = float(np.sqrt(np.mean(resid**2)))
    #         mae = float(np.mean(np.abs(resid)))
    #         ss_res = float(np.sum(resid**2))
    #         ss_tot = float(np.sum((u_true - np.mean(u_true))**2))
    #         r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    #         cubic_metrics = {"rmse": rmse, "mae": mae, "r2": r2}
            
    #         # Additional diagnostic metrics
    #         abs_resid = np.abs(resid)
    #         u_range = u_bounds[1] - u_bounds[0]
    #         u_std = np.std(u_true)
    #         u_mean_abs = np.mean(np.abs(u_true))
            
    #         # Relative errors (normalized by action range and magnitude)
    #         rel_error_range = mae / u_range if u_range > 0 else 0  # Error as fraction of action range
    #         rel_error_magnitude = mae / (u_mean_abs + 1e-8)  # Error relative to typical action magnitude
    #         rel_error_std = mae / (u_std + 1e-8)  # Error relative to action variability
            
    #         # Percentile analysis
    #         p95 = np.percentile(abs_resid, 95)
    #         p99 = np.percentile(abs_resid, 99)
    #         max_error = np.max(abs_resid)
            
    #         # Correlation
    #         correlation = np.corrcoef(u_true, u_pred)[0, 1]
            
    #         # Print comprehensive validation statistics
    #         print(f"\n=== Cubic Polynomial Surrogate Validation ===")
    #         print(f"Number of training samples: {n_fit} ({n_fit_uniform} uniform + {n_fit_focused} focused near equilibrium)")
    #         print(f"Number of validation samples: {n_val}")
    #         print(f"Optimal regularization (lambda): {best_lam:.2e} (selected via cross-validation)")
    #         print(f"\n--- Absolute Error Metrics ---")
    #         print(f"RMSE: {rmse:.6f}")
    #         print(f"MAE: {mae:.6f}")
    #         print(f"Max absolute error: {max_error:.6f}")
    #         print(f"95th percentile error: {p95:.6f}")
    #         print(f"99th percentile error: {p99:.6f}")
    #         print(f"\n--- Relative Error Metrics ---")
    #         print(f"MAE / Action range: {rel_error_range:.4%} (smaller is better)")
    #         print(f"MAE / Mean |action|: {rel_error_magnitude:.4%} (smaller is better)")
    #         print(f"MAE / Action std: {rel_error_std:.4%} (smaller is better)")
    #         print(f"\n--- Fit Quality Metrics ---")
    #         print(f"R²: {r2:.6f} (1.0 = perfect, >0.99 = excellent, >0.95 = good)")
    #         print(f"Correlation: {correlation:.6f} (1.0 = perfect)")
    #         print(f"\n--- Residual Statistics ---")
    #         print(f"  Mean: {np.mean(resid):.6f} (should be ~0 for unbiased fit)")
    #         print(f"  Std: {np.std(resid):.6f}")
    #         print(f"  Min: {np.min(resid):.6f}")
    #         print(f"  Max: {np.max(resid):.6f}")
    #         print(f"\n--- Action Statistics (for reference) ---")
    #         print(f"  Action range: [{u_bounds[0]:.2f}, {u_bounds[1]:.2f}]")
    #         print(f"  Mean |action|: {u_mean_abs:.6f}")
    #         print(f"  Action std: {u_std:.6f}")
    #         print(f"  Action range size: {u_range:.2f}")
    #         print(f"\n--- Interpretation ---")
    #         if r2 > 0.99 and rel_error_range < 0.01:
    #             print(f"✓ Excellent approximation: R² > 0.99, error < 1% of action range")
    #         elif r2 > 0.95 and rel_error_range < 0.05:
    #             print(f"✓ Good approximation: R² > 0.95, error < 5% of action range")
    #         elif r2 > 0.90:
    #             print(f"⚠ Moderate approximation: R² > 0.90, may have some inaccuracies")
    #         else:
    #             print(f"✗ Poor approximation: R² < 0.90, consider using original MM policy")
    #         print(f"=============================================\n")

    #         policy_for_eval = cubic_policy_full
    #     except Exception as e:
    #         print("Multivariate cubic fit failed, using original MM policy:", e)
    #         policy_for_eval = mm_policy

    if policy_for_eval is not None:
        results_policy = extractor.compare_policies(
            system, lqr_policy, policy_for_eval, test_states, N, horizon=10000
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
        # Cubic surrogate metrics (if used)
        "cubic_rmse": cubic_metrics["rmse"] if cubic_metrics is not None else None,
        "cubic_mae": cubic_metrics["mae"] if cubic_metrics is not None else None,
        "cubic_r2": cubic_metrics["r2"] if cubic_metrics is not None else None,
        # Value function positivity check results
        # "value_function_all_positive": positivity_results["all_positive"],
        # "value_function_positive_ratio": positivity_results["positive_ratio"],
        # "value_function_min_value": positivity_results["min_value"],
        # "value_function_max_value": positivity_results["max_value"],
        # "value_function_mean_value": positivity_results["mean_value"],
        # "value_function_n_samples": positivity_results["n_samples"],
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
        extractor = PolicyExtractor(
            degree=degree,
            M_c=M_c, M_p=M_p, l=l, dt=dt,
            C=C, rho=rho
        )
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
            (0.0, 0.0), (0.0, 0.0), (-1.0, 1.0), (0.0, 0.0),
            extractor.u_bounds, n_samples=5
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
        
        # Use robust polynomial feature scaler for degree 2+, standard for degree 1
        if use_scaling:
            # Use robust polynomial feature scaler with separate scaling for states and actions
            # Determine dimensions from the data
            dx = x_pool.shape[1]  # State dimensions
            du = u_pool.shape[1]   # Action dimensions
            
            # For degree 1, use standard scaling (with centering) to ensure position is centered around 0
            # For degree 2+, use robust scaling for better numerical stability
            scaling_method = 'standard' # if degree == 1 else 'robust'
            
            poly_scaler = PolynomialFeatureScaler(
                degree=degree, 
                scaling_method=scaling_method,
                dx=dx,
                du=du
            )
            P_pool = poly_scaler.fit_transform(Z_pool)
            print(f"  Fitted {scaling_method} polynomial scaler (degree={degree}, dx={dx}, du={du}) on {len(Z_pool)} samples for seed {seed}")
            
        else:
            poly_scaler = PolynomialFeatureScaler(
                degree=degree, 
                scaling_method='none'
            )
            P_pool = poly_scaler.fit_transform(Z_pool)
            print(f"  Fitted polynomial scaler (degree={degree}) on {len(Z_pool)} samples for seed {seed}")
        
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
                    data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler,
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
                        data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler
                    ))
            else:
                # For subsequent N values, use the fixed mu
                if mu_fixed is not None and fix_mu_from_first_n:
                    print(f"Using fixed mu for N={N}, seed={seed}")
                    results.append(run_one(
                        seed=seed, N=int(N), M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler,
                        fixed_mu=mu_fixed
                    ))
                else:
                    # Fallback to normal computation
                    results.append(run_one(
                        seed=seed, N=int(N), M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler
                    ))
    
    return results

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Percentage boundedness of LPs vs N.")
    parser.add_argument("--seeds", type=int, default=2, help="Number of seeds per N (default: 10)")
    parser.add_argument("--Nmin", type=int, default=500, help="Min N (default: 100)")
    parser.add_argument("--Nmax", type=int, default=5000, help="Max N (default: 2000)")
    parser.add_argument("--Nstep", type=int, default=250, help="Step for N (default: 100)")
    parser.add_argument("--out_json", type=str, default="../results/test_analytical_noscaling_deg1_1e-1_mu_sum_1000M.json", help="Where to save raw results JSON")
    parser.add_argument("--plot_png", type=str, default="../figures/test_analytical_noscaling_deg1_1e-1_mu_sum_1000M.pdf", help="Where to save the plot")
    parser.add_argument("--plot_policy", type=str, default="../figures/test_policy_analytical_noscaling_deg1_1e-2_mu_sum_1000M.pdf", help="Where to save the policy comparison plot")
    parser.add_argument("--M_offline", type=int, default=1000, help="Offline pool size (default: 500)")
    parser.add_argument("--use_scaling", action="store_true", default=False, help="Use feature scaling (default: False)")
    parser.add_argument("--regularization", type=float, default=1e-4, help="Regularization parameter (default: 1e-4)")
    parser.add_argument("--fix_mu", action="store_true", default=False, help="Fix mu from first N and reuse for all subsequent N values")
    
    args = parser.parse_args()
    N_values = list(range(args.Nmin, args.Nmax + 1, args.Nstep))
    seeds = list(range(args.seeds))
    # seeds = [s+6 for s in range(args.seeds)]

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
