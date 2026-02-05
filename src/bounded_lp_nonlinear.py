import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from polynomial_features import FilteredPolynomialFeatures, StateOnlyPolynomialFeatures
from dynamical_systems_polished import cart_pole_v2 as cart_pole
from dynamical_systems_polished import point_mass_cubic_drag
from dynamical_systems_polished import point_mass_2d_cubic_drag
from policy_extractor_polished import PolicyExtractor
from moment_matching_v2 import solve_identity_Q
from moment_matching_v2 import solve_moment_matching_Q
from config import (
    GAMMA, M_C, M_P, L, DT, C_CART_POLE, RHO_CART_POLE, DEGREE,
    X_BOUNDS, X_DOT_BOUNDS, THETA_BOUNDS, THETA_DOT_BOUNDS, U_BOUNDS,
    X_BOUNDS_GRID, X_DOT_BOUNDS_GRID, THETA_BOUNDS_GRID, THETA_DOT_BOUNDS_GRID,
    DEFAULT_M_OFFLINE, DEFAULT_REGULARIZATION,
    M_POINT_MASS_2D, K_POINT_MASS_2D, C_POINT_MASS_2D, DT_POINT_MASS_2D,
    RHO_POINT_MASS_2D,
    C_P_POINT_MASS, C_V_POINT_MASS,
    P_BOUNDS_POINT_MASS, V_BOUNDS_POINT_MASS, U_BOUNDS_POINT_MASS,
    P_BOUNDS_POINT_MASS_TEST, V_BOUNDS_POINT_MASS_TEST
)

# -------------------------
# Config - Using centralized configuration
# -------------------------
gamma = GAMMA  # Using centralized discount factor
degree = DEGREE

def ensure_parent_dir(path_str):
    """
    Create parent directories for an output file if they do not exist.
    """
    if not path_str:
        return
    parent = Path(path_str).expanduser().parent
    if str(parent) in ("", "."):
        return
    parent.mkdir(parents=True, exist_ok=True)


def build_descriptive_out_path(base_path, system_name, M_offline, use_scaling):
    """
    Append experiment descriptors to the output filename so that saved results
    are self-describing.
    """
    path = Path(base_path)
    suffix_parts = [
        f"sys-{system_name}",
        f"M-{M_offline}",
        "scaling" if use_scaling else "noscaling",
    ]
    new_name = f"{path.stem}_{'_'.join(suffix_parts)}{path.suffix}"
    return path.with_name(new_name)

def get_system_config(system_name: str, dx=None):
    """
    Build a configuration dictionary describing how to instantiate and sample from
    a supported dynamical system.
    
    Args:
        system_name: Name of the system ("cartpole" or "pointmass2d")
        dx: State dimension (for pointmass2d: dx = 2n where n is position/velocity dim, default n=2)
    """
    system_name = (system_name or "cartpole").lower()

    if system_name == "cartpole":
        def build_system(N, M_offline):
            return cart_pole(M_C, M_P, L, DT, C_CART_POLE, RHO_CART_POLE, GAMMA, N, M_offline)

        def sample_fn(system, n_samples):
            return system.generate_samples(
                X_BOUNDS, X_DOT_BOUNDS, THETA_BOUNDS, THETA_DOT_BOUNDS, U_BOUNDS,
                n_samples=n_samples
            )

        def aux_sample_fn(system, n_samples):
            return system.generate_samples_auxiliary(
                X_BOUNDS, X_DOT_BOUNDS, THETA_BOUNDS, THETA_DOT_BOUNDS, U_BOUNDS, n_samples=n_samples
            )

        def test_sampler(system, u_bounds_local, n_samples):
            # Start samples for testing policies performance
            return system.generate_samples_auxiliary(
                (0.0, 0.0), (0.0, 0.0), (-0.5, 0.5), (-0.0, 0.0),
                u_bounds_local, n_samples=n_samples
            )

        return {
            "name": "cartpole",
            "pretty_name": "Cart-Pole",
            "build_system": build_system,
            "sample_fn": sample_fn,
            "aux_sample_fn": aux_sample_fn,
            "test_sampler": test_sampler,
            "extractor_kwargs": {
                "M_c": M_C,
                "M_p": M_P,
                "l": L,
                "dt": DT,
                "C": C_CART_POLE,
                "rho": RHO_CART_POLE,
                "gamma": GAMMA,
                "u_bounds": U_BOUNDS,
                "system_type": "cartpole",
            },
            "u_bounds": U_BOUNDS,
            "supports_policy_heatmaps": True,
            "supports_lqr_rollouts": True,
            "lqr_rollout_length": 500,
        }

    if system_name in ("pointmass2d", "pointmass", "point_mass_2d", "mass2d"):
        # Use parameters from config.py
        # For point mass: state is [p, v] where p,v are n-dimensional
        # So dx = 2n, meaning n = dx / 2
        # Default to n=2 (2D point mass) if dx is None
        if dx is None:
            n = 2  # Default to 2D
        else:
            if dx % 2 != 0:
                raise ValueError(f"dx must be even for point mass system (dx={dx}). State is [p, v] where p and v are n-dimensional, so dx = 2n.")
            n = dx // 2
        
        # Control dimension matches position dimension (fully actuated)
        m_u = n
        
        # Build cost matrix C using config variables C_P_POINT_MASS and C_V_POINT_MASS
        # C = diag(sqrt([C_p, ..., C_p, C_v, ..., C_v])) where first n entries are C_p, last n entries are C_v
        C_cost = np.diag(np.sqrt(np.concatenate([
            np.full(n, C_P_POINT_MASS),  # Position weights
            np.full(n, C_V_POINT_MASS)   # Velocity weights
        ])))
        
        def build_system(N, M_offline):
            return point_mass_cubic_drag(
                n=n,  # Position/velocity dimension (customizable via dx)
                m_u=m_u,  # Control dimension (matches n for fully actuated)
                B=None,  # Use identity (fully actuated)
                mass=M_POINT_MASS_2D, 
                k=K_POINT_MASS_2D, 
                c=C_POINT_MASS_2D, 
                delta_t=DT_POINT_MASS_2D, 
                C=C_cost, 
                rho=RHO_POINT_MASS_2D, 
                gamma=GAMMA, 
                N=N, 
                M=M_offline
            )

        def sample_fn(system, n_samples):
            return system.generate_samples(
                P_BOUNDS_POINT_MASS, 
                V_BOUNDS_POINT_MASS, 
                U_BOUNDS_POINT_MASS, 
                n_samples=n_samples
            )

        def aux_sample_fn(system, n_samples):
            return system.generate_samples_auxiliary(
                P_BOUNDS_POINT_MASS, 
                V_BOUNDS_POINT_MASS, 
                U_BOUNDS_POINT_MASS, 
                n_samples=n_samples
            )

        def test_sampler(system, u_bounds_local, n_samples):
            # Start near the origin but not exactly at equilibrium (using test bounds from config)
            return system.generate_samples_auxiliary(
                P_BOUNDS_POINT_MASS_TEST, 
                V_BOUNDS_POINT_MASS_TEST, 
                u_bounds_local, 
                n_samples=n_samples
            )

        # Determine pretty name based on dimension
        if n == 2:
            pretty_name = "2D Point-Mass (cubic drag)"
        else:
            pretty_name = f"{n}D Point-Mass (cubic drag)"
        
        return {
            "name": "pointmass2d",
            "pretty_name": pretty_name,
            "build_system": build_system,
            "sample_fn": sample_fn,
            "aux_sample_fn": aux_sample_fn,
            "test_sampler": test_sampler,
            "extractor_kwargs": {
                "dt": DT_POINT_MASS_2D,
                "C": C_cost,  # Use dynamically created cost matrix
                "rho": RHO_POINT_MASS_2D,
                "gamma": GAMMA,
                "u_bounds": U_BOUNDS_POINT_MASS,
                "system_type": "pointmass2d",
            },
            "u_bounds": U_BOUNDS_POINT_MASS,
            "supports_policy_heatmaps": False,
            "supports_lqr_rollouts": True,
            "lqr_rollout_length": 500,
        }


    if system_name in ("pointmass2d_fixed", "pointmass_2d", "point_mass_2d"):
        # Fixed 2D point mass with cubic drag (hardcoded dimensions: dx=4, du=2)
        # Uses config parameters for physical properties
        
        # Build cost matrix C using config variables
        C_cost = np.diag(np.sqrt([C_P_POINT_MASS, C_P_POINT_MASS, C_V_POINT_MASS, C_V_POINT_MASS]))
        
        def build_system(N, M_offline):
            return point_mass_2d_cubic_drag(
                m=M_POINT_MASS_2D, 
                k=K_POINT_MASS_2D, 
                c=C_POINT_MASS_2D, 
                delta_t=DT_POINT_MASS_2D, 
                C=C_cost, 
                rho=RHO_POINT_MASS_2D, 
                gamma=GAMMA, 
                N=N, 
                M=M_offline
            )

        def sample_fn(system, n_samples):
            return system.generate_samples(
                P_BOUNDS_POINT_MASS, 
                V_BOUNDS_POINT_MASS, 
                U_BOUNDS_POINT_MASS, 
                n_samples=n_samples
            )

        def aux_sample_fn(system, n_samples):
            return system.generate_samples_auxiliary(
                P_BOUNDS_POINT_MASS, 
                V_BOUNDS_POINT_MASS, 
                U_BOUNDS_POINT_MASS, 
                n_samples=n_samples
            )

        def test_sampler(system, u_bounds_local, n_samples):
            # Start near the origin but not exactly at equilibrium
            return system.generate_samples_auxiliary(
                P_BOUNDS_POINT_MASS_TEST, 
                V_BOUNDS_POINT_MASS_TEST, 
                u_bounds_local, 
                n_samples=n_samples
            )

        return {
            "name": "pointmass2d_fixed",
            "pretty_name": "2D Point-Mass (cubic drag, fixed)",
            "build_system": build_system,
            "sample_fn": sample_fn,
            "aux_sample_fn": aux_sample_fn,
            "test_sampler": test_sampler,
            "extractor_kwargs": {
                "dt": DT_POINT_MASS_2D,
                "C": C_cost,
                "rho": RHO_POINT_MASS_2D,
                "gamma": GAMMA,
                "u_bounds": U_BOUNDS_POINT_MASS,
                "system_type": "pointmass2d",
            },
            "u_bounds": U_BOUNDS_POINT_MASS,
            "supports_policy_heatmaps": False,
            "supports_lqr_rollouts": True,
            "lqr_rollout_length": 500,
        }

    raise ValueError(f"Unsupported system '{system_name}'. Available options: cartpole, pointmass, pointmass2d_fixed.")


def run_one(seed, N, M_offline=DEFAULT_M_OFFLINE, use_scaling=True, regularization=DEFAULT_REGULARIZATION, 
                    data_pools=None, test_states=None, analyze_divergence=False, 
                    divergence_analysis_path=None, divergence_horizon=10000, system_cfg=None):
    """Generate data with given seed and N, then solve both LPs and return boundedness flags."""
    if system_cfg is None:
        raise ValueError("system_cfg must be provided to run_one")

    # Build system and data
    extractor = PolicyExtractor(
        degree=degree,
        **system_cfg["extractor_kwargs"]
    )
    system = system_cfg["build_system"](N, M_offline)

    lqr_policy, K_lqr, P_lqr = extractor.extract_lqr_policy(system)

    # Use pre-generated data pools
    if data_pools is not None:
        L_xu_pool = data_pools.get('L_xu')
        L_xu = L_xu_pool[:N] if L_xu_pool is not None else system.cost(data_pools['x'][:N], data_pools['u'][:N])

        P_z = data_pools.get('P_z')
        P_z_next = data_pools.get('P_z_next')
        P_y = data_pools.get('P_y')
        Mi_tensor = data_pools.get('Mi_tensor')
        Py_outer = data_pools.get('Py_outer')

        # Slice polynomial features to match first N samples
        if P_z is not None:
            P_z = P_z[:N]
        if P_z_next is not None:
            P_z_next = P_z_next[:N]
        if Mi_tensor is not None:
            Mi_tensor = Mi_tensor[:N]           
    else:
        raise ValueError("Data not provided.")

    # Solve moment matching LP
    Q_mm, mu, status_info = solve_moment_matching_Q(
        P_z, P_z_next, P_y, L_xu, N, M_offline, gamma,
        regularization=regularization,
        Mi_tensor=Mi_tensor, Py_outer=Py_outer
    )
    
    status_m2 = status_info["stage2_status"]
    status_m1 = status_info["stage1_status"]
    
    
    if Q_mm is not None:
        # Extract moment matching policy
        # Get dimensions from data_pools
        dx = data_pools['x'].shape[1] if data_pools is not None else None
        du = data_pools['u'].shape[1] if data_pools is not None else None
        
        mm_policy = extractor.extract_moment_matching_policy_analytical(
            system, Q_mm, degree=degree, dx=dx, du=du
        )

    else:
        print("Moment matching failed")
        mm_policy = None

    # Identity baseline
    identity_result = solve_identity_Q(P_z, P_z_next, L_xu, N, gamma)
    if isinstance(identity_result, tuple):
        status_id, Q_id = identity_result
    else:
        # Backward compatibility: if only status is returned
        status_id = identity_result
        Q_id = None
    
    # Extract identity policy if Q_id is available
    id_policy = None
    if Q_id is not None:
        try:
            # Get dimensions from data_pools
            dx = data_pools['x'].shape[1] if data_pools is not None else None
            du = data_pools['u'].shape[1] if data_pools is not None else None
            
            id_policy = extractor.extract_moment_matching_policy_analytical(
                system, Q_id, degree=degree, dx=dx, du=du
            )
        except Exception as e:
            print(f"Failed to extract identity policy: {e}")
            id_policy = None
    
    # Run policy comparison if moment matching succeeded
    if Q_mm is not None:
        # Prepare divergence analysis path if requested but not provided
        if analyze_divergence and divergence_analysis_path is None:
            divergence_analysis_path = f"divergence_analysis_N_{N}_seed_{seed}.png"
        
        results_policy = extractor.compare_policies(
            system, lqr_policy, mm_policy, test_states, N, horizon=10000,
            analyze_divergence=analyze_divergence,
            divergence_threshold=1.0,
            divergence_save_path=divergence_analysis_path,
            extra_policy=id_policy,
            extra_label="id"
        )
        
        # Print divergence analysis summary if available
        if analyze_divergence and "divergence_analysis" in results_policy:
            divergence_results = results_policy["divergence_analysis"]
            print(f"\nDivergence Analysis Summary:")
            print(f"  Found {divergence_results['n_divergence_cases']} divergence cases")
            if divergence_results['summary']:
                summary = divergence_results['summary']
                print(f"  Mean divergence step: {summary.get('mean_divergence_step', 'N/A'):.1f}")
                print(f"  Mean action difference at divergence: {summary.get('mean_action_diff_at_divergence', 'N/A'):.4f}")
    else:
        print("Moment matching failed")
        results_policy = {
            "lqr_success": 0.0,
            "mm_success": 0.0,
            "lqr_costs": np.inf,
            "mm_costs": np.inf,
            "id_success": 0.0 if id_policy is None else 0.0,
            "id_costs": np.inf,
            "id_costs_std": 0.0,
            "id_costs_median": 0.0
        }
        # Still try to evaluate identity policy even if MM failed
        if id_policy is not None:
            try:
                results_id_policy = extractor.compare_policies(
                    system, lqr_policy, id_policy, test_states, N, horizon=5000,
                    analyze_divergence=False,
                    divergence_threshold=0.5,
                    divergence_save_path=None
                )
                results_policy["id_success"] = results_id_policy["mm_success"]
                results_policy["id_costs"] = results_id_policy["mm_costs"]
                results_policy["id_costs_std"] = results_id_policy.get("mm_costs_std", 0.0)
                results_policy["id_costs_median"] = results_id_policy.get("mm_costs_median", 0.0)
            except Exception as e:
                print(f"Failed to evaluate identity policy: {e}")

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
    print("ID success: ", results_policy.get("id_success", 0.0))
    print("LQR costs: ", results_policy["lqr_costs"])
    print("MM costs: ", results_policy["mm_costs"])
    print("ID costs: ", results_policy.get("id_costs", np.inf))

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
        # Identity policy metrics
        "id_success": results_policy.get("id_success", 0.0),
        "id_costs": results_policy.get("id_costs", np.inf),
        "id_costs_std": results_policy.get("id_costs_std", 0.0),
        "id_costs_median": results_policy.get("id_costs_median", 0.0),
        # Costs when each method individually converged
        "lqr_costs_all_converged": results_policy.get("lqr_costs_all_converged", np.inf),
        "mm_costs_all_converged": results_policy.get("mm_costs_all_converged", np.inf),
        "id_costs_all_converged": results_policy.get("id_costs_all_converged", np.inf),
        "lqr_costs_all_converged_std": results_policy.get("lqr_costs_all_converged_std", 0.0),
        "mm_costs_all_converged_std": results_policy.get("mm_costs_all_converged_std", 0.0),
        "id_costs_all_converged_std": results_policy.get("id_costs_all_converged_std", 0.0),
        "lqr_costs_all_converged_median": results_policy.get("lqr_costs_all_converged_median", 0.0),
        "mm_costs_all_converged_median": results_policy.get("mm_costs_all_converged_median", 0.0),
        "id_costs_all_converged_median": results_policy.get("id_costs_all_converged_median", 0.0),
        # Total costs when each method individually converged
        "lqr_costs_all_converged_total": results_policy.get("lqr_costs_all_converged_total", 0.0),
        "mm_costs_all_converged_total": results_policy.get("mm_costs_all_converged_total", 0.0),
        "id_costs_all_converged_total": results_policy.get("id_costs_all_converged_total", 0.0),
        # Statistics when ALL THREE methods converge
        "n_all_convergent": results_policy.get("n_all_convergent", 0),
        "all_convergent_success_rate": results_policy.get("all_convergent_success_rate", 0.0),
        "all_convergent_lqr_costs": results_policy.get("all_convergent_lqr_costs", np.inf),
        "all_convergent_mm_costs": results_policy.get("all_convergent_mm_costs", np.inf),
        "all_convergent_id_costs": results_policy.get("all_convergent_id_costs", np.inf),
        "all_convergent_lqr_costs_std": results_policy.get("all_convergent_lqr_costs_std", 0.0),
        "all_convergent_mm_costs_std": results_policy.get("all_convergent_mm_costs_std", 0.0),
        "all_convergent_id_costs_std": results_policy.get("all_convergent_id_costs_std", 0.0),
        # Total costs when all three methods converge
        "all_convergent_lqr_costs_total": results_policy.get("all_convergent_lqr_costs_total", 0.0),
        "all_convergent_mm_costs_total": results_policy.get("all_convergent_mm_costs_total", 0.0),
        "all_convergent_id_costs_total": results_policy.get("all_convergent_id_costs_total", 0.0),
    }
    
    return result

def sweep(N_values=range(50, 501, 50), seeds=range(10), M_offline=DEFAULT_M_OFFLINE, 
                  use_scaling=True, regularization=DEFAULT_REGULARIZATION, 
                  analyze_divergence=False, divergence_dir=None, divergence_horizon=10000,
                  system_cfg=None):
    if system_cfg is None:
        raise ValueError("system_cfg must be provided to sweep()")

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
            **system_cfg["extractor_kwargs"]
        )
        system_temp = system_cfg["build_system"](N_max, M_offline)
        
        # Samples are generated randomly (uniformly distributed) across the bounds,
        # Optionally, half samples come from LQR rollouts near equilibrium for better MM performance.
        x_pool, u_pool, x_plus_pool, u_plus_pool = system_cfg["sample_fn"](
            system_temp, N_max
        )
        
        # Generate auxiliary data (fixed for all N)
        x_aux, u_aux = system_cfg["aux_sample_fn"](system_temp, M_offline)

        # Precompute costs once for the entire pool
        L_xu_pool = system_temp.cost(x_pool, u_pool)
        
        # Generate test states (fixed for all N)
        test_states, _ = system_cfg["test_sampler"](system_temp, extractor.u_bounds, n_samples=10)
        
        # Fit polynomial features ONCE on the full dataset
        z_pool = np.concatenate([x_pool, u_pool], axis=1)
        z_plus_pool = np.concatenate([x_plus_pool, u_plus_pool], axis=1)
        Z_pool = np.concatenate([z_pool, z_plus_pool], axis=0)
        
        # Use FilteredPolynomialFeatures directly (no scaling)
        dx = x_pool.shape[1]
        du = u_pool.shape[1]
        poly = StateOnlyPolynomialFeatures(
            degree=degree,
            include_bias=False,
            dx=dx,
            du=du
        )
        P_pool = poly.fit_transform(Z_pool)
        print(f"  Fitted polynomial features (degree={degree}, u^2 terms excluded) on {len(Z_pool)} samples for seed {seed}")

        # Split polynomial features once and cache
        # P_pool = [z_pool features; z_plus_pool features] where z_pool = [x_pool, u_pool]
        P_z_pool = P_pool[:N_max]  # Features for (x, u) pairs: shape (N_max, d_features)
        P_z_next_pool = P_pool[N_max:]  # Features for (x_plus, u_plus) pairs: shape (N_max, d_features)
        P_y_pool = poly.transform(np.concatenate([x_aux, u_aux], axis=1))

        # Precompute tensors for LPs
        # Mi_tensor_pool[i] = P_z_pool[i] @ P_z_pool[i].T - gamma * P_z_next_pool[i] @ P_z_next_pool[i].T
        Mi_tensor_pool = np.einsum('bi,bj->bij', P_z_pool, P_z_pool, optimize=True) - \
                         gamma * np.einsum('bi,bj->bij', P_z_next_pool, P_z_next_pool, optimize=True)
        # Py_outer is computed on full auxiliary pool (fixed for all N values)
        Py_outer_pool = np.einsum('bi,bj->bij', P_y_pool, P_y_pool, optimize=True)

        # Create data pools dictionary with cached preprocessing
        data_pools = {
            'x': x_pool,
            'u': u_pool,
            'x_plus': x_plus_pool,
            'u_plus': u_plus_pool,
            'x_aux': x_aux,
            'u_aux': u_aux,
            'L_xu': L_xu_pool,
            'P_z': P_z_pool,
            'P_z_next': P_z_next_pool,
            'P_y': P_y_pool,
            'Mi_tensor': Mi_tensor_pool,
            'Py_outer': Py_outer_pool
        }
        
        print(f"  Generated data pools: {len(x_pool)} training samples, {len(x_aux)} auxiliary samples, {len(test_states)} test states")

        # Run experiments for each N using nested datasets
        for i, N in enumerate(N_values):
            divergence_path = None
            if analyze_divergence and divergence_dir:
                os.makedirs(divergence_dir, exist_ok=True)
                divergence_path = os.path.join(divergence_dir, f"divergence_analysis_N_{N}_seed_{seed}.png")
            results.append(run_one(
                seed=seed, N=int(N), M_offline=M_offline, 
                use_scaling=use_scaling, regularization=regularization,
                data_pools=data_pools, test_states=test_states,
                analyze_divergence=analyze_divergence, divergence_analysis_path=divergence_path,
                divergence_horizon=divergence_horizon, system_cfg=system_cfg
            ))

    return results

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Percentage boundedness of LPs vs N.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per N (default: 10)")
    parser.add_argument("--Nmin", type=int, default=500, help="Min N (default: 100)")
    parser.add_argument("--Nmax", type=int, default=5000, help="Max N (default: 2000)")
    parser.add_argument("--Nstep", type=int, default=250, help="Step for N (default: 100)")
    parser.add_argument("--out_json", type=str, default="../results/test_cartpole_complete.json", help="Where to save raw results JSON")
    parser.add_argument("--plot_png", type=str, default="../figures/test_cartpole.pdf", help="Where to save the plot")
    parser.add_argument("--plot_policy", type=str, default="../figures/test_cartpole.pdf", help="Where to save the policy comparison plot")
    parser.add_argument("--M_offline", type=int, default=1000, help="Offline pool size (default: 500)")
    parser.add_argument("--use_scaling", action="store_true", default=False, help="Use feature scaling (default: False)")
    parser.add_argument("--regularization", type=float, default=1e-4, help="Regularization parameter (default: 1e-4)")
    parser.add_argument("--analyze_divergence", action="store_true", default=False, help="Analyze where LQR and MM policies diverge (default: False)")
    parser.add_argument("--divergence_dir", type=str, default="divergence_analysis", help="Directory to save divergence analysis plots (default: divergence_analysis)")
    parser.add_argument("--divergence_horizon", type=int, default=10000, help="Simulation horizon for divergence analysis (default: 10000, use smaller value like 1000 for faster analysis)")
    parser.add_argument("--system", type=str, default="cartpole",
                        choices=["cartpole", "pointmass", "pointmass2d_fixed"],
                        help="Dynamics to study (default: cartpole). pointmass2d_fixed uses the hardcoded 2D version.")
    parser.add_argument("--dx", type=int, default=4,
                        help="State dimension. For pointmass2d: dx = 2n where n is position/velocity dimension. Examples: dx=4 (2D), dx=6 (3D), dx=8 (4D). Default: 4 (2D point mass)")
    args = parser.parse_args()
    N_values = list(range(args.Nmin, args.Nmax + 1, args.Nstep))
    # seeds = list(range(args.seeds))
    seeds = [s+2026 for s in range(args.seeds)]
    
    system_cfg = get_system_config(args.system, dx=args.dx)

    try:
        results = sweep(
            N_values=N_values, seeds=seeds, M_offline=args.M_offline,
            use_scaling=args.use_scaling, regularization=args.regularization,
            analyze_divergence=args.analyze_divergence, divergence_dir=args.divergence_dir,
            divergence_horizon=args.divergence_horizon,
            system_cfg=system_cfg
        )
    except Exception as e:
        # Print error and traceback
        import sys, traceback
        print("ERROR while running sweep:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

    # Aggregate
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
        # Convergence rates
        lqr_success=("lqr_success", "mean"),
        mm_success=("mm_success", "mean"),
        id_success=("id_success", "mean"),
        # Costs when LQR and MM both converge
        lqr_costs_both=("lqr_costs", mean_no_inf),
        mm_costs_both=("mm_costs", mean_no_inf),
        # Average costs when each policy individually converges
        lqr_costs_all_converged=("lqr_costs_all_converged", mean_no_inf),
        mm_costs_all_converged=("mm_costs_all_converged", mean_no_inf),
        id_costs_all_converged=("id_costs_all_converged", mean_no_inf),
        # Average costs when all three converge
        all_convergent_lqr_costs=("all_convergent_lqr_costs", mean_no_inf),
        all_convergent_mm_costs=("all_convergent_mm_costs", mean_no_inf),
        all_convergent_id_costs=("all_convergent_id_costs", mean_no_inf),
    ).reset_index()

    # Convert to percentages
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_pct"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_pct"]
    agg["lqr_success"] = 100 * agg["lqr_success"]
    agg["mm_success"] = 100 * agg["mm_success"]
    agg["id_success"] = 100 * agg["id_success"]

    # Save raw + aggregated with descriptive filename
    descriptive_out_path = build_descriptive_out_path(
        args.out_json,
        args.system,
        args.M_offline,
        args.use_scaling,
    )
    ensure_parent_dir(descriptive_out_path)
    df.to_json(descriptive_out_path, orient="records", indent=2)
    agg.to_json("bounded_lp_percentages.json", orient="records", indent=2)

    # Plot
    import matplotlib.pyplot as plt

    # Plot 1: Convergence rates
    plt.figure(figsize=(8, 6))
    plt.plot(agg["N"], agg["lqr_success"], marker="o", label="LQR", linewidth=2, markersize=6, color="green")
    plt.plot(agg["N"], agg["mm_success"], marker="s", label="Moment Matching", linewidth=2, markersize=6, color="orange")
    plt.plot(agg["N"], agg["id_success"], marker="^", label="Identity", linewidth=2, markersize=6, color="purple")
    plt.xlabel("N (number of samples)", fontsize=12)
    plt.ylabel("Convergence Rate (%)", fontsize=12)
    plt.title("Policy Convergence Rates vs Sample Size", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    convergence_plot_path = args.plot_png.replace(".pdf", "_convergence.pdf").replace(".png", "_convergence.png")
    ensure_parent_dir(convergence_plot_path)
    plt.savefig(convergence_plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved convergence rates plot: {convergence_plot_path}")

    # Plot 2: Average costs when each policy individually converges
    plt.figure(figsize=(8, 6))
    plt.plot(agg["N"], agg["lqr_costs_all_converged"], 
             marker="o", label="LQR", linewidth=2, markersize=6, color="green", alpha=0.8)
    plt.plot(agg["N"], agg["mm_costs_all_converged"], 
             marker="s", label="Moment Matching", linewidth=2, markersize=6, color="orange", alpha=0.8)
    plt.plot(agg["N"], agg["id_costs_all_converged"], 
             marker="^", label="Identity", linewidth=2, markersize=6, color="purple", alpha=0.8)
    plt.xlabel("N (number of samples)", fontsize=12)
    plt.ylabel("Average Cost", fontsize=12)
    plt.title("Average Costs (When Each Policy Individually Converges)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    individual_costs_plot_path = args.plot_png.replace(".pdf", "_individual_costs.pdf").replace(".png", "_individual_costs.png")
    ensure_parent_dir(individual_costs_plot_path)
    plt.savefig(individual_costs_plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved individual costs plot: {individual_costs_plot_path}")

    # Plot 3: Average costs when all three methods converge
    plt.figure(figsize=(8, 6))
    plt.plot(agg["N"], agg["all_convergent_lqr_costs"], 
             marker="o", label="LQR", linewidth=2, markersize=6, color="green", alpha=0.8)
    plt.plot(agg["N"], agg["all_convergent_mm_costs"], 
             marker="s", label="Moment Matching", linewidth=2, markersize=6, color="orange", alpha=0.8)
    plt.plot(agg["N"], agg["all_convergent_id_costs"], 
             marker="^", label="Identity", linewidth=2, markersize=6, color="purple", alpha=0.8)
    plt.xlabel("N (number of samples)", fontsize=12)
    plt.ylabel("Average Cost", fontsize=12)
    plt.title("Average Costs (When All Three Methods Converge)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    all_convergent_costs_plot_path = args.plot_png.replace(".pdf", "_all_convergent_costs.pdf").replace(".png", "_all_convergent_costs.png")
    ensure_parent_dir(all_convergent_costs_plot_path)
    plt.savefig(all_convergent_costs_plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved all convergent costs plot: {all_convergent_costs_plot_path}")

    # Plot 4: Costs when LQR and MM both converge
    plt.figure(figsize=(8, 6))
    plt.plot(agg["N"], agg["lqr_costs_both"], 
             marker="o", label="LQR", linewidth=2, markersize=6, color="green", alpha=0.8)
    plt.plot(agg["N"], agg["mm_costs_both"], 
             marker="s", label="Moment Matching", linewidth=2, markersize=6, color="orange", alpha=0.8)
    plt.xlabel("N (number of samples)", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.title("Costs (When LQR and MM Both Converge)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    both_convergent_costs_plot_path = args.plot_png.replace(".pdf", "_both_convergent_costs.pdf").replace(".png", "_both_convergent_costs.png")
    ensure_parent_dir(both_convergent_costs_plot_path)
    plt.savefig(both_convergent_costs_plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved both convergent costs plot: {both_convergent_costs_plot_path}")
    
    print(f"\nSaved all plots and results: {descriptive_out_path}")
