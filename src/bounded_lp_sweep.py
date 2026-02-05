import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from polynomial_features import PolynomialFeatureScaler
from dynamical_systems import single_pendulum
from dynamical_systems import cart_pole_v2 as cart_pole
from dynamical_systems import mountain_car
from dynamical_systems import scalar_cubic
from policy_extraction import PolicyExtractor
from moment_matching import solve_identity_Q
from moment_matching import solve_moment_matching_Q
from moment_matching import extract_and_psd_project_uu, replace_uu_in_Q
from visualization import visualize_policies_2d_heatmaps
from config import (
    GAMMA, M_C, M_P, L, DT, C_CART_POLE, RHO_CART_POLE, DEGREE,
    X_BOUNDS, X_DOT_BOUNDS, THETA_BOUNDS, THETA_DOT_BOUNDS, U_BOUNDS,
    X_BOUNDS_GRID, X_DOT_BOUNDS_GRID, THETA_BOUNDS_GRID, THETA_DOT_BOUNDS_GRID,
    DEFAULT_M_OFFLINE, DEFAULT_REGULARIZATION,
    M_SINGLE_PENDULUM, L_SINGLE_PENDULUM, B_SINGLE_PENDULUM, DT_SINGLE_PENDULUM,
    C_SINGLE_PENDULUM, RHO_SINGLE_PENDULUM,
    THETA_BOUNDS_SINGLE_PENDULUM, THETA_DOT_BOUNDS_SINGLE_PENDULUM,
    U_BOUNDS_SINGLE_PENDULUM,
    THETA_BOUNDS_SINGLE_PENDULUM_TEST, THETA_DOT_BOUNDS_SINGLE_PENDULUM_TEST,
    DT_MOUNTAIN_CAR, C_MOUNTAIN_CAR, RHO_MOUNTAIN_CAR, GOAL_POSITION_MOUNTAIN_CAR,
    POSITION_BOUNDS_MOUNTAIN_CAR, VELOCITY_BOUNDS_MOUNTAIN_CAR, U_BOUNDS_MOUNTAIN_CAR,
    DT_SCALAR_CUBIC, C_SCALAR_CUBIC, RHO_SCALAR_CUBIC,
    X_BOUNDS_SCALAR_CUBIC, U_BOUNDS_SCALAR_CUBIC,
    SCALAR_CUBIC_OVERFLOW_THRESHOLD
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


def build_descriptive_out_path(base_path, system_name, M_offline, use_scaling, use_lqr_rollouts):
    """
    Append experiment descriptors to the output filename so that saved results
    are self-describing.
    """
    path = Path(base_path)
    suffix_parts = [
        f"sys-{system_name}",
        f"M-{M_offline}",
        "scaling" if use_scaling else "noscaling",
        "lqrrollouts" if use_lqr_rollouts else "nolqrrollouts",
    ]
    new_name = f"{path.stem}_{'_'.join(suffix_parts)}{path.suffix}"
    return path.with_name(new_name)

def get_system_config(system_name: str):
    """
    Build a configuration dictionary describing how to instantiate and sample from
    a supported dynamical system.
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
            },
            "u_bounds": U_BOUNDS,
            "supports_policy_heatmaps": True,
            "supports_lqr_rollouts": True,
            "lqr_rollout_length": 500,
        }

    if system_name == "single_pendulum":
        def _sample_mixed_pendulum(theta_bounds, theta_dot_bounds, u_bounds, n_samples, focus_ratio=0.3, focus_scale=0.2):
            """
            Mixed sampling: majority uniform over full bounds, a fraction focused near equilibrium.
            focus_ratio: fraction of samples drawn near (0,0)
            focus_scale: std as fraction of span for focused Gaussian samples
            """
            n_focus = int(focus_ratio * n_samples)
            n_uniform = n_samples - n_focus

            # Uniform samples
            theta_uni = np.random.uniform(*theta_bounds, size=n_uniform)
            theta_dot_uni = np.random.uniform(*theta_dot_bounds, size=n_uniform)
            u_uni = np.random.uniform(*u_bounds, size=(n_uniform, 1))

            # Focused near equilibrium
            span_theta = theta_bounds[1] - theta_bounds[0]
            span_theta_dot = theta_dot_bounds[1] - theta_dot_bounds[0]
            theta_std = max(1e-6, focus_scale * span_theta)
            theta_dot_std = max(1e-6, focus_scale * span_theta_dot)

            theta_foc = np.random.normal(0.0, theta_std, size=n_focus)
            theta_dot_foc = np.random.normal(0.0, theta_dot_std, size=n_focus)
            u_foc = np.random.uniform(*u_bounds, size=(n_focus, 1))

            # Combine and clip
            theta_all = np.concatenate([theta_uni, theta_foc])
            theta_dot_all = np.concatenate([theta_dot_uni, theta_dot_foc])
            theta_all = np.clip(theta_all, theta_bounds[0], theta_bounds[1])
            theta_dot_all = np.clip(theta_dot_all, theta_dot_bounds[0], theta_dot_bounds[1])
            u_all = np.vstack([u_uni, u_foc])

            states = np.column_stack([theta_all, theta_dot_all])
            return states, u_all

        def build_system(N, M_offline):
            return single_pendulum(
                m=M_SINGLE_PENDULUM,
                l=L_SINGLE_PENDULUM,
                b=B_SINGLE_PENDULUM,
                delta_t=DT_SINGLE_PENDULUM,
                C=C_SINGLE_PENDULUM,
                rho=RHO_SINGLE_PENDULUM,
                gamma=GAMMA,
                N=N,
                M=M_offline,
            )

        def sample_fn(system, n_samples, use_lqr_rollouts = False, rollout_length = 500):
            # Mixed sampling: uniform plus focused near equilibrium (0,0)
            states, actions = _sample_mixed_pendulum(
                THETA_BOUNDS_SINGLE_PENDULUM,
                THETA_DOT_BOUNDS_SINGLE_PENDULUM,
                U_BOUNDS_SINGLE_PENDULUM,
                n_samples=n_samples,
                focus_ratio=0.3,
                focus_scale=0.2,
            )
            # Propagate one step for next-state/actions matching existing interface
            # Wrap angles during data collection for consistency with MM/ID training
            X_plus = np.array([system.step(x, u[0], wrap_angles=True) for x, u in zip(states, actions)])
            U_plus = np.random.uniform(*U_BOUNDS_SINGLE_PENDULUM, size=(n_samples, 1))
            return states, actions, X_plus, U_plus

        def aux_sample_fn(system, n_samples):
            states, actions = _sample_mixed_pendulum(
                THETA_BOUNDS_SINGLE_PENDULUM,
                THETA_DOT_BOUNDS_SINGLE_PENDULUM,
                U_BOUNDS_SINGLE_PENDULUM,
                n_samples=n_samples,
                focus_ratio=0.3,
                focus_scale=0.2,
            )
            return states, actions

        def test_sampler(system, u_bounds_local, n_samples):
            # Use full pendulum bounds (large angles/velocities) to challenge LQR
            return system.generate_samples_auxiliary(
                (-0.5, 0.5),  # ±180 degrees 
                (0.0, 0.0),  # Full velocity range
                u_bounds_local,
                n_samples=n_samples,
            )

        return {
            "name": "single_pendulum",
            "pretty_name": "Single Pendulum",
            "build_system": build_system,
            "sample_fn": sample_fn,
            "aux_sample_fn": aux_sample_fn,
            "test_sampler": test_sampler,
            "extractor_kwargs": {
                "dt": DT_SINGLE_PENDULUM,
                "C": C_SINGLE_PENDULUM,
                "rho": RHO_SINGLE_PENDULUM,
                "gamma": GAMMA,
                "u_bounds": U_BOUNDS_SINGLE_PENDULUM,
            },
            "u_bounds": U_BOUNDS_SINGLE_PENDULUM,
            "supports_policy_heatmaps": False,
            "supports_lqr_rollouts": False,
            "lqr_rollout_length": None,
        }

    if system_name == "mountain_car":
        def build_system(N, M_offline):
            return mountain_car(
                delta_t=DT_MOUNTAIN_CAR,
                C=C_MOUNTAIN_CAR,
                rho=RHO_MOUNTAIN_CAR,
                gamma=GAMMA,
                N=N,
                M=M_offline,
                goal_position=GOAL_POSITION_MOUNTAIN_CAR,
            )

        def sample_fn(system, n_samples, use_lqr_rollouts = False, rollout_length = 500):
            return system.generate_samples(
                POSITION_BOUNDS_MOUNTAIN_CAR,
                VELOCITY_BOUNDS_MOUNTAIN_CAR,
                U_BOUNDS_MOUNTAIN_CAR,
                n_samples=n_samples,
            )

        def aux_sample_fn(system, n_samples):
            return system.generate_samples_auxiliary(
                POSITION_BOUNDS_MOUNTAIN_CAR,
                VELOCITY_BOUNDS_MOUNTAIN_CAR,
                U_BOUNDS_MOUNTAIN_CAR,
                n_samples=n_samples,
            )

        def test_sampler(system, u_bounds_local, n_samples):
            # Test states: mix of positions far from goal and near goal
            return system.generate_samples_auxiliary(
                POSITION_BOUNDS_MOUNTAIN_CAR,
                VELOCITY_BOUNDS_MOUNTAIN_CAR,
                u_bounds_local,
                n_samples=n_samples,
            )

        return {
            "name": "mountain_car",
            "pretty_name": "Mountain Car",
            "build_system": build_system,
            "sample_fn": sample_fn,
            "aux_sample_fn": aux_sample_fn,
            "test_sampler": test_sampler,
            "extractor_kwargs": {
                "dt": DT_MOUNTAIN_CAR,
                "C": C_MOUNTAIN_CAR,
                "rho": RHO_MOUNTAIN_CAR,
                "gamma": GAMMA,
                "u_bounds": U_BOUNDS_MOUNTAIN_CAR,
            },
            "u_bounds": U_BOUNDS_MOUNTAIN_CAR,
            "supports_policy_heatmaps": False,
            "supports_lqr_rollouts": False,
            "lqr_rollout_length": None,
        }

    if system_name == "scalar_cubic":
        def build_system(N, M_offline):
            return scalar_cubic(
                delta_t=DT_SCALAR_CUBIC,
                C=C_SCALAR_CUBIC,
                rho=RHO_SCALAR_CUBIC,
                gamma=GAMMA,
                N=N,
                M=M_offline,
            )

        def sample_fn(system, n_samples):
            return system.generate_samples(
                (X_BOUNDS_SCALAR_CUBIC),
                U_BOUNDS_SCALAR_CUBIC,
                n_samples=n_samples
            )

        def aux_sample_fn(system, n_samples):
            return system.generate_samples_auxiliary(
                X_BOUNDS_SCALAR_CUBIC,
                U_BOUNDS_SCALAR_CUBIC,
                n_samples=n_samples
            )

        def test_sampler(system, u_bounds_local, n_samples):
            return system.generate_samples_auxiliary(
                (-2, 2),
                u_bounds_local,
                n_samples=n_samples
            )

        return {
            "name": "scalar_cubic",
            "pretty_name": "Scalar Cubic",
            "build_system": build_system,
            "sample_fn": sample_fn,
            "aux_sample_fn": aux_sample_fn,
            "test_sampler": test_sampler,
            "extractor_kwargs": {
                "dt": DT_SCALAR_CUBIC,
                "C": C_SCALAR_CUBIC,
                "rho": RHO_SCALAR_CUBIC,
                "gamma": GAMMA,
                "u_bounds": U_BOUNDS_SCALAR_CUBIC,
            },
            "u_bounds": U_BOUNDS_SCALAR_CUBIC,
            "supports_policy_heatmaps": False,
            "supports_lqr_rollouts": False,
            "lqr_rollout_length": None,
        }

    raise ValueError(f"Unsupported system '{system_name}'. Available options: cartpole, single_pendulum, mountain_car, scalar_cubic.")


def run_one(seed, N, M_offline=DEFAULT_M_OFFLINE, use_scaling=True, regularization=DEFAULT_REGULARIZATION, 
                    data_pools=None, test_states=None, poly_scaler=None, return_mu=False, fixed_mu=None, 
                    visualize_policies=False, save_heatmap_path=None, analyze_divergence=False, 
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
    
    # Print K matrix for single pendulum
    if system_cfg["name"] == "single_pendulum":
        print(f"\n{'='*60}")
        print(f"LQR K matrix for single_pendulum (seed={seed}, N={N}):")
        print(f"{'='*60}")
        print(f"K shape: {K_lqr.shape}")
        print(f"K matrix:\n{K_lqr}")
        print(f"\nState representation: [theta, theta_dot]")
        print(f"K[0] (theta gain): {K_lqr[0, 0]:.6f}")
        print(f"K[1] (theta_dot gain): {K_lqr[0, 1]:.6f}")
        print(f"{'='*60}\n")

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
        regularization=regularization, fixed_mu=fixed_mu,
        Mi_tensor=Mi_tensor, Py_outer=Py_outer
    )
    
    status_m2 = status_info["stage2_status"]
    status_m1 = status_info["stage1_status"]
    
    # Extract and print gain K for moment matching policy (degree 1 only)
    if Q_mm is not None and degree == 1 and poly_scaler is not None:
        try:
            # For degree 1, features are [x1, x2, ..., xn, u]
            # Q matrix structure: [Q_xx, Q_xu; Q_ux, Q_uu]
            n_states = system.N_x
            n_controls = system.N_u
            
            # Partition Q matrix
            Q_xx = Q_mm[:n_states, :n_states]
            Q_xu = Q_mm[:n_states, n_states:]
            Q_ux = Q_mm[n_states:, :n_states]
            Q_uu = Q_mm[n_states:, n_states:]
            
            # Compute gain K: u = -K @ x, where K = Q_uu^{-1} @ Q_xu^T
            # From optimality: ∂/∂u [x^T Q_xx x + 2x^T Q_xu u + u^T Q_uu u] = 0
            # → 2Q_xu^T x + 2Q_uu u = 0 → u = -Q_uu^{-1} Q_xu^T x
            # So K = Q_uu^{-1} @ Q_xu^T has shape (n_controls, n_states)
            try:
                K_mm = np.linalg.solve(Q_uu, Q_xu.T)
            except np.linalg.LinAlgError:
                # Use regularized solve if Q_uu is singular
                reg = 1e-6 * np.trace(Q_uu) / Q_uu.shape[0]
                K_mm = np.linalg.solve(Q_uu + reg * np.eye(Q_uu.shape[0]), Q_xu.T)
            
            print(f"\n{'='*60}")
            print(f"Moment Matching K matrix (degree=1, seed={seed}, N={N}):")
            print(f"{'='*60}")
            print(f"K shape: {K_mm.shape}")
            print(f"K matrix:\n{K_mm}")
            
            # Print system-specific information
            if system_cfg["name"] == "single_pendulum":
                print(f"\nState representation: [theta, theta_dot]")
                print(f"K[0] (theta gain): {K_mm[0, 0]:.6f}")
                print(f"K[1] (theta_dot gain): {K_mm[0, 1]:.6f}")
            elif system_cfg["name"] == "cartpole":
                print(f"\nState representation: [x, x_dot, theta, theta_dot]")
                print(f"K[0] (x gain): {K_mm[0, 0]:.6f}")
                print(f"K[1] (x_dot gain): {K_mm[0, 1]:.6f}")
                print(f"K[2] (theta gain): {K_mm[0, 2]:.6f}")
                print(f"K[3] (theta_dot gain): {K_mm[0, 3]:.6f}")
            elif system_cfg["name"] == "scalar_cubic":
                print(f"\nState representation: [x]")
                print(f"K[0] (x gain): {K_mm[0, 0]:.6f}")
            else:
                print(f"\nState representation: {n_states} states")
                for i in range(n_states):
                    print(f"K[{i}] (state {i} gain): {K_mm[0, i]:.6f}")
            
            # Compare with LQR if available
            if 'K_lqr' in locals():
                print(f"\nLQR K matrix for comparison:")
                print(f"K_lqr:\n{K_lqr}")
                if K_mm.shape == K_lqr.shape:
                    K_diff = K_mm - K_lqr
                    print(f"\nDifference (MM - LQR):")
                    print(f"K_diff:\n{K_diff}")
                    print(f"Frobenius norm of difference: {np.linalg.norm(K_diff, 'fro'):.6f}")
            
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Failed to extract moment matching K matrix: {e}")
            import traceback
            traceback.print_exc()
    
    if Q_mm is not None:
        # Extract moment matching policy
        if poly_scaler is not None:
            mm_policy = extractor.extract_moment_matching_policy_analytical(
                system, Q_mm, poly_scaler
            )
            # mm_policy = extractor.extract_policy_grid_search(
            #     system, Q_mm, poly_scaler=poly_scaler, n_grid_points=101
            # )
        else:
            raise ValueError("Scaling method not supported")

    else:
        print("Moment matching failed")
        mm_policy = None

    # Visualize policies if requested
    if visualize_policies:
        if save_heatmap_path is None:
            save_heatmap_path = f"policy_heatmaps_N_{N}_seed_{seed}.png"
        visualize_policies_2d_heatmaps(lqr_policy, mm_policy, N, seed, save_path=save_heatmap_path)

    # Identity baseline
    identity_result = solve_identity_Q(P_z, P_z_next, L_xu, N, gamma)
    if isinstance(identity_result, tuple):
        status_id, Q_id = identity_result
    else:
        # Backward compatibility: if only status is returned
        status_id = identity_result
        Q_id = None
    
    # Extract and print gain K for identity policy (degree 1 only)
    if Q_id is not None and degree == 1 and poly_scaler is not None:
        try:
            # For degree 1, features are [x1, x2, ..., xn, u]
            # Q matrix structure: [Q_xx, Q_xu; Q_ux, Q_uu]
            n_states = system.N_x
            n_controls = system.N_u
            
            # Partition Q matrix
            Q_xx = Q_id[:n_states, :n_states]
            Q_xu = Q_id[:n_states, n_states:]
            Q_ux = Q_id[n_states:, :n_states]
            Q_uu = Q_id[n_states:, n_states:]
            
            # Compute gain K: u = -K @ x, where K = Q_uu^{-1} @ Q_xu^T
            # From optimality: ∂/∂u [x^T Q_xx x + 2x^T Q_xu u + u^T Q_uu u] = 0
            # → 2Q_xu^T x + 2Q_uu u = 0 → u = -Q_uu^{-1} Q_xu^T x
            # So K = Q_uu^{-1} @ Q_xu^T has shape (n_controls, n_states)
            try:
                K_id = np.linalg.solve(Q_uu, Q_xu.T)
            except np.linalg.LinAlgError:
                # Use regularized solve if Q_uu is singular
                reg = 1e-6 * np.trace(Q_uu) / Q_uu.shape[0]
                K_id = np.linalg.solve(Q_uu + reg * np.eye(Q_uu.shape[0]), Q_xu.T)
            
            print(f"\n{'='*60}")
            print(f"Identity K matrix (degree=1, seed={seed}, N={N}):")
            print(f"{'='*60}")
            print(f"K shape: {K_id.shape}")
            print(f"K matrix:\n{K_id}")
            
            # Print system-specific information
            if system_cfg["name"] == "single_pendulum":
                print(f"\nState representation: [theta, theta_dot]")
                print(f"K[0] (theta gain): {K_id[0, 0]:.6f}")
                print(f"K[1] (theta_dot gain): {K_id[0, 1]:.6f}")
            elif system_cfg["name"] == "cartpole":
                print(f"\nState representation: [x, x_dot, theta, theta_dot]")
                print(f"K[0] (x gain): {K_id[0, 0]:.6f}")
                print(f"K[1] (x_dot gain): {K_id[0, 1]:.6f}")
                print(f"K[2] (theta gain): {K_id[0, 2]:.6f}")
                print(f"K[3] (theta_dot gain): {K_id[0, 3]:.6f}")
            elif system_cfg["name"] == "scalar_cubic":
                print(f"\nState representation: [x]")
                print(f"K[0] (x gain): {K_id[0, 0]:.6f}")
            else:
                print(f"\nState representation: {n_states} states")
                for i in range(n_states):
                    print(f"K[{i}] (state {i} gain): {K_id[0, i]:.6f}")
            
            # Compare with LQR if available
            if 'K_lqr' in locals():
                print(f"\nLQR K matrix for comparison:")
                print(f"K_lqr:\n{K_lqr}")
                if K_id.shape == K_lqr.shape:
                    K_diff = K_id - K_lqr
                    print(f"\nDifference (Identity - LQR):")
                    print(f"K_diff:\n{K_diff}")
                    print(f"Frobenius norm of difference: {np.linalg.norm(K_diff, 'fro'):.6f}")
            
            # Compare with Moment Matching if available
            if 'K_mm' in locals() and K_mm is not None:
                if K_id.shape == K_mm.shape:
                    K_diff_mm = K_id - K_mm
                    print(f"\nDifference (Identity - Moment Matching):")
                    print(f"K_diff:\n{K_diff_mm}")
                    print(f"Frobenius norm of difference: {np.linalg.norm(K_diff_mm, 'fro'):.6f}")
            
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Failed to extract identity K matrix: {e}")
            import traceback
            traceback.print_exc()

    # Optionally replace MM policy with a multivariate cubic surrogate in full state to stabilize degree-2 behavior
    cubic_metrics = None
    policy_for_eval = mm_policy

    # Extract identity policy if Q_id is available
    id_policy = None
    if Q_id is not None and poly_scaler is not None:
        try:
            id_policy = extractor.extract_moment_matching_policy_analytical(
                system, Q_id, poly_scaler
            )
        except Exception as e:
            print(f"Failed to extract identity policy: {e}")
            id_policy = None
    
    if policy_for_eval is not None:
        # Prepare divergence analysis path if requested but not provided
        if analyze_divergence and divergence_analysis_path is None:
            divergence_analysis_path = f"divergence_analysis_N_{N}_seed_{seed}.png"
        
        results_policy = extractor.compare_policies(
            system, lqr_policy, policy_for_eval, test_states, N, horizon=5000,
            analyze_divergence=analyze_divergence,
            divergence_threshold=0.1,
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
                    system, lqr_policy, id_policy, test_states, N, horizon=4000,
                    analyze_divergence=False,
                    divergence_threshold=0.1,
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
        # Cubic surrogate metrics (if used)
        "cubic_rmse": cubic_metrics["rmse"] if cubic_metrics is not None else None,
        "cubic_mae": cubic_metrics["mae"] if cubic_metrics is not None else None,
        "cubic_r2": cubic_metrics["r2"] if cubic_metrics is not None else None,
    }
    
    if return_mu:
        return result, mu
    else:
        return result

def sweep(N_values=range(50, 501, 50), seeds=range(10), M_offline=DEFAULT_M_OFFLINE, 
                  use_scaling=True, regularization=DEFAULT_REGULARIZATION, fix_mu_from_first_n=False, 
                  exclude_u_squared=False, visualize_policies=False, heatmap_dir=None,
                  analyze_divergence=False, divergence_dir=None, divergence_horizon=10000,
                  use_lqr_rollouts=False, system_cfg=None):
    if system_cfg is None:
        raise ValueError("system_cfg must be provided to sweep()")

    if visualize_policies and not system_cfg["supports_policy_heatmaps"]:
        print(f"Policy visualization is not implemented for {system_cfg['pretty_name']}. Disabling visualize_policies.")
        visualize_policies = False

    if use_lqr_rollouts and not system_cfg["supports_lqr_rollouts"]:
        print(f"LQR rollout sampling not available for {system_cfg['pretty_name']}. Using purely random samples.")
        use_lqr_rollouts = False

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
        test_states, _ = system_cfg["test_sampler"](system_temp, extractor.u_bounds, n_samples=20)
        
        # Fit polynomial features ONCE on the full dataset
        z_pool = np.concatenate([x_pool, u_pool], axis=1)
        z_plus_pool = np.concatenate([x_plus_pool, u_plus_pool], axis=1)
        Z_pool = np.concatenate([z_pool, z_plus_pool], axis=0)
        
        if use_scaling:
            # Separate scaling for states and actions
            dx = x_pool.shape[1]  # State dimensions
            du = u_pool.shape[1]   # Action dimensions
            
            scaling_method = 'standard'
            
            poly_scaler = PolynomialFeatureScaler(
                degree=degree, 
                scaling_method=scaling_method,
                dx=dx,
                du=du,
                exclude_u_squared=exclude_u_squared
            )
            P_pool = poly_scaler.fit_transform(Z_pool)
            print(f"  Fitted {scaling_method} polynomial scaler (degree={degree}, dx={dx}, du={du}) on {len(Z_pool)} samples for seed {seed}")
            
        else:
            # Determine dimensions from the data for exclude_u_squared
            dx = x_pool.shape[1]  # State dimensions
            du = u_pool.shape[1]   # Action dimensions
            poly_scaler = PolynomialFeatureScaler(
                degree=degree, 
                scaling_method='none',
                dx=dx if exclude_u_squared else None,
                du=du if exclude_u_squared else None,
                exclude_u_squared=exclude_u_squared
            )
            P_pool = poly_scaler.fit_transform(Z_pool)
            print(f"  Fitted polynomial scaler (degree={degree}) on {len(Z_pool)} samples for seed {seed}")

        # Split polynomial features once and cache
        # P_pool = [z_pool features; z_plus_pool features] where z_pool = [x_pool, u_pool]
        P_z_pool = P_pool[:N_max]  # Features for (x, u) pairs: shape (N_max, d_features)
        P_z_next_pool = P_pool[N_max:]  # Features for (x_plus, u_plus) pairs: shape (N_max, d_features)
        P_y_pool = poly_scaler.transform(np.concatenate([x_aux, u_aux], axis=1))

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
        mu_fixed = None
        for i, N in enumerate(N_values):
            if i == 0 and fix_mu_from_first_n:
                # For the first N, compute mu normally and store it
                print(f"Computing mu for first N={N}, seed={seed}")
                save_path = None
                if visualize_policies and heatmap_dir:
                    os.makedirs(heatmap_dir, exist_ok=True)
                    save_path = os.path.join(heatmap_dir, f"policy_heatmaps_N_{N}_seed_{seed}.png")
                divergence_path = None
                if analyze_divergence and divergence_dir:
                    os.makedirs(divergence_dir, exist_ok=True)
                    divergence_path = os.path.join(divergence_dir, f"divergence_analysis_N_{N}_seed_{seed}.png")
                result = run_one(
                    seed=seed, N=int(N), M_offline=M_offline, 
                    use_scaling=use_scaling, regularization=regularization,
                    data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler,
                    return_mu=True, visualize_policies=visualize_policies, save_heatmap_path=save_path,
                    analyze_divergence=analyze_divergence, divergence_analysis_path=divergence_path,
                    divergence_horizon=divergence_horizon, system_cfg=system_cfg
                )
                if result is not None and len(result) == 2:
                    result_dict, mu = result
                    mu_fixed = mu
                    print(f"Fixed mu computed with shape {mu_fixed.shape}")
                    results.append(result_dict)
                else:
                    print(f"Failed to compute mu for first N, proceeding without fixed mu")
                    save_path = None
                    if visualize_policies and heatmap_dir:
                        import os
                        os.makedirs(heatmap_dir, exist_ok=True)
                        save_path = os.path.join(heatmap_dir, f"policy_heatmaps_N_{N}_seed_{seed}.png")
                    divergence_path = None
                    if analyze_divergence and divergence_dir:
                        os.makedirs(divergence_dir, exist_ok=True)
                        divergence_path = os.path.join(divergence_dir, f"divergence_analysis_N_{N}_seed_{seed}.png")
                    results.append(run_one(
                        seed=seed, N=int(N), M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler,
                        visualize_policies=visualize_policies, save_heatmap_path=save_path,
                        analyze_divergence=analyze_divergence, divergence_analysis_path=divergence_path,
                        divergence_horizon=divergence_horizon, system_cfg=system_cfg
                    ))
            else:
                # For subsequent N values, use the fixed mu
                if mu_fixed is not None and fix_mu_from_first_n:
                    print(f"Using fixed mu for N={N}, seed={seed}")
                    save_path = None
                    if visualize_policies and heatmap_dir:
                        import os
                        os.makedirs(heatmap_dir, exist_ok=True)
                        save_path = os.path.join(heatmap_dir, f"policy_heatmaps_N_{N}_seed_{seed}.png")
                    divergence_path = None
                    if analyze_divergence and divergence_dir:
                        os.makedirs(divergence_dir, exist_ok=True)
                        divergence_path = os.path.join(divergence_dir, f"divergence_analysis_N_{N}_seed_{seed}.png")
                    results.append(run_one(
                        seed=seed, N=int(N), M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler,
                        fixed_mu=mu_fixed, visualize_policies=visualize_policies, save_heatmap_path=save_path,
                        analyze_divergence=analyze_divergence, divergence_analysis_path=divergence_path,
                        system_cfg=system_cfg
                    ))
                else:
                    # Fallback to normal computation
                    save_path = None
                    if visualize_policies and heatmap_dir:
                        os.makedirs(heatmap_dir, exist_ok=True)
                        save_path = os.path.join(heatmap_dir, f"policy_heatmaps_N_{N}_seed_{seed}.png")
                    divergence_path = None
                    if analyze_divergence and divergence_dir:
                        os.makedirs(divergence_dir, exist_ok=True)
                        divergence_path = os.path.join(divergence_dir, f"divergence_analysis_N_{N}_seed_{seed}.png")
                    results.append(run_one(
                        seed=seed, N=int(N), M_offline=M_offline, 
                        use_scaling=use_scaling, regularization=regularization,
                        data_pools=data_pools, test_states=test_states, poly_scaler=poly_scaler,
                        visualize_policies=visualize_policies, save_heatmap_path=save_path,
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
    parser.add_argument("--out_json", type=str, default="../results/test_single_pendulum_mixedclipping_complete.json", help="Where to save raw results JSON")
    parser.add_argument("--plot_png", type=str, default="../figures/test_single_pendulum_mixedclipping.pdf", help="Where to save the plot")
    parser.add_argument("--plot_policy", type=str, default="../figures/test_single_pendulum_mixedclipping.pdf", help="Where to save the policy comparison plot")
    parser.add_argument("--M_offline", type=int, default=1000, help="Offline pool size (default: 500)")
    parser.add_argument("--use_scaling", action="store_true", default=False, help="Use feature scaling (default: False)")
    parser.add_argument("--regularization", type=float, default=1e-4, help="Regularization parameter (default: 1e-4)")
    parser.add_argument("--fix_mu", action="store_true", default=False, help="Fix mu from first N and reuse for all subsequent N values")
    parser.add_argument("--exclude_u_squared", action="store_true", default=False, help="Exclude u^2 terms from polynomial features (for degree 2)")
    parser.add_argument("--visualize_policies", action="store_true", default=False, help="Generate 2D heatmaps comparing LQR and MM policies")
    parser.add_argument("--heatmap_dir", type=str, default="policy_heatmaps", help="Directory to save policy heatmaps (default: policy_heatmaps)")
    parser.add_argument("--analyze_divergence", action="store_true", default=False, help="Analyze where LQR and MM policies diverge (default: False)")
    parser.add_argument("--divergence_dir", type=str, default="divergence_analysis", help="Directory to save divergence analysis plots (default: divergence_analysis)")
    parser.add_argument("--divergence_horizon", type=int, default=10000, help="Simulation horizon for divergence analysis (default: 10000, use smaller value like 1000 for faster analysis)")
    parser.add_argument("--use_lqr_rollouts", type=lambda x: (str(x).lower() == 'true'), default=None, 
                       help="Use mixed sampling: half random, half from LQR rollouts near equilibrium (default: True). Use --no_lqr_rollouts to disable.")
    parser.add_argument("--no_lqr_rollouts", dest="use_lqr_rollouts", action="store_const", const=False,
                       help="Disable LQR rollouts, use only random sampling")
    parser.add_argument("--system", type=str, default="cartpole",
                        choices=["cartpole", "single_pendulum", "mountain_car", "scalar_cubic"],
                        help="Dynamics to study (default: cartpole)")
    args = parser.parse_args()
    N_values = list(range(args.Nmin, args.Nmax + 1, args.Nstep))
    seeds = list(range(args.seeds))
    # seeds = [s+6 for s in range(args.seeds)]
    
    system_cfg = get_system_config(args.system)
    
    # Handle use_lqr_rollouts: default follows system capabilities unless explicitly set
    if args.use_lqr_rollouts is None:
        use_lqr_rollouts = system_cfg["supports_lqr_rollouts"]
    else:
        use_lqr_rollouts = args.use_lqr_rollouts

    try:
        results = sweep(
            N_values=N_values, seeds=seeds, M_offline=args.M_offline,
            use_scaling=args.use_scaling, regularization=args.regularization,
            fix_mu_from_first_n=args.fix_mu, exclude_u_squared=args.exclude_u_squared,
            visualize_policies=args.visualize_policies, heatmap_dir=args.heatmap_dir,
            analyze_divergence=args.analyze_divergence, divergence_dir=args.divergence_dir,
            divergence_horizon=args.divergence_horizon, use_lqr_rollouts=use_lqr_rollouts,
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
        id_success=("id_success", "mean"),
        id_costs=("id_costs", mean_no_inf),
        id_costs_min=("id_costs", min_no_inf),
        id_costs_max=("id_costs", max_no_inf),
        id_costs_std=("id_costs_std", mean_no_inf),
        id_costs_median=("id_costs_median", mean_no_inf),
    ).reset_index()

    # Convert to percentages
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_pct"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_pct"]
    agg["moment_bounded_pct_min"] = 100 * agg["moment_bounded_pct_min"]
    agg["identity_bounded_pct_min"] = 100 * agg["identity_bounded_pct_min"]
    agg["moment_bounded_pct_max"] = 100 * agg["moment_bounded_pct_max"]
    agg["identity_bounded_pct_max"] = 100 * agg["identity_bounded_pct_max"]

    # Save raw + aggregated with descriptive filename
    descriptive_out_path = build_descriptive_out_path(
        args.out_json,
        args.system,
        args.M_offline,
        args.use_scaling,
        use_lqr_rollouts,
    )
    ensure_parent_dir(descriptive_out_path)
    df.to_json(descriptive_out_path, orient="records", indent=2)
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
    ensure_parent_dir(args.plot_png)
    plt.savefig(args.plot_png, dpi=150)
    print("Saved:", descriptive_out_path, "and", args.plot_png)

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
    
    # Plot identity policy with shaded area
    plt.plot(agg["N"], agg["id_costs"], 
                marker="^", label="Identity", color="purple")
    plt.fill_between(agg["N"], agg["id_costs_min"], agg["id_costs_max"], 
                     alpha=0.3, color="purple")
    
    plt.xlabel("N (number of samples)")
    plt.ylabel("Cost")
    plt.title("Policy comparison vs sample size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ensure_parent_dir(args.plot_policy)
    plt.savefig(args.plot_policy, dpi=150)
    print("Saved policy comparison plot:", args.plot_policy)
