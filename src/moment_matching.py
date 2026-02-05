import cvxpy as cp
import numpy as np
# from config import ADAPTIVE_REGULARIZATION_SCALING
# from feature_scaling import AdaptiveRegularizationScaler

def extract_and_psd_project_uu(Q_matrix, poly_scaler, dx=None, du=None):
    """
    Extract the uu-component of Q matrix and project it to be positive semi-definite (PSD).
    
    The uu-component corresponds to features that depend only on control input u
    (no state variables). For polynomial features:
    - Degree 1: features like [u]
    - Degree 2: features like [u, u^2]
    - etc.
    
    Args:
        Q_matrix: Learned Q matrix in polynomial feature space, shape (d, d)
        poly_scaler: PolynomialFeatureScaler object with fitted polynomial features
        dx: Number of state dimensions (if None, inferred from poly_scaler)
        du: Number of action dimensions (if None, inferred from poly_scaler, default 1)
    
    Returns:
        Q_uu: PSD-projected uu-component of Q, shape (n_u_features, n_u_features)
        u_feature_indices: Indices of features that correspond to u-only terms
        Q_uu_original: Original (non-PSD) uu-component before projection
    """
    if Q_matrix is None:
        raise ValueError("Q_matrix cannot be None")
    
    # Get polynomial feature powers
    if not hasattr(poly_scaler, 'poly'):
        raise ValueError("poly_scaler must have a 'poly' attribute with polynomial features")
    
    poly = poly_scaler.poly
    if not hasattr(poly, 'powers_'):
        raise ValueError("Polynomial features must be fitted (have 'powers_' attribute)")
    
    powers = poly.powers_  # shape (n_features, n_inputs)
    n_features = powers.shape[0]
    
    # Infer dimensions if not provided
    if dx is None:
        if hasattr(poly_scaler, 'dx') and poly_scaler.dx is not None:
            dx = poly_scaler.dx
        else:
            # Try to infer from powers: assume last dimension is control
            # This is a heuristic - may not work for all cases
            dx = powers.shape[1] - 1
            if dx < 1:
                dx = 0  # Fallback
    
    if du is None:
        if hasattr(poly_scaler, 'du') and poly_scaler.du is not None:
            du = poly_scaler.du
        else:
            du = 1  # Default to 1 control dimension
    
    # Identify features that depend only on u (no state variables)
    # A feature is u-only if all state powers are zero
    state_powers = powers[:, :dx] if dx > 0 else np.zeros((n_features, 0))
    control_powers = powers[:, dx:dx+du] if du > 0 else np.zeros((n_features, 0))
    
    # Feature is u-only if sum of state powers is 0 for all state dimensions
    state_power_sum = np.sum(state_powers, axis=1)  # Sum of state powers for each feature
    u_only_mask = (state_power_sum == 0)  # True for features with no state variables
    
    # Get indices of u-only features
    u_feature_indices = np.where(u_only_mask)[0]
    
    if len(u_feature_indices) == 0:
        raise ValueError("No u-only features found. Check polynomial feature structure.")
    
    # Extract Q_uu submatrix
    Q_uu_original = Q_matrix[np.ix_(u_feature_indices, u_feature_indices)]
    
    # Project to PSD using eigenvalue decomposition
    # Q_uu_psd = V @ diag(max(0, lambda_i)) @ V^T
    eigenvals, eigenvecs = np.linalg.eigh(Q_uu_original)
    
    # Set negative eigenvalues to zero
    eigenvals_psd = np.maximum(eigenvals, 0.0)
    
    # Reconstruct PSD matrix
    Q_uu = eigenvecs @ np.diag(eigenvals_psd) @ eigenvecs.T
    
    # Ensure symmetry (numerical precision)
    Q_uu = (Q_uu + Q_uu.T) / 2.0
    
    return Q_uu, u_feature_indices, Q_uu_original

def replace_uu_in_Q(Q_matrix, Q_uu_psd, u_feature_indices):
    """
    Replace the uu-component in Q matrix with the PSD-projected version.
    
    Args:
        Q_matrix: Original Q matrix, shape (d, d)
        Q_uu_psd: PSD-projected uu-component, shape (n_u_features, n_u_features)
        u_feature_indices: Indices of u-only features
    
    Returns:
        Q_modified: Q matrix with uu-component replaced by PSD version, shape (d, d)
    """
    Q_modified = Q_matrix.copy()
    Q_modified[np.ix_(u_feature_indices, u_feature_indices)] = Q_uu_psd
    return Q_modified

def solve_moment_matching_Q(P_z, P_z_next, P_y, L_xu, N, M, gamma, regularization=1e-4, fixed_mu=None, Mi_tensor=None, Py_outer=None):
    """
    Moment matching approach to learn Q matrix with robust scaling for degree 2+ polynomials
    
    (LP #1) Solve for lambda, mu with moment-matching + C ≈ I objective.
    (LP #2) Given mu (so C), solve max c^T vec(Q) s.t. Bellman inequality.
    Returns: status_lp2 (string), and optionally statuses of LP #1 and shapes.
    
    Args:
        fixed_mu: If provided, skip LP #1 and use this fixed mu value for LP #2
    """
    d = P_z.shape[1]
    # M_actual = P_y.shape[0]

    if Mi_tensor is not None:
        if Mi_tensor.shape[0] != N or Mi_tensor.shape[1] != d or Mi_tensor.shape[2] != d:
            raise ValueError("Mi_tensor shape mismatch with provided features")
        Mi_tensor_local = Mi_tensor
    else:
        Mi_tensor_local = np.einsum('bi,bj->bij', P_z, P_z, optimize=True) - \
                          gamma * np.einsum('bi,bj->bij', P_z_next, P_z_next, optimize=True)

    if Py_outer is not None:
        if Py_outer.shape[0] != M or Py_outer.shape[1] != d or Py_outer.shape[2] != d:
            raise ValueError("Py_outer shape mismatch with provided features")
        Py_outer_local = Py_outer
    else:
        Py_outer_local = np.einsum('bi,bj->bij', P_y, P_y, optimize=True)

    Mi_flat = Mi_tensor_local.reshape(N, d * d).T
    Py_flat = Py_outer_local.reshape(M, d * d).T

    Mi_flat_const = cp.Constant(Mi_flat)
    Py_flat_const = cp.Constant(Py_flat)
    
    # Scale regularization by feature dimension to account for larger feature space
    # With more features, the Frobenius norm constraint naturally scales with sqrt(d)
    # So we need to scale regularization accordingly to maintain the same relative tolerance
    dimension_scaled_regularization = regularization * np.sqrt(d)

    # Common solver settings and robust solver wrapper
    # Note: Very strict tolerances (1e-10) can cause numerical issues with poorly scaled problems
    # We'll use more relaxed tolerances and adjust based on problem scale
    mosek_strict = {
        'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-8,  # Relaxed from 1e-10 for better numerical stability
        'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-8,  # Relaxed from 1e-10
        'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-8,  # Relaxed from 1e-10
        'MSK_IPAR_INTPNT_BASIS': 1,
        'MSK_IPAR_PRESOLVE_USE': 1,
    }
    scs_opts = dict(max_iters=20000, eps=1e-6, acceleration_lookback=50)
    ecos_opts = dict(max_iters=10000, abstol=1e-8, reltol=1e-8, feastol=1e-8)

    def solve_robust(prob):
        try:
            prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_strict)
        except Exception as e:
            import traceback
            print(f"  MOSEK solver exception: {type(e).__name__}: {e}")
            print(f"  Full exception traceback:")
            traceback.print_exc()
            # If exception occurs, status might be None - check and set a default
            if prob.status is None:
                print(f"  WARNING: prob.status is None after MOSEK exception - solver may not have run")
                # Try to get more info about the problem
                try:
                    # Check if problem is DCP (Disciplined Convex Programming)
                    if not prob.is_dcp():
                        print(f"  Problem is not DCP (Disciplined Convex Programming)")
                except Exception as dcp_e:
                    print(f"  Could not check DCP status: {dcp_e}")
            
            # Try again with verbose=True to get more detailed error information
            print(f"  Retrying MOSEK with verbose=True to get detailed error information...")
            try:
                prob.solve(solver=cp.MOSEK, verbose=True, mosek_params=mosek_strict)
            except Exception as e2:
                print(f"  MOSEK verbose retry also failed: {type(e2).__name__}: {e2}")
                import traceback
                traceback.print_exc()
        # if prob.status == "optimal_inaccurate":
        #     # Try to polish with SCS
        #     try:
        #         prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
        #     except Exception:
        #         pass
        # if prob.status not in ("optimal",):
        #     # Fallback chain
        #     try:
        #         prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
        #     except Exception:
        #         try:
        #             prob.solve(solver=cp.ECOS, verbose=False, **ecos_opts)
        #         except Exception:
        #             try:
        #                 prob.solve(solver=cp.CLARABEL, verbose=False)
        #             except Exception:
        #                 prob.solve(verbose=False)
        
        # Handle None status explicitly
        if prob.status is None:
            print(f"  ERROR: prob.status is None - solver did not complete")
            return "solver_error"
        
        return prob.status

    # ----- LP #1: Moment matching (λ, μ) -----
    if fixed_mu is not None:
        # Skip LP #1 and use the provided fixed mu
        print(f"Using fixed mu with shape {fixed_mu.shape}")
        mu = fixed_mu
        status1 = "fixed_mu_used"
        # Compute the weighted sum: sum_i mu_i * y_i y_i^T
        Py_weighted = np.tensordot(mu, Py_outer_local, axes=(0, 0))
        # Compute Frobenius norm distance from identity matrix
        identity_distance = np.linalg.norm(np.eye(d) - Py_weighted, ord='fro')
        print(f"Stage 1 (fixed mu): ||I - sum_i mu_i * y_i y_i^T||_F = {identity_distance:.6e}")
    else:
        # Solve LP #1 as usual
        lambda_var = cp.Variable(N, nonneg=True)
        mu_var = cp.Variable(M, nonneg=True)

        moment_vector = Mi_flat_const @ lambda_var - Py_flat_const @ mu_var
        moment_norm = cp.norm(moment_vector, 'fro')

        # Enhanced constraints for better numerical stability
        constraints = [
            # moment_norm <= dimension_scaled_regularization,  # Adaptive relaxed moment matching
            moment_norm <= 0.0,
            # cp.sum(mu_var) == 1.0,
        ]

        # Objective: minimize ||I - sum_i mu_i * y_i y_i^T||_F
        # Py_flat_const @ mu_var gives vec(sum_i mu_i * y_i y_i^T) with shape (d*d,)
        # np.eye(d).flatten() gives vec(I) with shape (d*d,)
        # For a vector, use norm type 2 (equivalent to Frobenius for flattened matrices)
        identity_norm = cp.norm(cp.Constant(np.eye(d).flatten()) - Py_flat_const @ mu_var, 'fro')
        objective = cp.Minimize(moment_norm + 0.5*identity_norm)


        prob = cp.Problem(objective, constraints)
        status1 = solve_robust(prob)

        if status1 not in ["optimal", "optimal_inaccurate"]:
            print(f"Stage 1 failed with status: {status1}")
            return None, None, {"stage1_status": status1, "stage2_status": "failed_stage1"}
        
        print(f"Stage 1 status: {status1}")
        mu = mu_var.value
        # print(f"mu: {mu}")
        
        # Compute the weighted sum: sum_i mu_i * y_i y_i^T
        Py_weighted = np.tensordot(mu, Py_outer_local, axes=(0, 0))
        # Compute Frobenius norm distance from identity matrix
        identity_distance = np.linalg.norm(np.eye(d) - Py_weighted, ord='fro')
        print(f"Stage 1: ||I - sum_i mu_i * y_i y_i^T||_F = {identity_distance:.6e}")
        print(f"Stage 1: Moment norm: {moment_norm.value:.6e}")

    # ----- LP #2: Solve for Q with c = vec(P_y^T diag(mu) P_y) -----
    Q_var = cp.Variable((d, d), symmetric=True)

    # DIAGNOSTIC: Check scale of constraint matrices and objective
    Mi_max_abs = np.max([np.max(np.abs(Mi_tensor_local[i])) for i in range(N)])
    Py_weighted_max_abs = np.max(np.abs(Py_weighted))
    L_xu_max = np.max(L_xu)
    L_xu_min = np.min(L_xu[L_xu > 0]) if np.any(L_xu > 0) else np.min(L_xu)
    
    print(f"Stage 2 scaling diagnostics:")
    print(f"  max|Mi|: {Mi_max_abs:.6e}")
    print(f"  max|Py_weighted|: {Py_weighted_max_abs:.6e}")
    print(f"  max L_xu: {L_xu_max:.6e}, min L_xu: {L_xu_min:.6e}")
    
    # NORMALIZATION: Scale problem to improve numerical stability
    # If values are too large (>1e6), normalize to prevent MOSEK numerical issues
    scale_factor = 1.0
    if Mi_max_abs > 1e6 or Py_weighted_max_abs > 1e6:
        # Choose scale factor to bring largest values to ~1e3 range
        max_val = max(Mi_max_abs, Py_weighted_max_abs)
        scale_factor = 1e3 / max_val
        print(f"  WARNING: Large values detected! Normalizing by factor {scale_factor:.6e}")
        print(f"  This will scale Mi matrices and Py_weighted to improve numerical stability")
    
    # Bellman constraints (with optional scaling)
    cons = []
    for i in range(N):
        Mi = Mi_tensor_local[i] * scale_factor
        cons.append(cp.sum(cp.multiply(Q_var, Mi)) <= L_xu[i] * scale_factor)
    
    # Objective (with optional scaling)
    Py_weighted_scaled = Py_weighted * scale_factor
    obj = cp.Maximize(cp.sum(cp.multiply(Q_var, Py_weighted_scaled)))
    prob_lp = cp.Problem(obj, cons)

    status2 = solve_robust(prob_lp)

    Q_learned_mat = Q_var.value
    if Q_learned_mat is None:
        print(f"Stage 2 failed with status: {status2}")
        return None, None, {"stage1_status": status1, "stage2_status": status2}
    
    print(f"Stage 2 status: {status2}")
    return Q_learned_mat, mu, {"stage1_status": status1, "stage2_status": status2}


def solve_identity_Q(P_z, P_z_next, L_xu, N, gamma):
    """
    Classical baseline: skip moment matching, solve
       min trace(Q)  s.t.  F vec(Q) <= l(x,u).
    Returns status string.
    """

    # Features
    d = P_z.shape[1]

    Q_id = cp.Variable((d, d), symmetric=True)  # identity baseline Q (must be symmetric for quadratic form)

    # Create block diagonal matrix C = diag(I_dx, 0.8 * I_du)
    C_matrix = np.eye(d)
    # C_matrix[:4, :4] = np.eye(4)  # Identity for state dimensions
    # C_matrix[4:, 4:] = 0.8 * np.eye(1)  # 0.8 * Identity for input dimensions

    cons = []
    for i in range(N):
        p = P_z[i]
        pn = P_z_next[i]
        Mi = np.outer(p, p) - gamma * np.outer(pn, pn)
        # Use trace instead of sum(multiply) for better numerical stability
        cons.append(cp.trace(Q_id @ cp.Constant(Mi)) <= L_xu[i])

    objective = cp.Maximize(cp.trace(cp.Constant(C_matrix) @ Q_id))

    # Local robust solver wrapper for this function
    mosek_strict = {
        'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-10,
        'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-10,
        'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-10,
        'MSK_IPAR_INTPNT_BASIS': 1,
        'MSK_IPAR_PRESOLVE_USE': 1,
    }
    scs_opts = dict(max_iters=20000, eps=1e-6, acceleration_lookback=50)
    ecos_opts = dict(max_iters=10000, abstol=1e-8, reltol=1e-8, feastol=1e-8)

    def solve_robust_local(prob):
        try:
            prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_strict)
        except Exception as e:
            import traceback
            print(f"  MOSEK solver exception: {type(e).__name__}: {e}")
            print(f"  Full exception traceback:")
            traceback.print_exc()
            # If exception occurs, status might be None - check and set a default
            if prob.status is None:
                print(f"  WARNING: prob.status is None after MOSEK exception - solver may not have run")
            
            # Try again with verbose=True to get more detailed error information
            print(f"  Retrying MOSEK with verbose=True to get detailed error information...")
            try:
                prob.solve(solver=cp.MOSEK, verbose=True, mosek_params=mosek_strict)
            except Exception as e2:
                print(f"  MOSEK verbose retry also failed: {type(e2).__name__}: {e2}")
                import traceback
                traceback.print_exc()
        # if prob.status == "optimal_inaccurate":
        #     try:
        #         prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
        #     except Exception:
        #         pass
        # if prob.status not in ("optimal",):
        #     try:
        #         prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
        #     except Exception:
        #         try:
        #             prob.solve(solver=cp.ECOS, verbose=False, **ecos_opts)
        #         except Exception:
        #             try:
        #                 prob.solve(solver=cp.CLARABEL, verbose=False)
        #             except Exception:
        #                 prob.solve(verbose=False)
        
        # Handle None status explicitly
        if prob.status is None:
            print(f"  ERROR: prob.status is None - solver did not complete")
            return "solver_error"
        
        return prob.status

    prob = cp.Problem(objective, cons)
    status = solve_robust_local(prob)

    print(f"Identity Q status: {status}")
    
    # Extract Q_id matrix if solution is optimal
    if status in ("optimal", "optimal_inaccurate"):
        Q_id_value = Q_id.value
        if Q_id_value is not None:
            return status, Q_id_value
        else:
            return status, None
    else:
        return status, None