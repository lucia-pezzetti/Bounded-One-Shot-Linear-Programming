import cvxpy as cp
import numpy as np
# from config import ADAPTIVE_REGULARIZATION_SCALING
# from feature_scaling import AdaptiveRegularizationScaler

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
    
        # Handle None status explicitly
        if prob.status is None:
            print(f"  ERROR: prob.status is None - solver did not complete")
            return "solver_error"
        
        return prob.status

    lambda_var = cp.Variable(N, nonneg=True)
    mu_var = cp.Variable(M, nonneg=True)

    moment_vector = Mi_flat_const @ lambda_var - Py_flat_const @ mu_var
    moment_norm = cp.norm(moment_vector, 'fro')

    # Enhanced constraints for better numerical stability
    constraints = [
        # moment_norm <= dimension_scaled_regularization,  # Adaptive relaxed moment matching
        moment_norm <= 0.0,
        cp.sum(mu_var) == 1.0,
    ]

    # Objective: minimize ||I - sum_i mu_i * y_i y_i^T||_F
    # Py_flat_const @ mu_var gives vec(sum_i mu_i * y_i y_i^T) with shape (d*d,)
    # np.eye(d).flatten() gives vec(I) with shape (d*d,)
    # For a vector, use norm type 2 (equivalent to Frobenius for flattened matrices)
    identity_norm = cp.norm(cp.Constant(np.eye(d).flatten()) - Py_flat_const @ mu_var, 'fro')
    objective = cp.Minimize(moment_norm) # + 0.5*identity_norm)


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

    # print the found direction
    # print(f"Stage 1: Found direction: {Py_weighted}")

    # ----- LP #2: Solve for Q with c = vec(P_y^T diag(mu) P_y) -----
    Q_var = cp.Variable((d, d))

    # Bellman constraints (with optional scaling)
    cons = []
    for i in range(N):
        Mi = Mi_tensor_local[i]
        cons.append(cp.sum(cp.multiply(Q_var, Mi)) <= L_xu[i])
    
    # Objective (with optional scaling)
    obj = cp.Maximize(cp.sum(cp.multiply(Q_var, Py_weighted)))
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

    Q_id = cp.Variable((d, d))  # identity baseline Q (must be symmetric for quadratic form)

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