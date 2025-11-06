import cvxpy as cp
from scipy import sparse
import numpy as np
# from config import ADAPTIVE_REGULARIZATION_SCALING
# from feature_scaling import AdaptiveRegularizationScaler

def solve_moment_matching_Q(P_z, P_z_next, P_y, L_xu, N, M, gamma, regularization=1e-4, fixed_mu=None):
    """
    Moment matching approach to learn Q matrix with robust scaling for degree 2+ polynomials
    
    (LP #1) Solve for lambda, mu with moment-matching + C ≈ I objective.
    (LP #2) Given mu (so C), solve max c^T vec(Q) s.t. Bellman inequality.
    Returns: status_lp2 (string), and optionally statuses of LP #1 and shapes.
    
    Args:
        fixed_mu: If provided, skip LP #1 and use this fixed mu value for LP #2
    """
    d = P_z.shape[1]
    
    # Scale regularization by feature dimension to account for larger feature space
    # With more features, the Frobenius norm constraint naturally scales with sqrt(d)
    # So we need to scale regularization accordingly to maintain the same relative tolerance
    dimension_scaled_regularization = regularization * np.sqrt(d)

    # Common solver settings and robust solver wrapper
    mosek_strict = {
        'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-10,
        'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-10,
        'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-10,
        'MSK_IPAR_INTPNT_BASIS': 1,
        'MSK_IPAR_PRESOLVE_USE': 1,
    }
    scs_opts = dict(max_iters=20000, eps=1e-6, acceleration_lookback=50)
    ecos_opts = dict(max_iters=10000, abstol=1e-8, reltol=1e-8, feastol=1e-8)

    def solve_robust(prob):
        try:
            prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_strict)
        except Exception:
            pass
        if prob.status == "optimal_inaccurate":
            # Try to polish with SCS
            try:
                prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
            except Exception:
                pass
        if prob.status not in ("optimal",):
            # Fallback chain
            try:
                prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
            except Exception:
                try:
                    prob.solve(solver=cp.ECOS, verbose=False, **ecos_opts)
                except Exception:
                    try:
                        prob.solve(solver=cp.CLARABEL, verbose=False)
                    except Exception:
                        prob.solve(verbose=False)
        return prob.status

    # ----- LP #1: Moment matching (λ, μ) -----
    if fixed_mu is not None:
        # Skip LP #1 and use the provided fixed mu
        print(f"Using fixed mu with shape {fixed_mu.shape}")
        mu = fixed_mu
        status1 = "fixed_mu_used"
    else:
        # Solve LP #1 as usual
        lambda_var = cp.Variable(N, nonneg=True)
        mu_var = cp.Variable(M, nonneg=True)

        Pz_const = cp.Constant(P_z)
        Pz_next_const = cp.Constant(P_z_next)
        Py_const = cp.Constant(P_y)

        def weighted_gram(Xc, w):
            return Xc.T @ cp.multiply(Xc, w[:, None])

        sum_PzPzT   = weighted_gram(Pz_const,  lambda_var)
        sum_PznPznT = weighted_gram(Pz_next_const, lambda_var)
        sum_PyPyT   = weighted_gram(Py_const,  mu_var)

        moment_match = sum_PzPzT - gamma * sum_PznPznT - sum_PyPyT

        # adaptive_regularization = dimension_scaled_regularization
        adaptive_regularization = dimension_scaled_regularization

        # Enhanced constraints for better numerical stability
        constraints = [
            cp.norm(moment_match, "fro") <= adaptive_regularization,  # Adaptive relaxed moment matching
            cp.sum(mu_var) == 1.0,
        ]

        I_d = cp.Constant(sparse.eye(d, format="csr"))
        C_approx = sum_PyPyT

        identity_penalty = cp.norm(C_approx - I_d, "fro")
        
        # Weighted objective: prioritize moment matching but ensure C is well-conditioned
        # objective = cp.Minimize(identity_penalty)
        moment_penalty = cp.norm(moment_match, "fro")
        # objective = cp.Minimize(moment_penalty + 0.1 * identity_penalty)
        objective = cp.Minimize(moment_penalty)


        prob = cp.Problem(objective, constraints)
        status1 = solve_robust(prob)

        if status1 not in ["optimal", "optimal_inaccurate"]:
            print(f"Stage 1 failed with status: {status1}")
            return None, None, {"stage1_status": status1, "stage2_status": "failed_stage1"}
        
        print(f"Stage 1 status: {status1}")
        mu = mu_var.value
        # print(f"mu: {mu}")

    # ----- LP #2: Solve for Q with c = vec(P_y^T diag(mu) P_y) -----
    Q_var = cp.Variable((d, d), symmetric=True)

    # Bellman constraints 
    cons = []
    for i in range(N):
        p = P_z[i]
        pn = P_z_next[i]
        Mi = np.outer(p, p) - gamma * np.outer(pn, pn)  
        # Use trace instead of sum(multiply) for better numerical stability
        cons.append(cp.trace(Q_var @ cp.Constant(Mi)) <= L_xu[i])
    
    # Objective: sum_i mu_i * <Q, y_i y_i^T>
    # Precompute the weighted sum of outer products: sum_i mu_i * (y_i @ y_i^T)
    weighted_outer_sum = np.zeros((d, d))
    for i in range(M):
        yi = P_y[i]
        weighted_outer_sum += mu[i] * np.outer(yi, yi)
    
    # Objective: trace(Q @ weighted_outer_sum)
    obj = cp.Maximize(cp.trace(Q_var @ cp.Constant(weighted_outer_sum)))
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

    Q_id = cp.Variable((d, d))  # identity baseline Q

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
        except Exception:
            pass
        if prob.status == "optimal_inaccurate":
            try:
                prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
            except Exception:
                pass
        if prob.status not in ("optimal",):
            try:
                prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
            except Exception:
                try:
                    prob.solve(solver=cp.ECOS, verbose=False, **ecos_opts)
                except Exception:
                    try:
                        prob.solve(solver=cp.CLARABEL, verbose=False)
                    except Exception:
                        prob.solve(verbose=False)
        return prob.status

    prob = cp.Problem(objective, cons)
    status = solve_robust_local(prob)

    print(f"Identity Q status: {status}")
    return status