import cvxpy as cp
from scipy import sparse
import numpy as np

def solve_moment_matching_Q(P_z, P_z_next, P_y, L_xu, N, M, gamma, regularization=1e-4, fixed_mu=None):
    """
    Moment matching approach to learn Q matrix
    
    (LP #1) Solve for lambda, mu with moment-matching + C ≈ I objective.
    (LP #2) Given mu (so C), solve max c^T vec(Q) s.t. Bellman inequality.
    Returns: status_lp2 (string), and optionally statuses of LP #1 and shapes.
    
    Args:
        fixed_mu: If provided, skip LP #1 and use this fixed mu value for LP #2
    """
    d = P_z.shape[1]

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

        # Enhanced constraints for better numerical stability
        constraints = [
            cp.norm(moment_match, "fro") <= regularization,  # Relaxed moment matching
            # moment_match == 0,  # Too strict - can cause infeasibility
            # cp.sum(lambda_var) == 1,       # Ensure sufficient auxiliary data weight
            # mu_var <= 0.1,
            cp.sum(mu_var) == 1,
        ]

        I_d = cp.Constant(sparse.eye(d, format="csr"))
        C_approx = sum_PyPyT

        identity_penalty = cp.norm(C_approx - I_d, "fro")
        
        # Weighted objective: prioritize moment matching but ensure C is well-conditioned
        # objective = cp.Minimize(identity_penalty)
        moment_penalty = cp.norm(moment_match, "fro")
        objective = cp.Minimize(moment_penalty + 0.5 * identity_penalty)


        prob = cp.Problem(objective, constraints)
        status1 = None
        try:
            prob.solve(solver=cp.MOSEK, verbose=False, 
                      mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6,
                                   'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-6,
                                   'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-6,
                                   'MSK_IPAR_INTPNT_BASIS': 1})
            status1 = prob.status
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, verbose=False, 
                          abstol=1e-6, reltol=1e-6, max_iters=2000)
                status1 = prob.status
            except Exception:
                try:
                    prob.solve(solver=cp.SCS, verbose=False, 
                              eps=1e-6, max_iters=2000)
                    status1 = prob.status
                except Exception:
                    try:
                        prob.solve(solver=cp.CLARABEL, verbose=False)
                        status1 = prob.status
                    except Exception:
                        status1 = "solver_error"

        if status1 not in ["optimal", "optimal_inaccurate"]:
            print(f"Stage 1 failed with status: {status1}")
            return None, None, {"stage1_status": status1, "stage2_status": "failed_stage1"}

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
        cons.append(cp.sum(cp.multiply(Q_var, Mi)) <= L_xu[i])

    # Objective: sum_i mu_i * <Q, y_i y_i^T> (same as lqr_optimized.py)
    terms = []
    for i in range(M):
        yi = P_y[i]
        terms.append(mu[i] * cp.sum(cp.multiply(Q_var, np.outer(yi, yi))))

    obj = cp.Maximize(cp.sum(terms))
    prob_lp = cp.Problem(obj, cons)

    status2 = None
    try:
        prob_lp.solve(solver=cp.MOSEK, verbose=False,
                     mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6,
                                  'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-6,
                                  'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-6,
                                  'MSK_IPAR_INTPNT_BASIS': 1})
        status2 = prob_lp.status
    except Exception:
        try:
            prob_lp.solve(solver=cp.ECOS, verbose=False,
                         abstol=1e-6, reltol=1e-6, max_iters=2000)
            status2 = prob_lp.status
        except Exception:
            try:
                prob_lp.solve(solver=cp.SCS, verbose=False,
                             eps=1e-6, max_iters=2000)
                status2 = prob_lp.status
            except Exception:
                try:
                    prob_lp.solve(solver=cp.CLARABEL, verbose=False)
                    status2 = prob_lp.status
                except Exception:
                    status2 = "solver_error"

    Q_learned_mat = Q_var.value
    if Q_learned_mat is None:
        print(f"Stage 2 failed with status: {status2}")
        return None, None, {"stage1_status": status1, "stage2_status": status2}
    
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
    C_matrix[:4, :4] = np.eye(4)  # Identity for state dimensions
    C_matrix[4:, 4:] = 0.8 * np.eye(1)  # 0.8 * Identity for input dimensions

    cons = []
    for i in range(N):
        p = P_z[i]
        pn = P_z_next[i]
        Mi = np.outer(p, p) - gamma * np.outer(pn, pn)
        cons.append(cp.sum(cp.multiply(Q_id, Mi)) <= L_xu[i])

    objective = cp.Maximize(cp.trace(cp.Constant(C_matrix) @ Q_id))

    prob = cp.Problem(objective, cons)
    status = None
    try:
        prob.solve(solver=cp.MOSEK, verbose=False,
                  mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6,
                               'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-6,
                               'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-6})
        status = prob.status
    except Exception:
        try:
            prob.solve(solver=cp.ECOS, verbose=False,
                      abstol=1e-6, reltol=1e-6, max_iters=1000)
            status = prob.status
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=False,
                          eps=1e-6, max_iters=1000)
                status = prob.status
            except Exception:
                try:
                    prob.solve(solver=cp.CLARABEL, verbose=False,
                              eps=1e-6, max_iters=1000)
                    status = prob.status
                except Exception:
                    # If all solvers fail, return a failure status
                    status = "solver_error"

    print(f"Identity Q status: {status}")
    return status