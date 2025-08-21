
from random import seed
import numpy as np
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from dynamical_systems import cart_pole

# -------------------------
# Config (edit if needed)
# -------------------------
M_c = 4.0     # cart mass (kg)
M_p = 2.0     # pole mass (kg)
l = 1.0       # half-pole length (m)
dt = 0.001    # sampling time (s)
gamma = 0.99  # discount factor

degree = 2         # polynomial feature degree (same as your script)

# State/action sampling bounds (same as your script)
x_bounds = (-3.0, 3.0)
x_dot_bounds = (-3.0, 3.0)
theta_bounds = (-0.5, 0.5)
theta_dot_bounds = (-1.0, 1.0)
u_bounds = (-10.0, 10.0)

def solve_moment_matching_Q(system, x, u, x_plus, u_plus, N, M, seed):
    """
    Reproduces your two-stage approach:
    (LP #1) Solve for lambda, mu with moment-matching + C ≈ I objective.
    (LP #2) Given mu (so C), solve max c^T vec(Q) s.t. Bellman inequality.
    Returns: status_lp2 (string), and optionally statuses of LP #1 and shapes.
    """
    # Seed all RNGs used here
    np.random.seed(seed)

    # ----- Build feature maps -----
    z = np.concatenate([x, u], axis=1)           # (N, dx+du)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)  # (N, dx+du)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler(with_mean=True, with_std=True)

    Z_all = np.concatenate([z, z_plus], axis=0)       # (2N, dx+du)
    P_all = poly.fit_transform(Z_all)                  # (2N, d)
    P_all = scaler.fit_transform(P_all)                # scaled
    d = P_all.shape[1]

    P_z = P_all[:N]
    P_z_next = P_all[N:]

    # Auxiliary sample for Py
    x_aux, u_aux = system.generate_samples_auxiliary(
        x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=M
    )
    y = np.concatenate([x_aux, u_aux], axis=1)  # (M, dx+du)
    P_y = poly.transform(y)
    P_y = scaler.transform(P_y)

    # ----- LP #1: Moment matching (lambda, mu) -----
    lambda_var = cp.Variable(N, nonneg=True)
    mu_var = cp.Variable(M, nonneg=True)

    Pz_const = cp.Constant(P_z)
    Pz_next_const = cp.Constant(P_z_next)
    Py_const = cp.Constant(P_y)

    sum_PzPzT  = Pz_const.T @ cp.diag(lambda_var) @ Pz_const
    sum_PznPznT = Pz_next_const.T @ cp.diag(lambda_var) @ Pz_next_const
    sum_PyPyT  = Py_const.T @ cp.diag(mu_var) @ Py_const

    moment_match = sum_PzPzT - gamma * sum_PznPznT - sum_PyPyT

    constraints = [
        moment_match == np.zeros((d, d)),
        cp.sum(lambda_var) == 1,     # break scale invariance
    ]
    I_d = np.eye(d)
    objective = cp.Minimize(cp.norm(sum_PyPyT - I_d, "fro"))

    prob = cp.Problem(objective, constraints)
    status1 = None
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        status1 = prob.status
    except Exception:
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            status1 = prob.status
        except Exception:
            prob.solve(verbose=False)
            status1 = prob.status

    if status1 not in ["optimal", "optimal_inaccurate"]:
        # If LP #1 fails, LP #2 cannot be formed meaningfully -> treat as "unbounded" downstream.
        return "failed_stage1", status1

    lam = lambda_var.value
    mu = mu_var.value

    # ----- LP #2: Solve for Q with c = vec(P_y^T diag(mu) P_y) -----
    C_val = P_y.T @ np.diag(mu) @ P_y
    c_vec = C_val.flatten(order="C")

    # Build F and l for Bellman inequality
    F_3D = (P_z[:, :, None] * P_z[:, None, :]) - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])
    F_mat = F_3D.reshape((N, d * d), order="C")

    # Quadratic stage costs l(x,u)
    L_xu = system.cost(x, u)
    L_xu_const = cp.Constant(L_xu)

    Q_var = cp.Variable((d, d))  # matrix variable
    Q_var_vec = cp.reshape(Q_var, (d * d,), order="C")

    constraints_lp = [F_mat @ Q_var_vec <= L_xu_const]
    objective_lp = cp.Maximize(c_vec @ Q_var_vec)

    prob_lp = cp.Problem(objective_lp, constraints_lp)
    status2 = None
    try:
        prob_lp.solve(solver=cp.MOSEK, verbose=False)
        status2 = prob_lp.status
    except Exception:
        try:
            prob_lp.solve(solver=cp.ECOS, verbose=False)
            status2 = prob_lp.status
        except Exception:
            prob_lp.solve(verbose=False)
            status2 = prob_lp.status

    return status2, status1

def solve_identity_Q(system, x, u, x_plus, u_plus, N, seed):
    """
    Classical baseline: skip moment matching, solve
       min trace(Q)  s.t.  F vec(Q) <= l(x,u).
    Returns status string.
    """
    np.random.seed(seed)

    # Features
    z = np.concatenate([x, u], axis=1)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    scaler = StandardScaler(with_mean=True, with_std=True)

    Z_all = np.concatenate([z, z_plus], axis=0)
    P_all = poly.fit_transform(Z_all)
    P_all = scaler.fit_transform(P_all)
    d = P_all.shape[1]

    P_z = P_all[:N]
    P_z_next = P_all[N:]

    # Build F and l
    F_3D = (P_z[:, :, None] * P_z[:, None, :]) - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])
    F_mat = F_3D.reshape((N, d * d), order="C")
    L_xu = system.cost(x, u)

    Q_id = cp.Variable((d, d))  # identity baseline Q
    Q_id_vec = cp.reshape(Q_id, (d * d,), order="C")

    constraints = [F_mat @ Q_id_vec <= cp.Constant(L_xu)]
    objective = cp.Minimize(cp.trace(Q_id))

    prob = cp.Problem(objective, constraints)
    status = None
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        status = prob.status
    except Exception:
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            status = prob.status
        except Exception:
            prob.solve(verbose=False)
            status = prob.status

    return status

def run_one(seed, N, M_offline=50):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    """Generate data with given seed and N, then solve both LPs and return boundedness flags."""
    # Consistent seeding across NumPy (and torch if your system uses it internally)
    np.random.seed(seed)
    # Build system and data
    system = cart_pole(M_c, M_p, l, dt, gamma, N, M_offline)

    x, u, x_plus, u_plus = system.generate_samples(
        x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=N
    )

    # Moment-matching
    status_m2, status_m1 = solve_moment_matching_Q(system, x, u, x_plus, u_plus, N, M_offline, seed=seed)

    # Identity baseline
    status_id = solve_identity_Q(system, x, u, x_plus, u_plus, N, seed=seed)

    # Interpret boundedness:
    # "optimal"/"optimal_inaccurate" -> bounded & feasible
    # "unbounded" -> unbounded
    # anything else (infeasible, error) -> treat as NOT bounded
    def is_bounded(status):
        return status in ("optimal", "optimal_inaccurate")

    return {
        "seed": seed,
        "N": N,
        "moment_matching_status": status_m2,
        "moment_matching_bounded": is_bounded(status_m2),
        "moment_stage1_status": status_m1,
        "identity_status": status_id,
        "identity_bounded": is_bounded(status_id),
    }

def sweep(N_values=range(50, 501, 50), seeds=range(10), M_offline=50):
    results = []
    for N in N_values:
        for s in seeds:
            results.append(run_one(seed=int(s), N=int(N), M_offline=M_offline))
    return results

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Sweep boundedness of LPs vs N.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per N (default: 10)")
    parser.add_argument("--Nmin", type=int, default=50, help="Min N (default: 50)")
    parser.add_argument("--Nmax", type=int, default=500, help="Max N (default: 500)")
    parser.add_argument("--Nstep", type=int, default=50, help="Step for N (default: 50)")
    parser.add_argument("--out_json", type=str, default="../results/test.json", help="Where to save raw results JSON")
    parser.add_argument("--plot_png", type=str, default="../figures/test.png", help="Where to save the plot")
    parser.add_argument("--M_offline", type=int, default=200, help="Offline pool size (default: 200)")
    args = parser.parse_args()

    N_values = list(range(args.Nmin, args.Nmax + 1, args.Nstep))
    seeds = list(range(args.seeds))

    try:
        results = sweep(N_values=N_values, seeds=seeds, M_offline=args.M_offline)
    except Exception as e:
        # If the user's environment is missing dependencies (e.g., MOSEK, dynamical_systems),
        # we surface a helpful error message instead of crashing silently.
        import sys, traceback
        print("ERROR while running the sweep:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

    # Aggregate
    import pandas as pd
    df = pd.DataFrame(results)
    agg = df.groupby("N").agg(
        moment_bounded_pct=("moment_matching_bounded", "mean"),
        identity_bounded_pct=("identity_bounded", "mean"),
    ).reset_index()

    # Convert to percentages
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_pct"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_pct"]

    # Save raw + aggregated
    df.to_json(args.out_json, orient="records", indent=2)
    agg.to_json("bounded_lp_percentages.json", orient="records", indent=2)

    # Plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(agg["N"], agg["moment_bounded_pct"], marker="o", label="Moment matching")
    plt.plot(agg["N"], agg["identity_bounded_pct"], marker="s", label="Identity covariance")
    plt.xlabel("N (number of samples)")
    plt.ylabel("Bounded LPs (%)")
    plt.title("Percentage of bounded LP problems vs N")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_png, dpi=150)
    print("Saved:", args.out_json, "and", args.plot_png)
