
import json
import numpy as np
import cvxpy as cp
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# -------------------------
# Config (defaults; can be overridden by CLI)
# -------------------------
degree = 2           # polynomial feature degree
M_offline = 1000      # auxiliary pool size for building P_y

# Sampling bounds for generating (x,u) datasets
# (Assumed symmetric for all dimensions)
x_bound = 3.0        # states sampled in [-x_bound, x_bound]
u_bound = 1.0        # inputs sampled in [-u_bound, u_bound]

du = 2               # fixed input dimension

def load_system_json(data_dir: Path, dx: int):
    """Load A, B, C from ../data/dx_%d.json"""
    fname = data_dir / f"dx_{dx}.json"
    with open(fname, "r") as f:
        data = json.load(f)
    A = np.array(data["A"], dtype=float)
    B = np.array(data["B"], dtype=float)
    C = np.array(data["C"], dtype=float)
    # Basic checks
    if A.shape != (dx, dx):
        raise ValueError(f"A shape mismatch for dx={dx}: got {A.shape}, expected {(dx, dx)}")
    if B.shape[0] != dx or B.shape[1] != du:
        raise ValueError(f"B shape mismatch for dx={dx}, du={du}: got {B.shape}, expected {(dx, du)}")
    if C.shape != (dx, dx):
        raise ValueError(f"C shape mismatch for dx={dx}: got {C.shape}, expected {(dx, dx)}")
    return A, B, C

def stage_cost(x: np.ndarray, u: np.ndarray, C: np.ndarray):
    """
    Quadratic cost: l(x,u) = x^T C x + u^T R u, with R = I_du.
    x: (N, dx), u: (N, du) -> returns (N,)
    """
    R = np.eye(u.shape[1])
    # x^T C x
    xCx = np.einsum("ni,ij,nj->n", x, C, x, optimize=True)
    # u^T R u = ||u||^2
    uRu = np.einsum("ni,ij,nj->n", u, R, u, optimize=True)
    return xCx + uRu

def generate_dataset(A, B, N, dx, du, seed=0):
    """
    Generate random (x,u) and compute x_next = A x + B u (no noise).
    Returns x, u, x_plus, u_plus (N samples).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-x_bound, x_bound, size=(N, dx))
    u = rng.uniform(-u_bound, u_bound, size=(N, du))
    x_plus = (A @ x.T + B @ u.T).T
    # Use u_plus = u (consistent dimensionality for feature construction)
    u_plus = u.copy()
    return x, u, x_plus, u_plus

def auxiliary_samples(M, dx, du, seed=0):
    """Auxiliary pool for building P_y"""
    rng = np.random.default_rng(seed + 12345)
    x_aux = rng.uniform(-x_bound, x_bound, size=(M, dx))
    u_aux = rng.uniform(-u_bound, u_bound, size=(M, du))
    return x_aux, u_aux

def solve_moment_matching_Q(x, u, x_plus, u_plus, C_cost, gamma, N, M, degree, seed):
    """
    Two-stage approach for general linear systems (features on [x,u]).
    Returns status strings for stage 2 and stage 1.
    """
    np.random.seed(seed)

    dx = x.shape[1]
    du = u.shape[1]

    # Build features
    z = np.concatenate([x, u], axis=1)                 # (N, dx+du)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)  # (N, dx+du)

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    scaler = StandardScaler(with_mean=True, with_std=True)

    Z_all = np.concatenate([z, z_plus], axis=0)  # (2N, dx+du)
    P_all = poly.fit_transform(Z_all)
    P_all = scaler.fit_transform(P_all)
    d = P_all.shape[1]

    P_z = P_all[:N]
    P_z_next = P_all[N:]

    # Auxiliary for P_y
    x_aux, u_aux = auxiliary_samples(M, dx, du, seed=seed)
    y = np.concatenate([x_aux, u_aux], axis=1)
    P_y = scaler.transform(poly.transform(y))

    # LP #1
    lambda_var = cp.Variable(N, nonneg=True)
    mu_var = cp.Variable(M, nonneg=True)

    Pz_const = cp.Constant(P_z)
    Pz_next_const = cp.Constant(P_z_next)
    Py_const = cp.Constant(P_y)

    sum_PzPzT   = Pz_const.T @ cp.diag(lambda_var) @ Pz_const
    sum_PznPznT = Pz_next_const.T @ cp.diag(lambda_var) @ Pz_next_const
    sum_PyPyT   = Py_const.T @ cp.diag(mu_var) @ Py_const

    moment_match = sum_PzPzT - gamma * sum_PznPznT - sum_PyPyT

    constraints = [
        moment_match == np.zeros((d, d)),
        cp.sum(lambda_var) == 1,
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
        return "failed_stage1", status1

    mu = mu_var.value

    # LP #2
    C_val = P_y.T @ np.diag(mu) @ P_y
    c_vec = C_val.flatten(order="C")

    F_3D = (P_z[:, :, None] * P_z[:, None, :]) - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])
    F_mat = F_3D.reshape((N, d * d), order="C")

    L_xu = stage_cost(x, u, C_cost)

    Q_var = cp.Variable((d, d), PSD=True)
    Q_var_vec = cp.reshape(Q_var, (d * d,), order="C")

    constraints_lp = [F_mat @ Q_var_vec <= cp.Constant(L_xu)]
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

def solve_identity_Q(x, u, x_plus, u_plus, C_cost, gamma, N, degree, seed):
    """Baseline: min trace(Q) s.t. F vec(Q) <= l(x,u)."""
    np.random.seed(seed)

    dx = x.shape[1]
    du = u.shape[1]

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

    F_3D = (P_z[:, :, None] * P_z[:, None, :]) - gamma * (P_z_next[:, :, None] * P_z_next[:, None, :])
    F_mat = F_3D.reshape((N, d * d), order="C")
    L_xu = stage_cost(x, u, C_cost)

    Q_id = cp.Variable((d, d))
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

def run_one(seed, dx, A, B, C_cost, N, gamma):
    """Generate a dataset and solve both LPs; return boundedness flags."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    x, u, x_plus, u_plus = generate_dataset(A, B, N=N, dx=dx, du=B.shape[1], seed=seed)

    status_m2, status_m1 = solve_moment_matching_Q(x, u, x_plus, u_plus, C_cost, gamma, N, M_offline, degree, seed)
    status_id = solve_identity_Q(x, u, x_plus, u_plus, C_cost, gamma, N, degree, seed)

    def is_bounded(status):
        return status in ("optimal", "optimal_inaccurate")

    return {
        "seed": seed,
        "dx": dx,
        "moment_matching_status": status_m2,
        "moment_matching_bounded": is_bounded(status_m2),
        "moment_stage1_status": status_m1,
        "identity_status": status_id,
        "identity_bounded": is_bounded(status_id),
    }

def sweep_over_dims(dims, data_dir: Path, seeds, N, gamma):
    results = []
    for dx in dims:
        path = data_dir / f"dx_{dx}.json"
        if not path.exists():
            print(f"[WARN] Missing file: {path}. Skipping dx={dx}.")
            continue
        try:
            A, B, C_cost = load_system_json(data_dir, dx)
        except Exception as e:
            print(f"[WARN] Failed to load/validate for dx={dx}: {e}. Skipping.")
            continue
        for s in seeds:
            results.append(run_one(seed=int(s), dx=int(dx), A=A, B=B, C_cost=C_cost, N=N, gamma=gamma))
    return results

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Boundedness vs state dimension (linear systems).")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory containing dx_%n.json files")
    parser.add_argument("--dims", type=str, default="", help="Comma-separated list of state dimensions (e.g., '4,8,12'). If empty, use --dmin.. arguments.")
    parser.add_argument("--dmin", type=int, default=4, help="Minimum dx (used if --dims is empty)")
    parser.add_argument("--dmax", type=int, default=40, help="Maximum dx (used if --dims is empty)")
    parser.add_argument("--dstep", type=int, default=4, help="Step for dx (used if --dims is empty)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per dimension (default: 10)")
    parser.add_argument("--N", type=int, default=200, help="Number of samples for dataset (default: 200)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--out_json", type=str, default="bounded_vs_dim_results.json", help="Raw results JSON")
    parser.add_argument("--plot_png", type=str, default="bounded_vs_dim_percentages.png", help="Plot filename")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.dims.strip():
        dims = [int(s) for s in args.dims.split(",") if s.strip()]
    else:
        dims = list(range(args.dmin, args.dmax + 1, args.dstep))
    seeds = list(range(args.seeds))

    results = sweep_over_dims(dims, data_dir, seeds, N=args.N, gamma=args.gamma)

    if len(results) == 0:
        print("No results (no valid dimensions or files). Exiting.")
        raise SystemExit(0)

    df = pd.DataFrame(results)
    agg = df.groupby("dx").agg(
        moment_bounded_mean=("moment_matching_bounded", "mean"),
        identity_bounded_mean=("identity_bounded", "mean"),
        n=("seed", "count"),
    ).reset_index()

    # Convert to percentages
    agg["moment_bounded_pct"] = 100 * agg["moment_bounded_mean"]
    agg["identity_bounded_pct"] = 100 * agg["identity_bounded_mean"]

    # Save raw + aggregated
    df.to_json(args.out_json, orient="records", indent=2)
    agg.to_json("bounded_vs_dim_percentages.json", orient="records", indent=2)
    
    # Print compact table
    cols = ["dx", "n",
            "moment_bounded_pct",
            "identity_bounded_pct"]
    print(agg[cols].to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

    # Plot mean ± 1 SD
    plt.figure()
    plt.errorbar(agg["dx"], agg["moment_bounded_pct"],
                 marker="o", capsize=3, label="Moment matching")
    plt.errorbar(agg["dx"], agg["identity_bounded_pct"],
                 marker="s", capsize=3, label="Identity covariance")
    plt.xlabel("State dimension (dx)")
    plt.ylabel("Bounded LPs (%)")
    plt.title("Percentage of bounded LP problems vs state dimension")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_png, dpi=150)
    print("Saved:", args.out_json, "bounded_vs_dim_percentages_deg2.json and", args.plot_png)
