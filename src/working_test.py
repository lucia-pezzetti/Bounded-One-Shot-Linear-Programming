import numpy as np
import cvxpy as cp
from control.matlab import rss

def p(z):
    x, u = z
    # Flatten x and u (if they are multidimensional) and concatenate them
    return np.concatenate([np.ravel(x), np.ravel(u)])

def outer_p(z):
    v = p(z)
    return np.outer(v, v)

def solve_riccati(A, B, gamma, L, tol=1e-6, max_iter=1000):
    P = np.eye(A.shape[0])  # Initialize P as an identity matrix of appropriate size
    for _ in range(max_iter):
        P_next = L[:A.shape[0], :A.shape[0]] + gamma * A.T @ P @ A - \
                (L[:A.shape[0], A.shape[0]:] + gamma * A.T @ P @ B) @ \
                np.linalg.inv(L[A.shape[0]:, A.shape[0]:] + gamma * B.T @ P @ B) @ \
                (L[A.shape[0]:, :A.shape[0]] + gamma * B.T @ P @ A)
        if np.linalg.norm(P_next - P, ord='fro') < tol:
            return P_next
        P = P_next
    raise ValueError("Riccati equation did not converge within the maximum number of iterations")

def compute_analytical_Q_matrix(A, B, gamma, L, basis_func=None, grid_size=1000):
    assert basis_func is not None, "Pass your basis function p(z) as `basis_func`"

    # Step 1: Solve Riccati equation numerically
    P = solve_riccati(A, B, gamma, L)

    # Step 2: Define Q(x, u)
    def Q_true_fn(x, u):
        """
        Compute the true Q-function for multidimensional states and inputs.
        """
        x = np.ravel(x)  # Ensure x is a 1D array
        u = np.ravel(u)  # Ensure u is a 1D array
        x_u = np.concatenate([x, u])  # Combine state and input into a single vector

        # Compute the next state
        x_next = A @ x + B @ u

        # Compute the quadratic cost
        return x_u.T @ L @ x_u + gamma * (x_next.T @ P @ x_next)
    
    # Step 3: Compute Q matrix
    Q_matrix = L + gamma * np.block([[A, B]]).T @ P @ np.block([[A, B]])
    return Q_true_fn, Q_matrix


def is_stable(Q_true, Q_learned, tol=1e-2):
    """Check closeness of two Q-functions over a sample of points."""
    x_vals = np.linspace(-10, 10, 20)
    u_vals = np.linspace(-1, 1, 10)
    errors = []

    for x in x_vals:
        for u in u_vals:
            err = abs(Q_true(x, u) - Q_learned(x, u))
            errors.append(err)

    print('avg_error:', np.mean(errors))
    
    return np.mean(errors) < tol, np.mean(errors)

def run_stability_test(num_trials=10, gamma_list=[0.7, 0.8, 0.9, 0.99]):
    results = []

    for gamma in gamma_list:
        for _ in range(num_trials):
            # Step 1: Generate a random stable state-space system
            # rss generates a random stable system with n states, m inputs, and p outputs
            np.random.seed(2025)
            
            n_states = 1  # Number of states
            m_inputs = 1  # Number of inputs
            p_outputs = 1  # Number of outputs
            sys = rss(n_states, p_outputs, m_inputs)

            # Extract A and B matrices
            A = sys.A
            B = sys.B


            # Step 2: Cost matrix          
            L = np.block([
                [np.eye(n_states), np.zeros((n_states, m_inputs))],
                [np.zeros((m_inputs, n_states)), np.eye(m_inputs)]
            ])

            # Step 3: Solve Riccati and get Q_true
            try:
                Q_true_fn, Q_true_mat = compute_analytical_Q_matrix(A, B, gamma, L, basis_func=p)
            except Exception as e:
                print("Riccati failed:", e)
                continue

            # Step 4: Run your LP code (reuse core of your script but adapt A, B, gamma, L)

            # Modify sampled data
            N = 500
            X = np.random.uniform(Xmin, Xmax, size=(N, n_states))  # Shape (N, n_states)
            U = np.random.uniform(Umin, Umax, size=(N, m_inputs))  # Shape (N, m_inputs)
            X_next = np.clip((A @ X.T + B @ U.T).T, Xmin, Xmax)  # Ensure proper matrix multiplication
            W = np.random.uniform(Umin, Umax, size=N)
            l = np.array([np.hstack([x, u]).T @ L @ np.hstack([x, u]) for x, u in zip(X, U)])

            # dimensions
            dim_p = len(p([X[0], U[0]]))  # = 6
            dim_vecQ = dim_p ** 2  # = 36       

            # Sample test points
            M = N
            x_vals = np.random.uniform(Xmin, Xmax, size=M)
            u_vals = np.random.uniform(Umin, Umax, size=M)
            Y = np.stack([x_vals, u_vals], axis=1)

            λ = cp.Variable(N, nonneg=True)
            μ = cp.Variable(M, nonneg=True)

            C_lambda_expr = sum([
                λ[i] * (outer_p([X[i], U[i]]) - gamma * outer_p([X_next[i], W[i]]))
                for i in range(N)
            ])

            dict_tensors = np.stack([outer_p(y) for y in Y], axis=2)
            C_mu_expr = sum([
                μ[i] * dict_tensors[:, :, i]
                for i in range(M)
            ])

            residual_matrix = C_lambda_expr - C_mu_expr
            objective = cp.Minimize(cp.norm(residual_matrix, "fro"))
            constraints = [λ >= 0, μ >= 0, cp.sum(λ) >= 1.0]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)

            C_matrix = sum(μ.value[i] * outer_p(Y[i]) for i in range(M))
            vec_C = C_matrix.reshape(-1)
            α = cp.Variable(dim_vecQ)
            constraints_lp = [
                α @ (outer_p([X[i], U[i]]).reshape(-1) - gamma * outer_p([X_next[i], W[i]]).reshape(-1)) <= l[i]
                for i in range(N)
            ]
            objective = cp.Maximize(α @ vec_C)
            lp = cp.Problem(objective, constraints_lp)
            lp.solve(solver=cp.ECOS, verbose=False)

            if α.value is None or np.any(np.isnan(α.value)) or np.any(np.isinf(α.value)):
                print("Invalid solution from LP.")
                continue


            Q_vec = α.value
            if Q_vec is None:
                print("LP failed to solve.")
                continue
            Q_mat = Q_vec.reshape((dim_p, dim_p))
            def Q_learned_fn(x, u):
                v = p([x, u])
                return v.T @ Q_mat @ v

            # Step 5: Compare Q_true vs Q_learned
            close, avg_err = is_stable(Q_true_fn, Q_learned_fn)
            results.append({
                "A": A, "B": B, "gamma": gamma,
                "stable": close, "avg_error": avg_err
            })

    return results


# state space
Xmin = -10
Xmax = 10
# action space
Umin = -1
Umax = 1

results = run_stability_test()
import pandas as pd
df = pd.DataFrame(results)
print(df.groupby("gamma")[["stable", "avg_error"]].agg(["mean", "std"]))