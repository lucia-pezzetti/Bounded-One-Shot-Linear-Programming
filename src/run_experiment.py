#!/usr/bin/env python3
"""
experiment.py

Runs the LQR learning routine in lqr.py across different numbers
of main samples (N), auxiliary samples (M), and random seeds, then
plots the mean optimality gap (E_Q* - E_Q̂ learned) with 95% confidence
intervals (over seeds only).

Usage:
    python experiment.py

Ensure that lqr.py and dynamical_systems.py are in the same directory.
"""
import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Experiment parameters
N_list = [50, 100, 200, 500, 1000, 2000]      # main sample sizes
M_list = [10, 20, 50, 100, 500]                  # auxiliary sample sizes
seeds = list(range(10))                    # random seeds for repetition

# Regex to parse the printed optimality gap
GAP_PATTERN = r"Optimality gap \(E_Qstar - E_Q_learned\):\s*([\-\d\.eE]+)"


def run_experiment(N, M, seed):
    """
    Runs lqr.py with given parameters and returns the parsed optimality gap.
    """
    cmd = [
        "python", "lqr.py",
        "--n_samples", str(N),
        "--m_samples", str(M),
        "--random_seed", str(seed)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error (N={N}, M={M}, seed={seed}): {result.stderr}")
    match = re.search(GAP_PATTERN, result.stdout)
    if not match:
        raise ValueError(f"Could not parse optimality gap from output:\n{result.stdout}")
    return float(match.group(1))

# Collect raw results in a DataFrame
records = []
for N in N_list:
    for M in M_list:
        for seed in seeds:
            print(f"Running experiment with N={N}, M={M}, seed={seed}...")
            gap = run_experiment(N, M, seed)
            print(f"Gap: {gap}")
            records.append({'N': N, 'M': M, 'seed': seed, 'gap': gap})

df = pd.DataFrame(records)

# Group by (N, M), compute mean and std over seeds
summary = df.groupby(['N', 'M'])['gap'].agg(['mean', 'std']).reset_index()

# Pivot for plotting
mean_pivot = summary.pivot(index='N', columns='M', values='mean')
std_pivot = summary.pivot(index='N', columns='M', values='std')

# Plotting with shaded ±1 std region
plt.figure()
for M in M_list:
    means = mean_pivot[M].values
    std = std_pivot[M].values
    plt.errorbar(
        N_list, means, yerr=std,
        capsize=5, marker='o', linestyle='-'
    , label=f"Aux samples = {M}")

plt.xscale('log')
plt.xlabel('Number of constraints N (log scale)')
plt.ylabel('Optimality gap')
plt.title('Effect of Sample Sizes on Solution Quality')
plt.legend(title='Auxiliary samples M')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

# Show or save
plt.savefig("../figures/lqr_experiment_results.pdf", dpi=150, bbox_inches='tight')