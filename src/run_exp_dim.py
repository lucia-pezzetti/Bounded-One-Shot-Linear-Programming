import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Experiment parameters
ns = list([2,3,4,5,6,7,8,9,10])  # n_x = 5, 10, 15, ..., 25
nu = 2  # n_u fixed
seeds = range(10)  # 10 different random seeds

gap_means = []
gap_stds = []
trace_means = []
trace_stds = []

# Path to the LQR script
script_path = "lqr_optimized.py"
data_dir = "../data"

# Ensure the script exists
if not os.path.isfile(script_path):
    raise FileNotFoundError(f"Could not find lqr.py at {script_path}")

# Regex to parse the printed optimality gap
GAP_PATTERN = r"Optimality gap \(E_Q_learned - E_Qstar\):\s*([\-\d\.eE]+)"
TRACE_PATTERN = r"Trace difference \(learned Q - optimal Q\):\s*([\-+\d\.eE]+)"

# Loop over state dimensions and seeds
for n in ns:
    gaps = []
    traces = []
    for seed in seeds:
        # Construct command to run the LQR script
        cmd = [
            sys.executable, script_path,
            "--n_x", str(n),
            "--n_u", str(nu),
            "--random_seed", str(seed),
            "--x_bounds", "-0.5", "0.5",
            "--u_bounds", "-3.0", "3.0",
            "--data_dir", data_dir
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error (n_x={n}, n_u={nu}, seed={seed}): {result.stderr}")
        gmatch = re.search(GAP_PATTERN, result.stdout)
        if not gmatch:
            raise ValueError(f"Could not parse optimality gap from output:\n{result.stdout}")
        print(f"n_x={n}, n_u={nu}, seed={seed} -> Gap: {gmatch.group(1)}")
        gaps.append(gmatch.group(1) if gmatch else np.nan)

        tmatch = re.search(TRACE_PATTERN, result.stdout)
        if not tmatch:
            raise ValueError(f"Could not parse trace difference from output:\n{result.stdout}")
        print(f"n_x={n}, n_u={nu}, seed={seed} -> Trace: {tmatch.group(1)}")
        traces.append(tmatch.group(1) if tmatch else np.nan)

    gaps = np.array(gaps, dtype=float)
    gap_means.append(gaps.mean() if gaps.size > 0 else np.nan)
    gap_stds.append(gaps.std(ddof=0) if gaps.size > 0 else np.nan)

    traces = np.array(traces, dtype=float)
    trace_means.append(traces.mean() if traces.size > 0 else np.nan)
    trace_stds.append(traces.std(ddof=0) if traces.size > 0 else np.nan)

# Plotting the results with error bars
plt.figure(figsize=(8, 5))
plt.errorbar(ns, gap_means, yerr=gap_stds, marker='o', linestyle='-')
plt.xlabel('State dimension $n_x$')
plt.ylabel('Optimality Gap')
plt.title('Optimality Gap vs State Dimension ($n_u = 2$)')
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/optimality_gap_vs_state_dimension.pdf", dpi=150)

plt.figure(figsize=(8, 5))
plt.errorbar(ns, trace_means, yerr=trace_stds, marker='o', linestyle='-')
plt.xlabel('State dimension $n_x$')
plt.ylabel('Trace Difference')
plt.title('Trace Difference vs State Dimension ($n_u = 2$)')
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/trace_difference_vs_state_dimension.pdf", dpi=150)