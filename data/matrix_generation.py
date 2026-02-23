import random
import json
import numpy as np
import os
GAMMA = 0.99

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_system(n, seed):
    """Generate a random system (A, B, C) for dimension n with given seed."""
    random.seed(seed)
    np.random.seed(seed)
    
    # A: diagonal 0.5, off-diagonals uniform in [-0.1, 0.1] with prob 0.9, 0.0 with prob 0.1
    A = [
        [0.5 if i == j else (random.uniform(-0.1, 0.1) if random.random() < 0.9 else 0.0) for j in range(n)]
        for i in range(n)
    ]
    # B: n x 2, uniform in [-0.1, 0.1] (2 inputs needed for controllability in larger dims)
    B = [[random.uniform(-0.1, 0.1) for _ in range(2)] for _ in range(n)]
    # C: identity matrix of size n
    C = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    return A, B, C

def is_controllable(A, B, tol=1e-10):
    """
    Check if the system (A, B) is controllable.
    A system is controllable if the controllability matrix [B, AB, A²B, ..., A^(n-1)B]
    has full row rank (rank n).
    """
    A = np.sqrt(GAMMA) * np.array(A)
    B = np.sqrt(GAMMA) * np.array(B)
    n = A.shape[0]
    
    # Build controllability matrix
    controllability_matrix = B.copy()
    AB = B.copy()
    for _ in range(n - 1):
        AB = A @ AB
        controllability_matrix = np.hstack([controllability_matrix, AB])
    
    # Check rank
    rank = np.linalg.matrix_rank(controllability_matrix, tol=tol)
    return rank == n

def generate_controllable_system(n, initial_seed, max_attempts=10000):
    """
    Generate a controllable system for dimension n.
    Keeps resampling with incremented seeds until a controllable system is found.
    Returns (A, B, C, final_seed, num_attempts).
    """
    seed = initial_seed
    for attempt in range(max_attempts):
        A, B, C = generate_system(n, seed)
        if is_controllable(A, B):
            return A, B, C, seed, attempt + 1
        seed += 1
    
    raise RuntimeError(f"Could not find controllable system for n={n} after {max_attempts} attempts")

# Generate and save matrices for n = 2, 3, ..., 29
# 50 different controllable systems per dimension
num_systems_per_dim = 50
created_files = []
stats = {}

print("Generating controllable systems...")
for n in range(2, 31, 1):
    print(f"\nDimension n={n}:")
    systems = []
    total_attempts = 0
    
    for system_idx in range(num_systems_per_dim):
        # Use different initial seed for each system
        initial_seed = n * 1 + system_idx * 100
        A, B, C, final_seed, attempts = generate_controllable_system(n, initial_seed)
        systems.append({
            "A": A,
            "B": B,
            "C": C,
            "seed": final_seed
        })
        total_attempts += attempts
    
    # Save all systems for this dimension to a single JSON file
    filename = os.path.join(SCRIPT_DIR, f"dx_{n}_du_2_systems.json")
    with open(filename, "w") as f:
        json.dump({
            "dimension": n,
            "num_systems": num_systems_per_dim,
            "systems": systems
        }, f, indent=4)
    created_files.append(filename)
    
    avg_attempts = total_attempts / num_systems_per_dim
    stats[n] = {"total_attempts": total_attempts, "avg_attempts": avg_attempts}
    print(f"  Created {num_systems_per_dim} controllable systems (avg {avg_attempts:.2f} attempts per system)")

# Display summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
existing_files = [f for f in created_files if os.path.exists(f)]
print(f"Created {len(existing_files)} JSON files:")
for f in existing_files:
    print(f"  - {f}")

print("\nControllability statistics:")
for n, s in stats.items():
    print(f"  n={n}: avg {s['avg_attempts']:.2f} attempts per controllable system")
