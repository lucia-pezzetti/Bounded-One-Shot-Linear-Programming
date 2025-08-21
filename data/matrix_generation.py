import random
import json

# Generate and save matrices for n = 15, 25
created_files = []
for n in range(15, 26, 10):
    # A: diagonal 0.5, off-diagonals uniform in [-0.1, 0.1]
    A = [
        [0.5 if i == j else random.uniform(-0.1, 0.1) for j in range(n)]
        for i in range(n)
    ]
    # B: n x 2, uniform in [-0.1, 0.1]
    B = [[random.uniform(-0.1, 0.1) for _ in range(2)] for _ in range(n)]
    # C: identity matrix of size n
    C = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Save to JSON file
    filename = f"dx_{n}.json"
    with open(filename, "w") as f:
        json.dump({"A": A, "B": B, "C": C}, f, indent=4)
    created_files.append(filename)

# Display the created JSON files
import os
existing_files = [f for f in created_files if os.path.exists(f)]
print("Created JSON files:", existing_files)
