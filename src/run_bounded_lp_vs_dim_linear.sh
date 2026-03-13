#!/bin/bash

# Script to run bounded_lp_vs_dim_linear.py with different N values
# Usage: ./run_bounded_lp_sweep.sh

# Array of N values to test
N_VALUES=(250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000)

# Change to the directory where the script is located (src directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Loop through each N value
for N in "${N_VALUES[@]}"; do
    echo "=========================================="
    echo "Running with N = $N"
    echo "=========================================="
    
    # Run the Python script with the current N value
    python bounded_lp_vs_dim_linear.py \
        --N "$N" \
        --seeds 10 \
        --M_offline 250 \
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed N = $N"
    else
        echo "✗ Failed for N = $N"
    fi
    
    echo ""
done

echo "=========================================="
echo "All runs completed!"
echo "=========================================="
