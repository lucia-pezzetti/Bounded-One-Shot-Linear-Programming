#!/bin/bash

# Script to run bounded_lp_vs_dim_linear.py with different N values
# Usage: ./run_bounded_lp_sweep.sh

# Array of N values to test
N_VALUES=(250 500 750 1000 1250 1500 1750 2000)
M_OFFLINE_VALUES=(250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000)

# Change to the directory where the script is located (src directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Loop through each N value
for N in "${N_VALUES[@]}"; do
    for M_OFFLINE in "${M_OFFLINE_VALUES[@]}"; do
        echo "=========================================="
        echo "Running with N = $N and M_OFFLINE = $M_OFFLINE"
        echo "=========================================="
        
        # Run the Python script with the current N value
        python bounded_lp_vs_dim_linear.py \
            --N "$N" \
            --seeds 10 \
            --M_offline "$M_OFFLINE"  # before was 250 (M_OFFLINE)
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed N = $N and M_OFFLINE = $M_OFFLINE"
        else
            echo "✗ Failed for N = $N and M_OFFLINE = $M_OFFLINE"
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "All runs completed!"
echo "=========================================="
