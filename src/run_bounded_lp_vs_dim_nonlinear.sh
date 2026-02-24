#!/bin/bash

# Script to run bounded_lp_vs_dim_nonlinear.py with different N values

# Array of N values to test
N_VALUES=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000 15000 20000)

# Change to the directory where the script is located (src directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Loop through each N value
for N in "${N_VALUES[@]}"; do
    echo "=========================================="
    echo "Running with N = $N"
    echo "=========================================="
    
    # Run the Python script with the current N value
    # Add other arguments as needed (you can modify these)
    python bounded_lp_vs_dim_nonlinear.py \
        --N "$N" \
        --dims 2,4,6,8,10 \
        --seeds 10 \
        --M_offline 500 \
        --fixed_du 1\
        --randomize_system \
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed N = $N"
    else
        echo "✗ Failed for N = $N"
        # Uncomment the next line if you want to stop on first error:
        # exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All runs completed!"
echo "=========================================="
