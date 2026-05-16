#!/bin/bash

# Script to run bounded_lp_vs_dim_nonlinear.py with different N values

# Array of N values to test
N_VALUES=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
M_OFFLINE_VALUES=(500 1000 2500 5000)

SYSTEM="${SYSTEM:-point_mass}"
POINT_MASS_GRAVITY="${POINT_MASS_GRAVITY:-1}"           # Options: 1/0, true/false, yes/no, on/off
POINT_MASS_GRAVITY_MARGIN="${POINT_MASS_GRAVITY_MARGIN:-1.0}"
POINT_MASS_GRAVITY_DIAG="${POINT_MASS_GRAVITY_DIAG:-}"
POINT_MASS_GRAVITY_TYPE="${POINT_MASS_GRAVITY_TYPE:-tanh}"  # Options: linear/sin/tanh/log
POINT_MASS_INTEGRATOR="${POINT_MASS_INTEGRATOR:-rk4}"        # Options: rk4/euler
POINT_MASS_LINEAR_DAMPING="${POINT_MASS_LINEAR_DAMPING:-5.0}"
POINT_MASS_Q4_P="${POINT_MASS_Q4_P:-1.0}"
POINT_MASS_Q4_V="${POINT_MASS_Q4_V:-0.0}"
POINT_MASS_R4_U="${POINT_MASS_R4_U:-0.0}"

# Change to the directory where the script is located (src directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

EXTRA_ARGS=()
case "$POINT_MASS_GRAVITY_TYPE" in
    linear|sin|tanh|log)
        ;;
    *)
        echo "POINT_MASS_GRAVITY_TYPE must be one of: linear, sin, tanh, log"
        exit 1
        ;;
esac

case "$POINT_MASS_INTEGRATOR" in
    rk4|euler)
        ;;
    *)
        echo "POINT_MASS_INTEGRATOR must be one of: rk4, euler"
        exit 1
        ;;
esac

POINT_MASS_GRAVITY_ENABLED=0
case "$POINT_MASS_GRAVITY" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        POINT_MASS_GRAVITY_ENABLED=1
        EXTRA_ARGS+=(
            --point_mass_gravity
            --point_mass_gravity_margin "$POINT_MASS_GRAVITY_MARGIN"
            --point_mass_gravity_type "$POINT_MASS_GRAVITY_TYPE"
        )
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off)
        ;;
    *)
        echo "POINT_MASS_GRAVITY must be one of 1/0, true/false, yes/no, on/off"
        exit 1
        ;;
esac

if [ -n "$POINT_MASS_GRAVITY_DIAG" ]; then
    EXTRA_ARGS+=(--point_mass_gravity_diag "$POINT_MASS_GRAVITY_DIAG")
    if [ "$POINT_MASS_GRAVITY_ENABLED" -eq 0 ]; then
        EXTRA_ARGS+=(--point_mass_gravity_type "$POINT_MASS_GRAVITY_TYPE")
    fi
fi

EXTRA_ARGS+=(
    --point_mass_integrator "$POINT_MASS_INTEGRATOR"
    --point_mass_linear_damping "$POINT_MASS_LINEAR_DAMPING"
    --point_mass_q4_p "$POINT_MASS_Q4_P"
    --point_mass_q4_v "$POINT_MASS_Q4_V"
    --point_mass_r4_u "$POINT_MASS_R4_U"
)

# Loop through each N value
for N in "${N_VALUES[@]}"; do
    for M_OFFLINE in "${M_OFFLINE_VALUES[@]}"; do
        echo "=========================================="
        echo "Running with N = $N and M_OFFLINE = $M_OFFLINE"
        echo "=========================================="

        # Run the Python script with the current N value
        CMD=(
            python bounded_lp_vs_dim_nonlinear.py
            --N "$N"
            --dims 2,4,6,8,10
            --seeds 10
            --M_offline "$M_OFFLINE"
            --fixed_du 1
            --randomize_system
            --plot_trajectories
            "${EXTRA_ARGS[@]}"
        )

        if [ "$SYSTEM" != "point_mass" ]; then
            CMD+=(--system "$SYSTEM")
        fi

        "${CMD[@]}"
        
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
