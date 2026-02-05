"""
Centralized configuration for the Bounded One-Shot Linear Programming project.

This module contains all the key parameters used across the project to ensure
consistency and make it easy to modify parameters in one place.
"""

import numpy as np

# =============================================================================
# CART-POLE SYSTEM PARAMETERS
# =============================================================================
M_C = 2.0     # cart mass (kg)
M_P = 1.0     # pole mass (kg)
L = 1.0       # half-pole length (m)
DT = 0.001    # sampling time (s)

# =============================================================================
# SINGLE PENDULUM SYSTEM PARAMETERS
# =============================================================================
M_SINGLE_PENDULUM = 1.0     # bob mass (kg)
L_SINGLE_PENDULUM = 1.0     # pendulum length (m)
B_SINGLE_PENDULUM = 0.05    # viscous damping coefficient (N·m·s/rad)
DT_SINGLE_PENDULUM = 0.02   # sampling time (s)

# =============================================================================
# DISCOUNT FACTOR
# =============================================================================
# Continuous-time discount rate (per second) - independent of discretization
DISCOUNT_RATE_PER_SECOND = 1e-3  # 1% per second discount rate

# Convert continuous-time discount rate to discrete-time discount factor
# gamma = exp(-discount_rate * dt)
# GAMMA = np.exp(-DISCOUNT_RATE_PER_SECOND * DT)
GAMMA = 0.99

# Legacy ALPHA parameter (kept for backward compatibility)
ALPHA = DISCOUNT_RATE_PER_SECOND

# Cost matrices for cart-pole system
C_CART_POLE = np.diag(np.sqrt([1, 1, 100, 10])) # np.diag(np.sqrt([1, 1, 100, 10]))
RHO_CART_POLE = 0.001

# Cost matrices for single pendulum system
C_SINGLE_PENDULUM = np.diag(np.sqrt([100.0, 1.0]))
RHO_SINGLE_PENDULUM = 1.0

# State/action sampling bounds for cart-pole - expanded for nonlinear regime
X_BOUNDS = (-5.0, 5.0)
X_DOT_BOUNDS = (-5.0, 5.0)
THETA_BOUNDS = (-1.5, 1.5)  # Expanded to include more nonlinear dynamics
THETA_DOT_BOUNDS = (-1.5, 1.5)  # Expanded to include more nonlinear dynamics
U_BOUNDS = (-100.0, 100.0)

# State/action sampling bounds for single pendulum
THETA_BOUNDS_SINGLE_PENDULUM = (-np.pi, np.pi)
THETA_DOT_BOUNDS_SINGLE_PENDULUM = (-2.0, 2.0)
U_BOUNDS_SINGLE_PENDULUM = (-3.0, 3.0)

# =============================================================================
# MOUNTAIN CAR SYSTEM PARAMETERS
# =============================================================================
DT_MOUNTAIN_CAR = 0.1  # sampling time (s)
GOAL_POSITION_MOUNTAIN_CAR = 0.5  # Goal position (top of hill)

# Cost matrix for mountain car: penalize distance from goal and velocity
# C matrix defines cost weights: [position_error_weight, velocity_weight]
C_MOUNTAIN_CAR = np.diag(np.sqrt([100.0, 0.0]))  # High weight on position error
RHO_MOUNTAIN_CAR = 1.0  # Control cost weight

# State/action sampling bounds for mountain car
POSITION_BOUNDS_MOUNTAIN_CAR = (-1.2, 0.6)  # Position bounds
VELOCITY_BOUNDS_MOUNTAIN_CAR = (-0.07, 0.07)  # Velocity bounds
U_BOUNDS_MOUNTAIN_CAR = (-1.0, 1.0)  # Control force bounds

# =============================================================================
# SCALAR CUBIC SYSTEM PARAMETERS
# =============================================================================
DT_SCALAR_CUBIC = 0.001  # sampling time (s)

# Cost matrix for scalar cubic system: penalize state deviation
C_SCALAR_CUBIC = np.array([[0.1]])  # State cost weight
RHO_SCALAR_CUBIC = 0.1  # Control cost weight

# State/action sampling bounds for scalar cubic system
X_BOUNDS_SCALAR_CUBIC = (-10.0, 10.0)  # State bounds (keep small to avoid instability)
U_BOUNDS_SCALAR_CUBIC = (-10.0, 10.0)  # Control bounds

# Early termination threshold for scalar cubic system
# Simulation will terminate early if |x| exceeds this value to prevent overflow
# The cubic term x^3 can cause rapid divergence, so this prevents numerical issues
SCALAR_CUBIC_OVERFLOW_THRESHOLD = 500.0  # Terminate if |x| > this value

# =============================================================================
# COUPLED CUBIC SYSTEM PARAMETERS
# =============================================================================
DT_COUPLED_CUBIC = 0.001  # sampling time (s)

# Cost matrix for coupled cubic system: penalize state deviation
# C matrix is identity (defined dynamically based on dx)
RHO_COUPLED_CUBIC = 0.1  # Control cost weight

# State/action sampling bounds for coupled cubic system
X_BOUNDS_COUPLED_CUBIC = (-2.0, 2.0)  # State bounds (keep small to avoid instability)
U_BOUNDS_COUPLED_CUBIC = (-10.0, 10.0)  # Control bounds

# Matrix generation parameters for coupled cubic system
COUPLED_CUBIC_A_DIAGONAL = 0.5  # Diagonal value for A matrix
COUPLED_CUBIC_A_OFFDIAGONAL_RANGE = (-0.1, 0.1)  # Range for off-diagonal coupling terms
COUPLED_CUBIC_B_RANGE = (-0.1, 0.1)  # Range for B matrix entries
COUPLED_CUBIC_MATRIX_SEED = 42  # Fixed seed for matrix generation reproducibility

# Focused bounds for evaluation grids (single pendulum)
THETA_BOUNDS_SINGLE_PENDULUM_TEST = (-3.0, 3.0)
THETA_DOT_BOUNDS_SINGLE_PENDULUM_TEST = (-6.0, 6.0)

X_BOUNDS_GRID = (-5.0, 5.0)
X_DOT_BOUNDS_GRID = (-5.0, 5.0)
THETA_BOUNDS_GRID = (-1.5, 1.5)
THETA_DOT_BOUNDS_GRID = (-1.5, 1.5)

# =============================================================================
# BALL-PLATE SYSTEM PARAMETERS
# =============================================================================
DT_BALL_PLATE = 0.01  # sampling time (s) (can be same as DT, but kept separate)

# Physical parameters (from your figure)
M_BALL_PLATE = 0.258       # ball mass (kg)
MU_S_BALL_PLATE = 0.12     # static friction coefficient
MU_C_BALL_PLATE = 0.0045   # kinetic/coulomb friction coefficient
V_S_BALL_PLATE = 0.01      # characteristic velocity (m/s) used in theta_c
G_BALL_PLATE = 9.8         # gravity (m/s^2)

# Inertia ratio term I/r^2 (not shown explicitly in the table; if unknown, set 0.0)
# Then theta_I = mg/(m + I/r^2) reduces to g.
I_OVER_R2_BALL_PLATE = 0.0

# Cost matrices for ball-plate
# State: [s_x, v_x, s_y, v_y]
C_BALL_PLATE = np.diag(np.sqrt([50.0, 1.0, 50.0, 1.0]))
RHO_BALL_PLATE = 0.1  # control cost weight for [alpha, beta]

# State/action sampling bounds for ball-plate (uniform hyper-rectangle)
# Position bounds (meters)
S_MAX_BALL_PLATE = 0.15  # ±15 cm
# Velocity bounds (m/s)
V_MAX_BALL_PLATE = 0.3   # ±0.3 m/s

# Angle bounds (radians): ±10 degrees
ANGLE_MAX_BALL_PLATE = 10.0 * np.pi / 180.0  # ≈ 0.1745

# Bounds packed as (low, high) arrays (matches your sampling functions nicely)
X_BOUNDS_BALL_PLATE = (
    np.array([-S_MAX_BALL_PLATE, -V_MAX_BALL_PLATE, -S_MAX_BALL_PLATE, -V_MAX_BALL_PLATE]),
    np.array([+S_MAX_BALL_PLATE, +V_MAX_BALL_PLATE, +S_MAX_BALL_PLATE, +V_MAX_BALL_PLATE]),
)

U_BOUNDS_BALL_PLATE = (
    np.array([-ANGLE_MAX_BALL_PLATE, -ANGLE_MAX_BALL_PLATE]),  # [alpha, beta]
    np.array([+ANGLE_MAX_BALL_PLATE, +ANGLE_MAX_BALL_PLATE]),
)




# =============================================================================
# 2D POINT-MASS WITH SPRING + CUBIC DRAG (NEW MODEL)
# =============================================================================
# State x = [p_x, p_y, v_x, v_y] in R^4
# Input u = [u_x, u_y] in R^2
#
# Continuous-time model:
#   p_dot = v
#   v_dot = -(k/m) p - (c/m) ||v||^2 v + (1/m) u
#
# Discretization is handled in the dynamical system class via forward Euler.

DT_POINT_MASS_2D = DT  # reuse global project sampling time by default

# Physical parameters
M_POINT_MASS_2D = 2.0   # mass (kg)
K_POINT_MASS_2D = 10.0   # spring constant (N/m) pulling towards origin
C_POINT_MASS_2D = 1.2   # cubic/quad-drag coefficient (N*s^2/m^2)

# Quadratic stage cost weights (in shifted/original coordinates as you define in your scripts)
# cost(x,u) = || C x ||^2 + rho * ||u||^2
# Cost matrix is built as: C = diag(sqrt([C_p, ..., C_p, C_v, ..., C_v]))
# where C_p is the weight for position components and C_v is the weight for velocity components
C_P_POINT_MASS = 1.0  # Position cost weight
C_V_POINT_MASS = 1.0   # Velocity cost weight
# Legacy: keep for backward compatibility (2D case: n=2)
C_POINT_MASS_2D_COST = np.diag(np.sqrt([C_P_POINT_MASS, C_P_POINT_MASS, C_V_POINT_MASS, C_V_POINT_MASS]))
RHO_POINT_MASS_2D = 0.01

# Sampling bounds for data generation (uniform hyper-rectangle)
# Position and velocity bounds apply component-wise to all dimensions.
# These are tuples (low, high) that are applied to each component.
P_BOUNDS_POINT_MASS = (-3.0, 3.0)    # Position bounds (applied to each component)
V_BOUNDS_POINT_MASS = (-3.0, 3.0)    # Velocity bounds (applied to each component)
U_BOUNDS_POINT_MASS = (-8.0, 8.0)    # Control bounds (applied to each component)

# Focused bounds for test/evaluation sampling (optional)
P_BOUNDS_POINT_MASS_TEST = (-3.0, 3.0)  # Test position bounds
V_BOUNDS_POINT_MASS_TEST = (-3.0, 3.0)  # Test velocity bounds

# Legacy names for backward compatibility (2D-specific naming)
P_BOUNDS_POINT_MASS_2D = P_BOUNDS_POINT_MASS
V_BOUNDS_POINT_MASS_2D = V_BOUNDS_POINT_MASS
U_BOUNDS_POINT_MASS_2D = U_BOUNDS_POINT_MASS
P_BOUNDS_POINT_MASS_2D_TEST = P_BOUNDS_POINT_MASS_TEST
V_BOUNDS_POINT_MASS_2D_TEST = V_BOUNDS_POINT_MASS_TEST


# =============================================================================
# POLYNOMIAL FEATURE PARAMETERS
# =============================================================================
DEGREE = 2    # polynomial feature degree (increased to capture nonlinear dynamics)

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
DEFAULT_M_OFFLINE = 5000    # Default offline pool size
DEFAULT_REGULARIZATION = 1e-6  # Default regularization parameter (increased for degree 2 stability)
ADAPTIVE_REGULARIZATION_SCALING = False  # Enable adaptive regularization scaling with sample size

# =============================================================================
# LUNAR LANDING PARAMETERS
# =============================================================================
DX_LUNAR = 3  # state dimension (height, velocity, mass)
DU_LUNAR = 1  # input dimension (control force)

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def validate_gamma(gamma):
    """Validate that gamma is in the correct range for discount factors."""
    if not isinstance(gamma, (int, float)):
        raise TypeError(f"Gamma must be a number, got {type(gamma)}")
    if not 0 < gamma < 1:
        raise ValueError(f"Gamma must be in (0, 1), got {gamma}")
    return gamma

def validate_discount_rate(rate):
    """Validate that discount rate is non-negative."""
    if not isinstance(rate, (int, float)):
        raise TypeError(f"Discount rate must be a number, got {type(rate)}")
    if rate < 0:
        raise ValueError(f"Discount rate must be non-negative, got {rate}")
    return rate

def continuous_to_discrete_discount(discount_rate_per_second, dt):
    """
    Convert continuous-time discount rate to discrete-time discount factor.
    
    Args:
        discount_rate_per_second: Continuous-time discount rate (per second)
        dt: Discretization time step (seconds)
        
    Returns:
        gamma: Discrete-time discount factor
        
    Formula: gamma = exp(-discount_rate * dt)
    """
    discount_rate_per_second = validate_discount_rate(discount_rate_per_second)
    if dt <= 0:
        raise ValueError(f"Time step must be positive, got {dt}")
    
    gamma = np.exp(-discount_rate_per_second * dt)
    return validate_gamma(gamma)

def discrete_to_continuous_discount(gamma, dt):
    """
    Convert discrete-time discount factor to continuous-time discount rate.
    
    Args:
        gamma: Discrete-time discount factor
        dt: Discretization time step (seconds)
        
    Returns:
        discount_rate_per_second: Continuous-time discount rate (per second)
        
    Formula: discount_rate = -log(gamma) / dt
    """
    gamma = validate_gamma(gamma)
    if dt <= 0:
        raise ValueError(f"Time step must be positive, got {dt}")
    
    discount_rate_per_second = -np.log(gamma) / dt
    return validate_discount_rate(discount_rate_per_second)

def get_gamma():
    """Get the validated gamma value for the current DT."""
    return validate_gamma(GAMMA)

def get_gamma_for_dt(dt):
    """Get the discount factor for a specific discretization time step."""
    return continuous_to_discrete_discount(DISCOUNT_RATE_PER_SECOND, dt)

def get_discount_rate_per_second():
    """Get the continuous-time discount rate per second."""
    return validate_discount_rate(DISCOUNT_RATE_PER_SECOND)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def get_cart_pole_params():
    """Get all cart-pole system parameters as a dictionary."""
    return {
        'M_c': M_C,
        'M_p': M_P,
        'l': L,
        'dt': DT,
        'C': C_CART_POLE,
        'rho': RHO_CART_POLE,
        'gamma': GAMMA,
        'discount_rate_per_second': DISCOUNT_RATE_PER_SECOND,
        'x_bounds': X_BOUNDS,
        'x_dot_bounds': X_DOT_BOUNDS,
        'theta_bounds': THETA_BOUNDS,
        'theta_dot_bounds': THETA_DOT_BOUNDS,
        'u_bounds': U_BOUNDS
    }

def get_single_pendulum_params():
    """Get all single pendulum system parameters as a dictionary."""
    return {
        'm': M_SINGLE_PENDULUM,
        'l': L_SINGLE_PENDULUM,
        'b': B_SINGLE_PENDULUM,
        'dt': DT_SINGLE_PENDULUM,
        'C': C_SINGLE_PENDULUM,
        'rho': RHO_SINGLE_PENDULUM,
        'gamma': GAMMA,
        'theta_bounds': THETA_BOUNDS_SINGLE_PENDULUM,
        'theta_dot_bounds': THETA_DOT_BOUNDS_SINGLE_PENDULUM,
        'u_bounds': U_BOUNDS_SINGLE_PENDULUM,
        'theta_bounds_test': THETA_BOUNDS_SINGLE_PENDULUM_TEST,
        'theta_dot_bounds_test': THETA_DOT_BOUNDS_SINGLE_PENDULUM_TEST
    }

def get_lunar_landing_params():
    """Get lunar landing system parameters as a dictionary."""
    return {
        'dx': DX_LUNAR,
        'du': DU_LUNAR,
        'gamma': GAMMA,
        'discount_rate_per_second': DISCOUNT_RATE_PER_SECOND
    }

def get_mountain_car_params():
    """Get all mountain car system parameters as a dictionary."""
    return {
        'dt': DT_MOUNTAIN_CAR,
        'C': C_MOUNTAIN_CAR,
        'rho': RHO_MOUNTAIN_CAR,
        'gamma': GAMMA,
        'goal_position': GOAL_POSITION_MOUNTAIN_CAR,
        'position_bounds': POSITION_BOUNDS_MOUNTAIN_CAR,
        'velocity_bounds': VELOCITY_BOUNDS_MOUNTAIN_CAR,
        'u_bounds': U_BOUNDS_MOUNTAIN_CAR
    }