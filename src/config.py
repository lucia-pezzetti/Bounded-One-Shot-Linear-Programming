"""
Centralized configuration for the Bounded One-Shot Linear Programming project.

This module contains all the key parameters used across the project to ensure
consistency and make it easy to modify parameters in one place.
"""

import numpy as np

# =============================================================================
# CART-POLE SYSTEM PARAMETERS
# =============================================================================
M_C = 1.0     # cart mass (kg)
M_P = 0.1     # pole mass (kg)
L = 0.5       # half-pole length (m)
DT = 0.001     # sampling time (s)

# =============================================================================
# DISCOUNT FACTOR
# =============================================================================
# Continuous-time discount rate (per second) - independent of discretization
DISCOUNT_RATE_PER_SECOND = 0.001  # 1% per second discount rate

# Convert continuous-time discount rate to discrete-time discount factor
# gamma = exp(-discount_rate * dt)
GAMMA = np.exp(-DISCOUNT_RATE_PER_SECOND * DT)

# Legacy ALPHA parameter (kept for backward compatibility)
ALPHA = DISCOUNT_RATE_PER_SECOND

# Cost matrices for cart-pole system
C_CART_POLE = np.diag(np.sqrt([1, 1, 100, 10])) # np.diag(np.sqrt([1, 1, 100, 10]))
RHO_CART_POLE = 0.01

# State/action sampling bounds for cart-pole - expanded for nonlinear regime
X_BOUNDS = (-2.0, 2.0)
X_DOT_BOUNDS = (-3.0, 3.0)
THETA_BOUNDS = (-0.5, 0.5)  # Expanded to include more nonlinear dynamics
THETA_DOT_BOUNDS = (-1.0, 1.0)  # Expanded to include more nonlinear dynamics
U_BOUNDS = (-50.0, 50.0)

X_BOUNDS_GRID = (-5.0, 5.0)
X_DOT_BOUNDS_GRID = (-5.0, 5.0)
THETA_BOUNDS_GRID = (-1.5, 1.5)
THETA_DOT_BOUNDS_GRID = (-1.5, 1.5)

# =============================================================================
# POLYNOMIAL FEATURE PARAMETERS
# =============================================================================
DEGREE = 2    # polynomial feature degree (increased to capture nonlinear dynamics)

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
DEFAULT_M_OFFLINE = 1000    # Default offline pool size
DEFAULT_REGULARIZATION = 1e-4  # Default regularization parameter (increased for degree 2 stability)
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

def get_lunar_landing_params():
    """Get lunar landing system parameters as a dictionary."""
    return {
        'dx': DX_LUNAR,
        'du': DU_LUNAR,
        'gamma': GAMMA,
        'discount_rate_per_second': DISCOUNT_RATE_PER_SECOND
    }
