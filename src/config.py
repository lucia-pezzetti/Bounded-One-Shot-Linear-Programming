"""
Centralized configuration for the Bounded One-Shot Linear Programming project.

This module contains all the key parameters used across the project to ensure
consistency and make it easy to modify parameters in one place.
"""

import numpy as np

# =============================================================================
# DISCOUNT FACTOR
# =============================================================================
GAMMA = 0.99  # Discount factor for infinite horizon problems

# =============================================================================
# CART-POLE SYSTEM PARAMETERS
# =============================================================================
M_C = 4.0     # cart mass (kg)
M_P = 2.0     # pole mass (kg)
L = 1.0       # half-pole length (m)
DT = 0.01    # sampling time (s)

# Cost matrices for cart-pole system
C_CART_POLE = np.diag(np.sqrt([1, 1, 100, 10]))
RHO_CART_POLE = 0.001

# State/action sampling bounds for cart-pole
X_BOUNDS = (-3.0, 3.0)
X_DOT_BOUNDS = (-3.0, 3.0)
THETA_BOUNDS = (-1.0, 1.0)
THETA_DOT_BOUNDS = (-1.0, 1.0)
U_BOUNDS = (-100.0, 100.0)

# =============================================================================
# POLYNOMIAL FEATURE PARAMETERS
# =============================================================================
DEGREE = 1    # polynomial feature degree

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
DEFAULT_M_OFFLINE = 500    # Default offline pool size
DEFAULT_REGULARIZATION = 1e-6  # Default regularization parameter

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

def get_gamma():
    """Get the validated gamma value."""
    return validate_gamma(GAMMA)

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
        'gamma': GAMMA
    }
