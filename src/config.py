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
GAMMA = 0.99  # discount factor

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

# Physical parameters (nominal values)
M_POINT_MASS_2D = 5.0   # mass (kg)
K_POINT_MASS_2D = 10.0   # spring constant (N/m) pulling towards origin (used by fully-actuated variant)
C_POINT_MASS_2D = 0.5   # cubic/quad-drag coefficient (N*s^2/m^2) ##### with larger values, it's more difficult to get bounded id LP

# Modal stiffness scaling for point_mass_cubic_drag_2du:
#   k0 = m * OMEGA_MAX^2 / n^alpha   so that the highest modal frequency is always OMEGA_MAX,
#   regardless of n and alpha.  (lambda_n = k0 * n^alpha = m * OMEGA_MAX^2)
OMEGA_MAX_POINT_MASS = 5.0  # highest modal natural frequency (rad/s)

# LogNormal distribution parameters for system randomization
# Sample X ~ LogNormal(μ, σ) where μ = log(X0) - 0.5*σ² ensures E[X] = X0
# σ controls coefficient of variation: CV ≈ σ for small σ, CV = sqrt(exp(σ²)-1) exactly
SIGMA_M_POINT_MASS = 0.3   # ~30% CV for mass uncertainty
SIGMA_K_POINT_MASS = 0.4   # ~40% CV for spring constant uncertainty  
SIGMA_C_POINT_MASS = 0.5   # ~50% CV for damping uncertainty (hardest to measure)

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
P_BOUNDS_POINT_MASS = (-8.0, 8.0)    # Position bounds (applied to each component)
V_BOUNDS_POINT_MASS = (-8.0, 8.0)    # Velocity bounds (applied to each component)
U_BOUNDS_POINT_MASS = (-20.0, 20.0)    # Control bounds (applied to each component)

# Focused bounds for test/evaluation sampling (optional)
P_BOUNDS_POINT_MASS_TEST = (-6.0, 6.0)  # Test position bounds
V_BOUNDS_POINT_MASS_TEST = (-6.0, 6.0)  # Test velocity bounds

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