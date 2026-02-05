#!/usr/bin/env python3
"""
Convergence Region Analysis for Cart Pole Policies

This script compares the convergence regions of moment matching and LQR policies
in the angle-angular velocity space, with fixed initial position and velocity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from feature_scaling import PolynomialFeatureScaler
from dynamical_systems import cart_pole, single_pendulum, scalar_cubic
from policy_extraction import PolicyExtractor
from moment_matching import solve_moment_matching_Q, solve_identity_Q
from bounded_lp_sweep import get_system_config
from matplotlib.patches import Patch, Rectangle, Circle
import argparse
import json
import time
import warnings

# Suppress CVXPY warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
import os
from config import (
    GAMMA, M_C, M_P, L, DT, C_CART_POLE, RHO_CART_POLE, DEGREE,
    X_BOUNDS, X_DOT_BOUNDS, THETA_BOUNDS, THETA_DOT_BOUNDS, U_BOUNDS,
    DEFAULT_M_OFFLINE, DEFAULT_REGULARIZATION,
    M_SINGLE_PENDULUM, L_SINGLE_PENDULUM, B_SINGLE_PENDULUM, DT_SINGLE_PENDULUM,
    C_SINGLE_PENDULUM, RHO_SINGLE_PENDULUM,
    THETA_BOUNDS_SINGLE_PENDULUM, THETA_DOT_BOUNDS_SINGLE_PENDULUM, U_BOUNDS_SINGLE_PENDULUM,
    THETA_BOUNDS_SINGLE_PENDULUM_TEST, THETA_DOT_BOUNDS_SINGLE_PENDULUM_TEST,
    DT_SCALAR_CUBIC, C_SCALAR_CUBIC, RHO_SCALAR_CUBIC,
    X_BOUNDS_SCALAR_CUBIC, U_BOUNDS_SCALAR_CUBIC, SCALAR_CUBIC_OVERFLOW_THRESHOLD
)

# -------------------------
# Configuration
# -------------------------
M_c = M_C
M_p = M_P
l = L
dt = DT
gamma = GAMMA  # Using centralized discount factor
C = C_CART_POLE
rho = RHO_CART_POLE

degree = DEGREE

# State/action sampling bounds
x_bounds = X_BOUNDS
x_dot_bounds = X_DOT_BOUNDS
theta_bounds = THETA_BOUNDS
theta_dot_bounds = THETA_DOT_BOUNDS   
u_bounds = U_BOUNDS

# Grid search parameters for convergence analysis
# Defaults for cartpole (will be overridden for single_pendulum)
theta_grid_bounds = THETA_BOUNDS      # radians
theta_dot_grid_bounds = THETA_DOT_BOUNDS  # rad/s
grid_resolution = 100                  # points per dimension
simulation_horizon = 5000             # simulation steps (matches bounded_lp_sweep.py)
convergence_threshold = 0.05         # threshold for considering converged (matches policy_extraction.py)

class ConvergenceAnalyzer:
    """
    Analyzes convergence regions for different policies
    """
    
    def __init__(self, system, extractor, mm_policy, lqr_policy, convergence_threshold, system_type="cartpole"):
        self.system = system
        self.extractor = extractor
        self.mm_policy = mm_policy
        self.lqr_policy = lqr_policy
        self.convergence_threshold = convergence_threshold
        self.horizon = simulation_horizon
        self.system_type = system_type.lower()
        self.n_states = system.N_x
        
    def is_converged(self, trajectory, threshold=None):
        """
        Check if a trajectory has converged to the equilibrium point
        
        Args:
            trajectory: array of shape (horizon, n_states) 
                        For cartpole: [x, x_dot, theta, theta_dot]
                        For single_pendulum: [theta, theta_dot]
            threshold: convergence threshold (default: self.convergence_threshold)
            
        Returns:
            bool: True if converged, False otherwise
        """
        if threshold is None:
            threshold = self.convergence_threshold
            
        if len(trajectory) < 100:  # Need minimum trajectory length
            return False
            
        # Check if the last 100 states are close to equilibrium
        last_states = trajectory[-100:]
        
        # Equilibrium point depends on system type
        if self.system_type == "cartpole":
            equilibrium = np.array([0.0, 0.0, 0.0, 0.0])
        elif self.system_type == "single_pendulum":
            equilibrium = np.array([0.0, 0.0])
        elif self.system_type == "scalar_cubic":
            equilibrium = np.array([0.0])
        else:
            equilibrium = np.zeros(self.n_states)
        
        # Check if all states in the last 100 steps are within threshold
        # Match policy_extraction.py: use L2 norm and check ALL states are within threshold
        distances_to_equilibrium = np.linalg.norm(last_states - equilibrium, axis=1)
        
        # If any of the last 100 states are too far from equilibrium, mark as not converged
        # This matches the logic in policy_extraction.py _simulate_policy method
        return np.all(distances_to_equilibrium <= threshold)
    
    def simulate_policy(self, policy, initial_state, horizon=None, return_controls=False, wrap_angles=False):
        """
        Simulate a policy from an initial state
        
        Args:
            policy: policy function
            initial_state: initial state [x, x_dot, theta, theta_dot]
            horizon: simulation horizon (default: self.horizon)
            return_controls: whether to return control sequence
            wrap_angles: if True, wrap angles to [-pi, pi) during simulation
            
        Returns:
            trajectory: state trajectory
            success: whether simulation was successful
            controls: control sequence (if return_controls=True)
        """
        if horizon is None:
            horizon = self.horizon
            
        x_current = initial_state.copy()
        trajectory = [x_current.copy()]
        controls = []
        success = True
        
        for step in range(horizon):
            try:
                # Get control action with error handling
                u_current = policy(x_current)
                if u_current is None or not np.isfinite(u_current).all():
                    success = False
                    break
                
                # Handle different return types
                if hasattr(u_current, '__len__') and len(u_current) > 0:
                    # Ensure we extract a scalar value properly
                    if hasattr(u_current[0], 'item'):
                        u_current = u_current[0].item()
                    else:
                        u_current = float(np.asarray(u_current[0]).item())
                elif hasattr(u_current, 'item'):
                    u_current = u_current.item()
                else:
                    u_current = float(u_current)
                
                # Store control
                controls.append(u_current)
                
                # Check for invalid control values
                if not np.isfinite(u_current):
                    success = False
                    break
                
                # Update state (pass wrap_angles parameter)
                x_current = self.system.step(x_current, u_current, wrap_angles=wrap_angles)
                trajectory.append(x_current.copy())
                
                # Check for divergence during simulation (early termination for scalar_cubic)
                if self.system_type == "scalar_cubic" and np.any(np.abs(x_current) > SCALAR_CUBIC_OVERFLOW_THRESHOLD):
                    success = False
                    break
                    
            except Exception as e:
                print(f"Simulation error at step {step}: {e}")
                success = False
                break
        
        # Check stability based on last 100 samples if we have enough data
        # This matches the logic in policy_extraction.py _simulate_policy method
        if success and len(trajectory) >= 100:
            # Get the last 100 states
            last_100_states = np.array(trajectory[-100:])
            
            # Determine equilibrium point based on system type
            if self.system_type == "cartpole":
                equilibrium = np.array([0.0, 0.0, 0.0, 0.0])
            elif self.system_type == "single_pendulum":
                equilibrium = np.array([0.0, 0.0])
            elif self.system_type == "scalar_cubic":
                equilibrium = np.array([0.0])
            else:
                equilibrium = np.zeros(self.n_states)
            
            # Check if all last 100 states are sufficiently close to equilibrium (origin)
            equilibrium_tolerance = self.convergence_threshold
            distances_to_equilibrium = np.linalg.norm(last_100_states - equilibrium, axis=1)
            
            # If any of the last 100 states are too far from equilibrium, mark as unstable
            if not np.all(distances_to_equilibrium <= equilibrium_tolerance):
                success = False
        
        if return_controls:
            return np.array(trajectory), success, np.array(controls)
        else:
            return np.array(trajectory), success
    
    def analyze_convergence_grid(self, theta_range=None, theta_dot_range=None, theta_resolution=50, theta_dot_resolution=None, x_range=None, x_resolution=100):
        """
        Analyze convergence regions on a grid
        
        Args:
            theta_range: tuple (min, max) for angle (for 2D systems)
            theta_dot_range: tuple (min, max) for angular velocity (for 2D systems)
            theta_resolution: grid resolution for angle (points) (for 2D systems)
            theta_dot_resolution: grid resolution for angular velocity (points). If None, uses theta_resolution.
            x_range: tuple (min, max) for state (for 1D systems like scalar_cubic)
            x_resolution: grid resolution for state (points) (for 1D systems)
            
        Returns:
            dict: results containing convergence maps and statistics
        """
        # Handle 1D system (scalar_cubic)
        if self.system_type == "scalar_cubic":
            if x_range is None:
                x_range = (-5.0, 5.0)  # Default range for scalar_cubic
            if x_resolution is None:
                x_resolution = 100
            
            print(f"Analyzing convergence on 1D grid with {x_resolution} points...")
            
            # Create grid
            x_vals = np.linspace(x_range[0], x_range[1], x_resolution)
            
            # Initialize results (1D arrays)
            mm_convergence = np.zeros(x_resolution, dtype=bool)
            lqr_convergence = np.zeros(x_resolution, dtype=bool)
            mm_success = np.zeros(x_resolution, dtype=bool)
            lqr_success = np.zeros(x_resolution, dtype=bool)
            
            total_points = x_resolution
            processed = 0
            
            for i, x_val in enumerate(x_vals):
                initial_state = np.array([x_val])
                
                # Test moment matching policy (with angle wrapping)
                try:
                    mm_traj, mm_sim_success = self.simulate_policy(self.mm_policy, initial_state, wrap_angles=True)
                    mm_success[i] = mm_sim_success
                    if mm_sim_success:
                        mm_convergence[i] = self.is_converged(mm_traj)
                except Exception as e:
                    print(f"MM policy failed at x={x_val:.3f}: {e}")
                    mm_success[i] = False
                    mm_convergence[i] = False
                
                # Test LQR policy (without angle wrapping)
                try:
                    lqr_traj, lqr_sim_success = self.simulate_policy(self.lqr_policy, initial_state, wrap_angles=False)
                    lqr_success[i] = lqr_sim_success
                    if lqr_sim_success:
                        lqr_convergence[i] = self.is_converged(lqr_traj)
                except Exception as e:
                    print(f"LQR policy failed at x={x_val:.3f}: {e}")
                    lqr_success[i] = False
                    lqr_convergence[i] = False
                
                processed += 1
                if processed % (total_points // 10) == 0:
                    print(f"Progress: {processed}/{total_points} ({100*processed/total_points:.1f}%)")
                    print(f"MM success progress: {np.mean(mm_success)}")
                    print(f"LQR success progress: {np.mean(lqr_success)}")
            
            # Compute statistics
            mm_convergence_rate = np.mean(mm_convergence)
            lqr_convergence_rate = np.mean(lqr_convergence)
            mm_success_rate = np.mean(mm_success)
            lqr_success_rate = np.mean(lqr_success)
            
            # Find intersection and union
            both_converge = mm_convergence & lqr_convergence
            either_converges = mm_convergence | lqr_convergence
            only_mm = mm_convergence & ~lqr_convergence
            only_lqr = ~mm_convergence & lqr_convergence
            
            results = {
                'x_vals': x_vals,
                'mm_convergence': mm_convergence,
                'lqr_convergence': lqr_convergence,
                'mm_success': mm_success,
                'lqr_success': lqr_success,
                'both_converge': both_converge,
                'either_converges': either_converges,
                'only_mm': only_mm,
                'only_lqr': only_lqr,
                'statistics': {
                    'mm_convergence_rate': mm_convergence_rate,
                    'lqr_convergence_rate': lqr_convergence_rate,
                    'mm_success_rate': mm_success_rate,
                    'lqr_success_rate': lqr_success_rate,
                    'both_converge_rate': np.mean(both_converge),
                    'either_converges_rate': np.mean(either_converges),
                    'only_mm_rate': np.mean(only_mm),
                    'only_lqr_rate': np.mean(only_lqr),
                }
            }
            
            return results
        
        # Handle 2D systems (cartpole, single_pendulum)
        if theta_dot_resolution is None:
            theta_dot_resolution = theta_resolution
        
        print(f"Analyzing convergence on {theta_resolution}x{theta_dot_resolution} grid...")
        
        # Create grid
        theta_vals = np.linspace(theta_range[0], theta_range[1], theta_resolution)
        theta_dot_vals = np.linspace(theta_dot_range[0], theta_dot_range[1], theta_dot_resolution)
        
        # Initialize results
        mm_convergence = np.zeros((theta_resolution, theta_dot_resolution), dtype=bool)
        lqr_convergence = np.zeros((theta_resolution, theta_dot_resolution), dtype=bool)
        mm_success = np.zeros((theta_resolution, theta_dot_resolution), dtype=bool)
        lqr_success = np.zeros((theta_resolution, theta_dot_resolution), dtype=bool)
        
        total_points = theta_resolution * theta_dot_resolution
        processed = 0
        
        for i, theta in enumerate(theta_vals):
            for j, theta_dot in enumerate(theta_dot_vals):
                # Initial state depends on system type
                if self.system_type == "cartpole":
                    # Fixed initial conditions: position=0, velocity=0
                    initial_state = np.array([0.0, 0.0, theta, theta_dot])
                elif self.system_type == "single_pendulum":
                    # Only angle and angular velocity for single pendulum
                    initial_state = np.array([theta, theta_dot])
                else:
                    raise ValueError(f"Unknown system type: {self.system_type}")
                
                # Test moment matching policy (with angle wrapping)
                try:
                    mm_traj, mm_sim_success = self.simulate_policy(self.mm_policy, initial_state, wrap_angles=True)
                    mm_success[i, j] = mm_sim_success
                    if mm_sim_success:
                        mm_convergence[i, j] = self.is_converged(mm_traj)
                except Exception as e:
                    print(f"MM policy failed at ({theta:.3f}, {theta_dot:.3f}): {e}")
                    mm_success[i, j] = False
                    mm_convergence[i, j] = False
                
                # Test LQR policy (without angle wrapping)
                try:
                    lqr_traj, lqr_sim_success = self.simulate_policy(self.lqr_policy, initial_state, wrap_angles=False)
                    lqr_success[i, j] = lqr_sim_success
                    if lqr_sim_success:
                        lqr_convergence[i, j] = self.is_converged(lqr_traj)
                except Exception as e:
                    print(f"LQR policy failed at ({theta:.3f}, {theta_dot:.3f}): {e}")
                    lqr_success[i, j] = False
                    lqr_convergence[i, j] = False
                
                processed += 1
                if processed % (total_points // 10) == 0:
                    print(f"Progress: {processed}/{total_points} ({100*processed/total_points:.1f}%)")
                    print(f"MM success progress: {np.mean(mm_success)}")
                    print(f"LQR success progress: {np.mean(lqr_success)}")
        
        # Compute statistics
        mm_convergence_rate = np.mean(mm_convergence)
        lqr_convergence_rate = np.mean(lqr_convergence)
        mm_success_rate = np.mean(mm_success)
        lqr_success_rate = np.mean(lqr_success)
        
        # Find intersection and union
        both_converge = mm_convergence & lqr_convergence
        either_converges = mm_convergence | lqr_convergence
        only_mm = mm_convergence & ~lqr_convergence
        only_lqr = ~mm_convergence & lqr_convergence
        
        results = {
            'theta_vals': theta_vals,
            'theta_dot_vals': theta_dot_vals,
            'mm_convergence': mm_convergence,
            'lqr_convergence': lqr_convergence,
            'mm_success': mm_success,
            'lqr_success': lqr_success,
            'both_converge': both_converge,
            'either_converges': either_converges,
            'only_mm': only_mm,
            'only_lqr': only_lqr,
            'statistics': {
                'mm_convergence_rate': mm_convergence_rate,
                'lqr_convergence_rate': lqr_convergence_rate,
                'mm_success_rate': mm_success_rate,
                'lqr_success_rate': lqr_success_rate,
                'both_converge_rate': np.mean(both_converge),
                'either_converges_rate': np.mean(either_converges),
                'only_mm_rate': np.mean(only_mm),
                'only_lqr_rate': np.mean(only_lqr),
            }
        }
        
        return results
    
    def plot_convergence_regions(self, results, save_path=None):
        """
        Plot convergence regions comparison
        
        Args:
            results: results from analyze_convergence_grid
            save_path: path to save the plot
        """
        global plt
        
        # Handle 1D system (scalar_cubic)
        if 'x_vals' in results:
            x_vals = results['x_vals']
            stats = results['statistics']
            
            # Create a single figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Use matplotlib tab10 palette for colorblind-friendly colors
            tab10_colors = plt.cm.tab10.colors
            
            colorblind_colors = {
                'both': tab10_colors[2],      # Green
                'mm_only': tab10_colors[1],   # Orange
                'lqr_only': tab10_colors[3],  # Red
                'neither': tab10_colors[0]    # Blue
            }
            
            # Define markers for additional visual distinction
            markers = {
                'both': 'o',        # Circle
                'mm_only': '^',     # Triangle up
                'lqr_only': 's',    # Square
                'neither': 'x'      # X mark
            }
            
            # Create separate scatter plots for each category
            categories = ['both', 'mm_only', 'lqr_only', 'neither']
            category_labels = ['Both converge', 'Only MM converges', 'Only LQR converges', 'Neither converges']
            
            for cat, label in zip(categories, category_labels):
                # Find points in this category
                points_x = []
                
                for i in range(len(x_vals)):
                    mm_converges = results['mm_convergence'][i]
                    lqr_converges = results['lqr_convergence'][i]
                    
                    if cat == 'both' and mm_converges and lqr_converges:
                        points_x.append(x_vals[i])
                    elif cat == 'mm_only' and mm_converges and not lqr_converges:
                        points_x.append(x_vals[i])
                    elif cat == 'lqr_only' and not mm_converges and lqr_converges:
                        points_x.append(x_vals[i])
                    elif cat == 'neither' and not mm_converges and not lqr_converges:
                        points_x.append(x_vals[i])
                
                # Plot points for this category
                if points_x:  # Only plot if there are points in this category
                    ax.scatter(points_x, [0]*len(points_x), 
                              color=colorblind_colors[cat], 
                              marker=markers[cat],
                              alpha=0.8, 
                              s=60,
                              label=label,
                              edgecolors='black',
                              linewidths=0.5)
            
            # Set labels and title
            ax.set_xlabel('State x', fontsize=12)
            ax.set_ylabel('', fontsize=12)
            ax.set_title(f'Scalar Cubic Policy Convergence Regions (Tab10 Palette)\n' + 
                        f'MM: {stats["mm_convergence_rate"]:.3f}, LQR: {stats["lqr_convergence_rate"]:.3f}, ' +
                        f'Both: {stats["both_converge_rate"]:.3f}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 0.1)  # Small range for visualization
            ax.axhline(y=0, color='black', linewidth=0.5)
            
            # Create legend
            ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
            
            # Add statistics text box
            stats_text = f'Convergence Statistics:\n'
            stats_text += f'MM: {stats["mm_convergence_rate"]:.3f}\n'
            stats_text += f'LQR: {stats["lqr_convergence_rate"]:.3f}\n'
            stats_text += f'Both: {stats["both_converge_rate"]:.3f}\n'
            stats_text += f'Either: {stats["either_converges_rate"]:.3f}\n'
            stats_text += f'Only MM: {stats["only_mm_rate"]:.3f}\n'
            stats_text += f'Only LQR: {stats["only_lqr_rate"]:.3f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black'))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Convergence analysis plot saved to {save_path}")
            
            plt.show()
            
            return fig
        
        # Handle 2D systems (cartpole, single_pendulum)
        theta_vals = results['theta_vals']
        theta_dot_vals = results['theta_dot_vals']
        stats = results['statistics']
        
        # Create a single figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create meshgrid for plotting
        theta_grid, theta_dot_grid = np.meshgrid(theta_vals, theta_dot_vals)
        
        # Flatten the grids for scatter plot
        theta_flat = theta_grid.flatten()
        theta_dot_flat = theta_dot_grid.flatten()
        
        # Use matplotlib tab10 palette for colorblind-friendly colors
        tab10_colors = plt.cm.tab10.colors
        
        colorblind_colors = {
            'both': tab10_colors[2],      # Green
            'mm_only': tab10_colors[1],   # Orange
            'lqr_only': tab10_colors[3],  # Red
            'neither': tab10_colors[0]    # Blue
        }
        
        # Define markers for additional visual distinction
        markers = {
            'both': 'o',        # Circle
            'mm_only': '^',     # Triangle up
            'lqr_only': 's',    # Square
            'neither': 'x'      # X mark
        }
        
        # Create separate scatter plots for each category for better control
        categories = ['both', 'mm_only', 'lqr_only', 'neither']
        category_labels = ['Both converge', 'Only MM converges', 'Only LQR converges', 'Neither converges']
        
        for cat, label in zip(categories, category_labels):
            # Find points in this category
            points_x = []
            points_y = []
            
            for i in range(len(theta_flat)):
                theta_idx = np.argmin(np.abs(theta_vals - theta_flat[i]))
                theta_dot_idx = np.argmin(np.abs(theta_dot_vals - theta_dot_flat[i]))
                
                mm_converges = results['mm_convergence'][theta_idx, theta_dot_idx]
                lqr_converges = results['lqr_convergence'][theta_idx, theta_dot_idx]
                
                if cat == 'both' and mm_converges and lqr_converges:
                    points_x.append(theta_flat[i])
                    points_y.append(theta_dot_flat[i])
                elif cat == 'mm_only' and mm_converges and not lqr_converges:
                    points_x.append(theta_flat[i])
                    points_y.append(theta_dot_flat[i])
                elif cat == 'lqr_only' and not mm_converges and lqr_converges:
                    points_x.append(theta_flat[i])
                    points_y.append(theta_dot_flat[i])
                elif cat == 'neither' and not mm_converges and not lqr_converges:
                    points_x.append(theta_flat[i])
                    points_y.append(theta_dot_flat[i])
            
            # Plot points for this category
            if points_x:  # Only plot if there are points in this category
                ax.scatter(points_x, points_y, 
                          color=colorblind_colors[cat], 
                          marker=markers[cat],
                          alpha=0.8, 
                          s=60,
                          label=label,
                          edgecolors='black',
                          linewidths=0.5)
        
        # Set labels and title
        system_name = "Cart Pole" if self.system_type == "cartpole" else "Single Pendulum"
        ax.set_xlabel('Angle (rad)', fontsize=12)
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
        ax.set_title(f'{system_name} Policy Convergence Regions (Tab10 Palette)\n' + 
                    f'MM: {stats["mm_convergence_rate"]:.3f}, LQR: {stats["lqr_convergence_rate"]:.3f}, ' +
                    f'Both: {stats["both_converge_rate"]:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Create legend with better positioning
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Add statistics text box with better styling
        stats_text = f'Convergence Statistics:\n'
        stats_text += f'MM: {stats["mm_convergence_rate"]:.3f}\n'
        stats_text += f'LQR: {stats["lqr_convergence_rate"]:.3f}\n'
        stats_text += f'Both: {stats["both_converge_rate"]:.3f}\n'
        stats_text += f'Either: {stats["either_converges_rate"]:.3f}\n'
        stats_text += f'Only MM: {stats["only_mm_rate"]:.3f}\n'
        stats_text += f'Only LQR: {stats["only_lqr_rate"]:.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black'))
        
        # Add note about colorblind accessibility
        ax.text(0.98, 0.02, 'Tab10 palette\nwith distinct markers', 
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Convergence analysis plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_trajectory_analysis(self, results, save_path=None):
        """
        Plot trajectory analysis to understand convergence behavior
        
        Args:
            results: results from analyze_convergence_grid
            save_path: path to save the plot
        """
        theta_vals = results['theta_vals']
        theta_dot_vals = results['theta_dot_vals']
        
        # Find some interesting initial conditions
        interesting_points = []
        
        # Use matplotlib tab10 palette for colorblind-friendly colors
        tab10_colors = plt.cm.tab10.colors
        
        colorblind_colors = {
            'lqr': tab10_colors[0],        # Blue
            'mm': tab10_colors[1],         # Orange  
            'non_convergent': tab10_colors[3]  # Red
        }
        
        # Find points where LQR converges
        lqr_convergence = results['lqr_convergence']
        lqr_indices = np.where(lqr_convergence)
        if len(lqr_indices[0]) > 0:
            # Pick a few LQR convergent points
            for i in range(min(3, len(lqr_indices[0]))):
                theta_idx = lqr_indices[0][i]
                theta_dot_idx = lqr_indices[1][i]
                interesting_points.append({
                    'theta': theta_vals[theta_idx],
                    'theta_dot': theta_dot_vals[theta_dot_idx],
                    'label': f'LQR Convergent {i+1}',
                    'color': colorblind_colors['lqr']
                })
        
        # Find points where MM converges
        mm_convergence = results['mm_convergence']
        mm_indices = np.where(mm_convergence)
        if len(mm_indices[0]) > 0:
            # Pick a few MM convergent points
            for i in range(min(2, len(mm_indices[0]))):
                theta_idx = mm_indices[0][i]
                theta_dot_idx = mm_indices[1][i]
                interesting_points.append({
                    'theta': theta_vals[theta_idx],
                    'theta_dot': theta_dot_vals[theta_dot_idx],
                    'label': f'MM Convergent {i+1}',
                    'color': colorblind_colors['mm']
                })
        
        # Add some non-convergent points for comparison
        non_convergent_indices = np.where(~(lqr_convergence | mm_convergence))
        if len(non_convergent_indices[0]) > 0:
            for i in range(min(2, len(non_convergent_indices[0]))):
                theta_idx = non_convergent_indices[0][i]
                theta_dot_idx = non_convergent_indices[1][i]
                interesting_points.append({
                    'theta': theta_vals[theta_idx],
                    'theta_dot': theta_dot_vals[theta_dot_idx],
                    'label': f'Non-Convergent {i+1}',
                    'color': colorblind_colors['non_convergent']
                })
        
        if not interesting_points:
            print("No interesting points found for trajectory analysis")
            return None
        
        # Create figure with subplots
        n_points = len(interesting_points)
        fig, axes = plt.subplots(n_points, 2, figsize=(15, 4*n_points))
        if n_points == 1:
            axes = axes.reshape(1, -1)
        
        for i, point in enumerate(interesting_points):
            # Initial state depends on system type
            if self.system_type == "cartpole":
                initial_state = np.array([0.0, 0.0, point['theta'], point['theta_dot']])
            elif self.system_type == "single_pendulum":
                initial_state = np.array([point['theta'], point['theta_dot']])
            elif self.system_type == "scalar_cubic":
                # For scalar_cubic, we need to adapt the point structure
                # This function is designed for 2D systems, so we'll skip it for scalar_cubic
                continue
            else:
                raise ValueError(f"Unknown system type: {self.system_type}")
            
            # Simulate both policies with controls
            lqr_traj, lqr_success, lqr_controls = self.simulate_policy(self.lqr_policy, initial_state, return_controls=True, wrap_angles=False)
            mm_traj, mm_success, mm_controls = self.simulate_policy(self.mm_policy, initial_state, return_controls=True, wrap_angles=True)
            
            # Plot states with colorblind-friendly colors and distinct line styles
            ax_states = axes[i, 0]
            time_steps = np.arange(len(lqr_traj))
            time_steps_mm = np.arange(len(mm_traj))
            
            if self.system_type == "cartpole":
                # LQR trajectories with solid lines and blue color
                ax_states.plot(time_steps, lqr_traj[:, 0], color=colorblind_colors['lqr'], linestyle='-', linewidth=2, label='Position (LQR)', alpha=0.8)
                ax_states.plot(time_steps, lqr_traj[:, 1], color=colorblind_colors['lqr'], linestyle='--', linewidth=2, label='Velocity (LQR)', alpha=0.8)
                ax_states.plot(time_steps, lqr_traj[:, 2], color=colorblind_colors['lqr'], linestyle=':', linewidth=2, label='Angle (LQR)', alpha=0.8)
                ax_states.plot(time_steps, lqr_traj[:, 3], color=colorblind_colors['lqr'], linestyle='-.', linewidth=2, label='Angular Vel (LQR)', alpha=0.8)
                
                # MM trajectories with solid lines and orange color
                ax_states.plot(time_steps_mm, mm_traj[:, 0], color=colorblind_colors['mm'], linestyle='-', linewidth=2, label='Position (MM)', alpha=0.8)
                ax_states.plot(time_steps_mm, mm_traj[:, 1], color=colorblind_colors['mm'], linestyle='--', linewidth=2, label='Velocity (MM)', alpha=0.8)
                ax_states.plot(time_steps_mm, mm_traj[:, 2], color=colorblind_colors['mm'], linestyle=':', linewidth=2, label='Angle (MM)', alpha=0.8)
                ax_states.plot(time_steps_mm, mm_traj[:, 3], color=colorblind_colors['mm'], linestyle='-.', linewidth=2, label='Angular Vel (MM)', alpha=0.8)
            elif self.system_type == "single_pendulum":
                # LQR trajectories
                ax_states.plot(time_steps, lqr_traj[:, 0], color=colorblind_colors['lqr'], linestyle='-', linewidth=2, label='Angle (LQR)', alpha=0.8)
                ax_states.plot(time_steps, lqr_traj[:, 1], color=colorblind_colors['lqr'], linestyle='--', linewidth=2, label='Angular Vel (LQR)', alpha=0.8)
                
                # MM trajectories
                ax_states.plot(time_steps_mm, mm_traj[:, 0], color=colorblind_colors['mm'], linestyle='-', linewidth=2, label='Angle (MM)', alpha=0.8)
                ax_states.plot(time_steps_mm, mm_traj[:, 1], color=colorblind_colors['mm'], linestyle='--', linewidth=2, label='Angular Vel (MM)', alpha=0.8)
            elif self.system_type == "scalar_cubic":
                # LQR trajectory
                ax_states.plot(time_steps, lqr_traj[:, 0], color=colorblind_colors['lqr'], linestyle='-', linewidth=2, label='State x (LQR)', alpha=0.8)
                
                # MM trajectory
                ax_states.plot(time_steps_mm, mm_traj[:, 0], color=colorblind_colors['mm'], linestyle='-', linewidth=2, label='State x (MM)', alpha=0.8)
            
            ax_states.set_xlabel('Time Step')
            ax_states.set_ylabel('State Value')
            ax_states.set_title(f'{point["label"]}: θ={point["theta"]:.3f}, θ̇={point["theta_dot"]:.3f}')
            ax_states.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_states.grid(True, alpha=0.3)
            # Fix y-axis limits for state plots
            ax_states.set_ylim(-20, 20)
            
            # Plot controls with colorblind-friendly colors
            ax_controls = axes[i, 1]
            
            ax_controls.plot(lqr_controls, color=colorblind_colors['lqr'], linestyle='-', linewidth=2, label='LQR Control', alpha=0.8)
            ax_controls.plot(mm_controls, color=colorblind_colors['mm'], linestyle='-', linewidth=2, label='MM Control', alpha=0.8)
            ax_controls.set_xlabel('Time Step')
            ax_controls.set_ylabel('Control Value')
            ax_controls.set_title(f'Control Trajectories')
            ax_controls.legend()
            ax_controls.grid(True, alpha=0.3)
            
            # Add convergence analysis
            if self.system_type == "cartpole":
                equilibrium = np.array([0.0, 0.0, 0.0, 0.0])
            elif self.system_type == "single_pendulum":
                equilibrium = np.array([0.0, 0.0])
            elif self.system_type == "scalar_cubic":
                equilibrium = np.array([0.0])
            else:
                raise ValueError(f"Unknown system type: {self.system_type}")
            
            if len(lqr_traj) >= 100:
                lqr_last_100 = lqr_traj[-100:]
                lqr_distances = np.linalg.norm(lqr_last_100 - equilibrium, axis=1)
                lqr_max_dist = np.max(lqr_distances)
                lqr_mean_dist = np.mean(lqr_distances)
                lqr_converged = lqr_max_dist < self.convergence_threshold
            else:
                lqr_converged = False
                lqr_max_dist = np.inf
                lqr_mean_dist = np.inf
            
            if len(mm_traj) >= 100:
                mm_last_100 = mm_traj[-100:]
                mm_distances = np.linalg.norm(mm_last_100 - equilibrium, axis=1)
                mm_max_dist = np.max(mm_distances)
                mm_mean_dist = np.mean(mm_distances)
                mm_converged = mm_max_dist < self.convergence_threshold
            else:
                mm_converged = False
                mm_max_dist = np.inf
                mm_mean_dist = np.inf
            
            # Add text box with convergence info
            conv_text = f'LQR: max={lqr_max_dist:.3f}, mean={lqr_mean_dist:.3f}, conv={lqr_converged}\n'
            conv_text += f'MM: max={mm_max_dist:.3f}, mean={mm_mean_dist:.3f}, conv={mm_converged}'
            
            ax_states.text(0.02, 0.98, conv_text, transform=ax_states.transAxes, 
                          fontsize=8, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add colorblind accessibility note
        fig.text(0.99, 0.01, 'Tab10 palette\nwith distinct line styles', 
                ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory analysis plot saved to {save_path}")
        
        plt.show()
        
        return fig

    def plot_policy_comparison_2d(self, theta_range, theta_dot_range, 
                                   theta_resolution=100, theta_dot_resolution=100, 
                                   save_path=None):
        """
        Plot 2D heatmap comparison of LP and LQR policies side by side
        
        Args:
            theta_range: tuple (min, max) for angle
            theta_dot_range: tuple (min, max) for angular velocity
            theta_resolution: grid resolution for angle
            theta_dot_resolution: grid resolution for angular velocity
            save_path: path to save the plot
        """
        print("Generating 2D policy comparison plots...")
        
        # Create grid
        theta_vals = np.linspace(theta_range[0], theta_range[1], theta_resolution)
        theta_dot_vals = np.linspace(theta_dot_range[0], theta_dot_range[1], theta_dot_resolution)
        theta_grid, theta_dot_grid = np.meshgrid(theta_vals, theta_dot_vals)
        
        # Evaluate policies on grid
        lqr_policy_grid = np.zeros_like(theta_grid)
        mm_policy_grid = np.zeros_like(theta_grid)
        
        for i, theta in enumerate(theta_vals):
            for j, theta_dot in enumerate(theta_dot_vals):
                # Create state based on system type
                if self.system_type == "cartpole":
                    state = np.array([0.0, 0.0, theta, theta_dot])
                elif self.system_type == "single_pendulum":
                    state = np.array([theta, theta_dot])
                else:
                    raise ValueError(f"Unknown system type: {self.system_type}")
                
                # Evaluate LQR policy
                try:
                    u_lqr = self.lqr_policy(state)
                    if hasattr(u_lqr, '__len__') and len(u_lqr) > 0:
                        u_lqr = float(u_lqr[0])
                    elif hasattr(u_lqr, 'item'):
                        u_lqr = u_lqr.item()
                    else:
                        u_lqr = float(u_lqr)
                    lqr_policy_grid[j, i] = u_lqr
                except Exception as e:
                    lqr_policy_grid[j, i] = np.nan
                
                # Evaluate MM policy
                try:
                    u_mm = self.mm_policy(state)
                    if hasattr(u_mm, '__len__') and len(u_mm) > 0:
                        u_mm = float(u_mm[0])
                    elif hasattr(u_mm, 'item'):
                        u_mm = u_mm.item()
                    else:
                        u_mm = float(u_mm)
                    mm_policy_grid[j, i] = u_mm
                except Exception as e:
                    mm_policy_grid[j, i] = np.nan
        
        # Find common color scale
        vmin = min(np.nanmin(lqr_policy_grid), np.nanmin(mm_policy_grid))
        vmax = max(np.nanmax(lqr_policy_grid), np.nanmax(mm_policy_grid))
        
        # Create figure with side-by-side subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        system_name = "Cart Pole" if self.system_type == "cartpole" else "Single Pendulum"
        control_label = "Control Force (N)" if self.system_type == "cartpole" else "Control Torque (N·m)"
        
        # Plot LQR policy
        ax1 = axes[0]
        im1 = ax1.contourf(theta_grid, theta_dot_grid, lqr_policy_grid, levels=50, 
                          cmap='RdYlBu_r', extend='both', vmin=vmin, vmax=vmax)
        ax1.set_xlabel('Angle (rad)', fontsize=12)
        ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
        ax1.set_title(f'LQR Policy - {system_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label=control_label)
        
        # Plot MM policy
        ax2 = axes[1]
        im2 = ax2.contourf(theta_grid, theta_dot_grid, mm_policy_grid, levels=50, 
                          cmap='RdYlBu_r', extend='both', vmin=vmin, vmax=vmax)
        ax2.set_xlabel('Angle (rad)', fontsize=12)
        ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
        ax2.set_title(f'LP Policy - {system_name}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label=control_label)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"2D policy comparison plot saved to {save_path}")
        
        plt.show()
        
        return fig

    def plot_policy_comparison_3d(self, theta_range, theta_dot_range, 
                                   theta_resolution=100, theta_dot_resolution=100, 
                                   save_path=None):
        """
        Plot 3D surface comparison of LP and LQR policies side by side
        
        Args:
            theta_range: tuple (min, max) for angle
            theta_dot_range: tuple (min, max) for angular velocity
            theta_resolution: grid resolution for angle
            theta_dot_resolution: grid resolution for angular velocity
            save_path: path to save the plot
        """
        print("Generating 3D policy comparison plots...")
        
        # Create grid
        theta_vals = np.linspace(theta_range[0], theta_range[1], theta_resolution)
        theta_dot_vals = np.linspace(theta_dot_range[0], theta_dot_range[1], theta_dot_resolution)
        theta_grid, theta_dot_grid = np.meshgrid(theta_vals, theta_dot_vals)
        
        # Evaluate policies on grid
        lqr_policy_grid = np.zeros_like(theta_grid)
        mm_policy_grid = np.zeros_like(theta_grid)
        
        for i, theta in enumerate(theta_vals):
            for j, theta_dot in enumerate(theta_dot_vals):
                # Create state based on system type
                if self.system_type == "cartpole":
                    state = np.array([0.0, 0.0, theta, theta_dot])
                elif self.system_type == "single_pendulum":
                    state = np.array([theta, theta_dot])
                else:
                    raise ValueError(f"Unknown system type: {self.system_type}")
                
                # Evaluate LQR policy
                try:
                    u_lqr = self.lqr_policy(state)
                    if hasattr(u_lqr, '__len__') and len(u_lqr) > 0:
                        u_lqr = float(u_lqr[0])
                    elif hasattr(u_lqr, 'item'):
                        u_lqr = u_lqr.item()
                    else:
                        u_lqr = float(u_lqr)
                    lqr_policy_grid[j, i] = u_lqr
                except Exception as e:
                    lqr_policy_grid[j, i] = np.nan
                
                # Evaluate MM policy
                try:
                    u_mm = self.mm_policy(state)
                    if hasattr(u_mm, '__len__') and len(u_mm) > 0:
                        u_mm = float(u_mm[0])
                    elif hasattr(u_mm, 'item'):
                        u_mm = u_mm.item()
                    else:
                        u_mm = float(u_mm)
                    mm_policy_grid[j, i] = u_mm
                except Exception as e:
                    mm_policy_grid[j, i] = np.nan
        
        # Find common z-axis limits
        zmin = min(np.nanmin(lqr_policy_grid), np.nanmin(mm_policy_grid))
        zmax = max(np.nanmax(lqr_policy_grid), np.nanmax(mm_policy_grid))
        
        # Create figure with side-by-side 3D subplots
        fig = plt.figure(figsize=(18, 8))
        
        system_name = "Cart Pole" if self.system_type == "cartpole" else "Single Pendulum"
        control_label = "Control Force (N)" if self.system_type == "cartpole" else "Control Torque (N·m)"
        
        # Plot LQR policy (3D)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(theta_grid, theta_dot_grid, lqr_policy_grid, 
                                 cmap='RdYlBu_r', alpha=0.9, linewidth=0, 
                                 antialiased=True, vmin=zmin, vmax=zmax)
        ax1.set_xlabel('Angle (rad)', fontsize=11)
        ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
        ax1.set_zlabel(control_label, fontsize=11)
        ax1.set_title(f'LQR Policy - {system_name}', fontsize=13, fontweight='bold')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=20, label=control_label)
        
        # Plot MM policy (3D)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(theta_grid, theta_dot_grid, mm_policy_grid, 
                                 cmap='RdYlBu_r', alpha=0.9, linewidth=0, 
                                 antialiased=True, vmin=zmin, vmax=zmax)
        ax2.set_xlabel('Angle (rad)', fontsize=11)
        ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
        ax2.set_zlabel(control_label, fontsize=11)
        ax2.set_title(f'LP Policy - {system_name}', fontsize=13, fontweight='bold')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=20, label=control_label)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"3D policy comparison plot saved to {save_path}")
        
        plt.show()
        
        return fig

    def create_system_video(self, initial_state, policy, policy_name, save_path=None, duration=10, fps=30, wrap_angles=False):
        """
        Create a video animation of the system (cartpole or single_pendulum) controlled by a policy
        
        Args:
            initial_state: initial state 
                          For cartpole: [x, x_dot, theta, theta_dot]
                          For single_pendulum: [theta, theta_dot]
            policy: policy function to control the system
            policy_name: name of the policy for display
            save_path: path to save the video (default: auto-generate)
            duration: duration of video in seconds (default: 10)
            fps: frames per second (default: 30)
            wrap_angles: if True, wrap angles to [-pi, pi) during simulation
            
        Returns:
            str: path to saved video file
        """
        if self.system_type == "cartpole":
            system_name = "cartpole"
        elif self.system_type == "single_pendulum":
            system_name = "single_pendulum"
        elif self.system_type == "scalar_cubic":
            system_name = "scalar_cubic"
        else:
            raise ValueError(f"Unknown system type: {self.system_type}")
        print(f"Creating {system_name} video for {policy_name}...")
        
        # Calculate simulation parameters
        total_frames = duration * fps
        simulation_steps = int(duration / self.system.delta_t)
        
        # Simulate the policy
        trajectory, success, controls = self.simulate_policy(
            policy, initial_state, horizon=simulation_steps, return_controls=True, wrap_angles=wrap_angles
        )
        
        if not success:
            print(f"Warning: Simulation failed for {policy_name}, creating video anyway")
        
        # Set up the figure and axis - vertical layout with main plot on top
        fig = plt.figure(figsize=(18, 10))
        
        # Create subplots with main plot spanning full width on top
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], width_ratios=[1, 1], 
                      hspace=0.3, wspace=0.3)
        
        ax_main = fig.add_subplot(gs[0, :])    # Main visualization (spans full width on top)
        ax_control = fig.add_subplot(gs[1, 0]) # Control plot (bottom left)
        ax_states = fig.add_subplot(gs[1, 1])  # State plots (bottom right)
        
        # Get control bounds
        u_bounds = self.extractor.u_bounds
        
        # Control plot
        control_time = np.arange(len(controls)) * self.system.delta_t
        ax_control.plot(control_time, controls, 'r-', linewidth=2)
        ax_control.set_xlabel('Time (s)')
        if self.system_type == "cartpole":
            control_label = 'Control Force (N)'
        elif self.system_type == "single_pendulum":
            control_label = 'Control Torque (N·m)'
        elif self.system_type == "scalar_cubic":
            control_label = 'Control u'
        else:
            control_label = 'Control'
        ax_control.set_ylabel(control_label)
        ax_control.set_title('Control Input')
        ax_control.grid(True, alpha=0.3)
        ax_control.axhline(y=u_bounds[0], color='gray', linestyle='--', alpha=0.5, label='Bounds')
        ax_control.axhline(y=u_bounds[1], color='gray', linestyle='--', alpha=0.5)
        ax_control.legend()
        
        # State plots
        time_axis = np.arange(len(trajectory)) * self.system.delta_t
        if self.system_type == "cartpole":
            state_names = ['Position (m)', 'Velocity (m/s)', 'Angle (rad)', 'Angular Velocity (rad/s)']
            colors = ['blue', 'green', 'orange', 'red']
            n_states_plot = 4
        elif self.system_type == "single_pendulum":
            state_names = ['Angle (rad)', 'Angular Velocity (rad/s)']
            colors = ['blue', 'green']
            n_states_plot = 2
        elif self.system_type == "scalar_cubic":
            state_names = ['State x']
            colors = ['blue']
            n_states_plot = 1
        else:
            raise ValueError(f"Unknown system type: {self.system_type}")
        
        for i in range(n_states_plot):
            ax_states.plot(time_axis, trajectory[:, i], color=colors[i], linewidth=2, 
                          alpha=0.7, label=state_names[i])
        
        ax_states.set_xlabel('Time (s)')
        ax_states.set_ylabel('State Value')
        ax_states.set_title('State Trajectories')
        ax_states.grid(True, alpha=0.3)
        ax_states.legend()
        ax_states.set_ylim(-20, 20)
        
        # Main visualization - different for each system
        if self.system_type == "cartpole":
            # Cartpole visualization
            ax_main.set_xlim(-5, 5)
            ax_main.set_ylim(-1.5, 2)
            ax_main.set_aspect('equal')
            ax_main.set_title(f'Cartpole Control - {policy_name}', fontsize=14, fontweight='bold')
            ax_main.grid(True, alpha=0.3)
            
            cart_width = 0.3
            cart_height = 0.2
            pole_length = 0.8
            
            cart = Rectangle((0, 0), cart_width, cart_height, 
                            facecolor='lightblue', edgecolor='black', linewidth=2)
            ax_main.add_patch(cart)
            
            pole_line, = ax_main.plot([], [], 'brown', linewidth=8, solid_capstyle='round')
            mass = Circle((0, 0), 0.05, facecolor='red', edgecolor='black', linewidth=2)
            ax_main.add_patch(mass)
            ax_main.axhline(y=0, color='black', linewidth=3)
            
            cart_pos_line = ax_main.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='Cart Position')
            pole_angle_line = ax_main.plot([], [], 'red', linestyle='--', alpha=0.7, label='Pole Angle')[0]
            
            def animate_cartpole(frame):
                step = min(frame * simulation_steps // total_frames, len(trajectory) - 1)
                x, x_dot, theta, theta_dot = trajectory[step]
                u = controls[step] if step < len(controls) else 0
                
                cart.set_x(x - cart_width/2)
                pole_x = x
                pole_y = cart_height
                pole_angle = theta
                
                pole_start_x = pole_x
                pole_start_y = pole_y
                pole_end_x = pole_x + pole_length * np.sin(pole_angle)
                pole_end_y = pole_y + pole_length * np.cos(pole_angle)
                
                pole_line.set_data([pole_start_x, pole_end_x], [pole_start_y, pole_end_y])
                
                mass_x = pole_x + pole_length * np.sin(pole_angle)
                mass_y = pole_y + pole_length * np.cos(pole_angle)
                mass.center = (mass_x, mass_y)
                
                cart_pos_line.set_xdata([x, x])
                pole_angle_line.set_data([pole_x, mass_x], [pole_y, mass_y])
                
                current_time = step * self.system.delta_t
                time_text.set_text(f'Time: {current_time:.2f}s')
                state_text.set_text(f'Position: {x:.3f}m\nVelocity: {x_dot:.3f}m/s\nAngle: {theta:.3f}rad\nAngular Vel: {theta_dot:.3f}rad/s')
                control_text.set_text(f'Control: {u:.3f}N')
                
                return cart, pole_line, mass, cart_pos_line, pole_angle_line, time_text, state_text, control_text
            
            animate_func = animate_cartpole
            
        elif self.system_type == "single_pendulum":
            # Single pendulum visualization
            pendulum_length = self.system.l
            ax_main.set_xlim(-1.5 * pendulum_length, 1.5 * pendulum_length)
            ax_main.set_ylim(-1.5 * pendulum_length, 1.5 * pendulum_length)
            ax_main.set_aspect('equal')
            ax_main.set_title(f'Single Pendulum Control - {policy_name}', fontsize=14, fontweight='bold')
            ax_main.grid(True, alpha=0.3)
            
            # Pivot point at origin
            pivot = Circle((0, 0), 0.05, facecolor='black', edgecolor='black', linewidth=2)
            ax_main.add_patch(pivot)
            
            # Pendulum rod (line)
            rod_line, = ax_main.plot([], [], 'brown', linewidth=4, solid_capstyle='round')
            
            # Pendulum bob (mass)
            bob = Circle((0, 0), 0.1, facecolor='red', edgecolor='black', linewidth=2)
            ax_main.add_patch(bob)
            
            # Reference line (vertical, pointing up)
            ref_line = ax_main.plot([0, 0], [0, pendulum_length], 'gray', linestyle='--', alpha=0.5, label='Upright')[0]
            
            def animate_pendulum(frame):
                step = min(frame * simulation_steps // total_frames, len(trajectory) - 1)
                theta, theta_dot = trajectory[step]
                u = controls[step] if step < len(controls) else 0
                
                # Pendulum hangs down, so angle=0 means pointing down
                # For inverted pendulum visualization, we want angle=0 to be upright
                # So we use theta directly (upright is theta=0)
                bob_x = pendulum_length * np.sin(theta)
                bob_y = pendulum_length * np.cos(theta)
                
                rod_line.set_data([0, bob_x], [0, bob_y])
                bob.center = (bob_x, bob_y)
                
                current_time = step * self.system.delta_t
                time_text.set_text(f'Time: {current_time:.2f}s')
                state_text.set_text(f'Angle: {theta:.3f}rad\nAngular Vel: {theta_dot:.3f}rad/s')
                control_text.set_text(f'Torque: {u:.3f}N·m')
                
                return rod_line, bob, time_text, state_text, control_text
            
            animate_func = animate_pendulum
        
        else:  # scalar_cubic
            # Scalar cubic visualization - simple 1D plot
            ax_main.set_xlim(-10, 10)
            ax_main.set_ylim(-0.5, 0.5)
            ax_main.set_aspect('equal')
            ax_main.set_title(f'Scalar Cubic Control - {policy_name}', fontsize=14, fontweight='bold')
            ax_main.grid(True, alpha=0.3)
            ax_main.axhline(y=0, color='black', linewidth=2)
            ax_main.axvline(x=0, color='black', linewidth=2, linestyle='--', alpha=0.5)
            
            # State marker (point on line)
            state_marker = Circle((0, 0), 0.1, facecolor='red', edgecolor='black', linewidth=2)
            ax_main.add_patch(state_marker)
            
            # State trajectory line
            state_line, = ax_main.plot([], [], 'b-', linewidth=2, label='State Trajectory')
            
            def animate_scalar_cubic(frame):
                step = min(frame * simulation_steps // total_frames, len(trajectory) - 1)
                x = trajectory[step][0]
                u = controls[step] if step < len(controls) else 0
                
                # Update state marker position
                state_marker.center = (x, 0)
                
                # Update trajectory line (show last 100 points)
                traj_window = max(0, step - 100)
                state_line.set_data(trajectory[traj_window:step+1, 0], np.zeros(step+1-traj_window))
                
                current_time = step * self.system.delta_t
                time_text.set_text(f'Time: {current_time:.2f}s')
                state_text.set_text(f'State x: {x:.3f}')
                control_text.set_text(f'Control u: {u:.3f}')
                
                return state_marker, state_line, time_text, state_text, control_text
            
            animate_func = animate_scalar_cubic
        
        # Common text elements
        time_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes, 
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        state_text = ax_main.text(0.02, 0.20, '', transform=ax_main.transAxes, 
                                 fontsize=10, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        control_text = ax_main.text(0.02, 0.30, '', transform=ax_main.transAxes, 
                                   fontsize=10, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax_main.legend(loc='upper right')
        
        # Create animation
        anim = FuncAnimation(fig, animate_func, frames=total_frames, interval=1000/fps, 
                           blit=False, repeat=True)
        
        # Save video
        if save_path is None:
            state_str = '_'.join([f'{s:.2f}' for s in initial_state])
            save_path = f"../figures/{system_name}_{policy_name.lower().replace(' ', '_')}_{state_str}.gif"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        
        print(f"Video saved to {save_path}")
        
        # Close figure to free memory
        plt.close(fig)
        
        return save_path

def train_moment_matching_policy(system_cfg, seed, N_train, M_offline, degree, regularization=DEFAULT_REGULARIZATION, 
                                 use_scaling=True, exclude_u_squared=False, use_lqr_rollouts=False):
    """
    Train moment matching policy using the same procedure as bounded_lp_sweep.py
    
    Args:
        system_cfg: system configuration dictionary from get_system_config()
        seed: random seed for data generation
        N_train: number of training samples
        M_offline: number of auxiliary samples
        degree: polynomial feature degree
        regularization: regularization parameter
        use_scaling: whether to use feature scaling
        exclude_u_squared: whether to exclude u^2 terms from polynomial features
        use_lqr_rollouts: whether to use LQR rollouts for data generation (if supported)
    
    Returns:
        mm_policy: trained moment matching policy
        poly: polynomial feature transformer
        scaler: feature scaler (if use_scaling=True)
        system: the system instance used
    """
    print(f"Training moment matching policy with seed={seed}, N={N_train}...")
    
    # Set seed for reproducibility (matches bounded_lp_sweep.py)
    np.random.seed(seed)
    
    # Build system (matches bounded_lp_sweep.py)
    extractor = PolicyExtractor(
        degree=degree,
        **system_cfg["extractor_kwargs"]
    )
    system = system_cfg["build_system"](N_train, M_offline)
    
    # Generate training data using system_cfg["sample_fn"] (matches bounded_lp_sweep.py)
    # This supports LQR rollouts if enabled and system supports it
    if use_lqr_rollouts and system_cfg.get("supports_lqr_rollouts", False):
        rollout_length = system_cfg.get("lqr_rollout_length", 500)
        x, u, x_plus, u_plus = system_cfg["sample_fn"](
            system, N_train, use_lqr_rollouts=True, rollout_length=rollout_length
        )
    else:
        x, u, x_plus, u_plus = system_cfg["sample_fn"](system, N_train)
    
    # Generate auxiliary data using system_cfg["aux_sample_fn"] (matches bounded_lp_sweep.py)
    x_aux, u_aux = system_cfg["aux_sample_fn"](system, M_offline)
    
    # Create features (matches bounded_lp_sweep.py)
    z = np.concatenate([x, u], axis=1)           # (N, dx+du)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)  # (N, dx+du)
    Z_all = np.concatenate([z, z_plus], axis=0)       # (2N, dx+du)
    
    # Compute costs (needed for both scaling and non-scaling cases)
    L_xu = system.cost(x, u)
    
    # Handle polynomial feature generation (matches bounded_lp_sweep.py exactly)
    dx = x.shape[1]  # State dimensions
    du = u.shape[1]   # Action dimensions
    
    if use_scaling:
        # Use PolynomialFeatureScaler with standard scaling (matches bounded_lp_sweep.py)
        poly_scaler = PolynomialFeatureScaler(
            degree=degree, 
            scaling_method='standard',
            dx=dx,
            du=du,
            exclude_u_squared=exclude_u_squared
        )
        P_all = poly_scaler.fit_transform(Z_all)
        print(f"  Fitted standard polynomial scaler (degree={degree}, dx={dx}, du={du}) on {len(Z_all)} samples for seed {seed}")
    else:
        # Determine dimensions from the data for exclude_u_squared (matches bounded_lp_sweep.py)
        poly_scaler = PolynomialFeatureScaler(
            degree=degree, 
            scaling_method='none',
            dx=dx if exclude_u_squared else None,
            du=du if exclude_u_squared else None,
            exclude_u_squared=exclude_u_squared
        )
        P_all = poly_scaler.fit_transform(Z_all)
        print(f"  Fitted polynomial scaler (degree={degree}) on {len(Z_all)} samples for seed {seed}")
    
    # Split polynomial features (matches bounded_lp_sweep.py)
    P_z = P_all[:N_train]
    P_z_next = P_all[N_train:]
    
    y = np.concatenate([x_aux, u_aux], axis=1)  # (M, dx+du)
    P_y = poly_scaler.transform(y)
    
    # Precompute tensors for LPs (matches bounded_lp_sweep.py)
    Mi_tensor = np.einsum('bi,bj->bij', P_z, P_z, optimize=True) - \
                gamma * np.einsum('bi,bj->bij', P_z_next, P_z_next, optimize=True)
    Py_outer = np.einsum('bi,bj->bij', P_y, P_y, optimize=True)
    
    # Solve moment matching LP (matches bounded_lp_sweep.py)
    Q_mm, mu, status_info = solve_moment_matching_Q(
        P_z, P_z_next, P_y, L_xu, N_train, M_offline, gamma,
        regularization=regularization,
        Mi_tensor=Mi_tensor, Py_outer=Py_outer
    )
    
    if Q_mm is None:
        print(f"Moment matching training failed with status: {status_info}")
        # Match bounded_lp_sweep.py behavior: no fallback, just return None
        return None, None, None, None
    
    print(f"Moment matching training status: {status_info}")
    
    # Extract policy (matches bounded_lp_sweep.py exactly)
    # extractor is already initialized above with system_cfg["extractor_kwargs"]
    mm_policy = extractor.extract_moment_matching_policy_analytical(
        system, Q_mm, poly_scaler
    )
    
    # Return poly_scaler as scaler for backward compatibility
    return mm_policy, poly_scaler.poly, poly_scaler, system

def main():
    parser = argparse.ArgumentParser(description="Analyze convergence regions of cart pole policies")
    parser.add_argument("--resolution", type=int, default=None, help="Grid resolution for both dimensions (default: system-dependent)")
    parser.add_argument("--theta_resolution", type=int, default=None, help="Grid resolution for angle (default: system-dependent)")
    parser.add_argument("--theta_dot_resolution", type=int, default=None, help="Grid resolution for angular velocity (default: system-dependent)")
    parser.add_argument("--horizon", type=int, default=5000, help="Simulation horizon (default: 5000, matches bounded_lp_sweep.py)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Convergence threshold (default: 0.05, matches policy_extraction.py)")
    parser.add_argument("--N_train", type=int, default=20000, help="Training samples (default: 2000)")
    parser.add_argument("--M_offline", type=int, default=DEFAULT_M_OFFLINE, help="Offline pool size (default: from config)")
    parser.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION, help="Regularization parameter (default: from config)")
    parser.add_argument("--use_scaling", action="store_true", default=False, help="Use feature scaling (default: False)")
    parser.add_argument("--out_json", type=str, default="../results/convergence_analysis.json", help="Output JSON file")
    parser.add_argument("--plot_png", type=str, default="../figures/convergence_analysis.pdf", help="Output plot file")
    parser.add_argument("--create_videos", action="store_true", default=False, help="Create cartpole videos (default: False)")
    parser.add_argument("--video_duration", type=int, default=30, help="Video duration in seconds (default: 10)")
    parser.add_argument("--video_fps", type=int, default=30, help="Video frames per second (default: 30)")
    parser.add_argument("--video_initial_states", nargs='+', type=float, default=None, 
                       help="Initial state for videos. For cartpole: [x, x_dot, theta, theta_dot]. For single_pendulum: [theta, theta_dot]")
    parser.add_argument("--system", type=str, default="cartpole", choices=["cartpole", "single_pendulum", "scalar_cubic"],
                       help="System type to analyze (default: cartpole)")
    parser.add_argument("--seed", type=int, default=2, help="Random seed for data generation (default: 0)")
    parser.add_argument("--exclude_u_squared", action="store_true", default=False, 
                       help="Exclude u^2 terms from polynomial features (for degree 2)")
    parser.add_argument("--use_lqr_rollouts", action="store_true", default=False,
                       help="Use LQR rollouts for data generation (if system supports it)")
    args = parser.parse_args()
    
    # Set default resolutions based on system type
    system_type = args.system
    if system_type == "single_pendulum":
        default_theta_resolution = 10
        default_theta_dot_resolution = 10
    elif system_type == "scalar_cubic":
        default_theta_resolution = None  # Not used for 1D system
        default_theta_dot_resolution = None  # Not used for 1D system
    else:  # cartpole
        default_theta_resolution = 100
        default_theta_dot_resolution = 100
    
    # Use provided resolutions or defaults
    if args.resolution is not None:
        # If --resolution is provided, use it for both dimensions
        theta_resolution = args.resolution
        theta_dot_resolution = args.resolution
    else:
        # Use individual resolutions if provided, otherwise use defaults
        theta_resolution = args.theta_resolution if args.theta_resolution is not None else default_theta_resolution
        theta_dot_resolution = args.theta_dot_resolution if args.theta_dot_resolution is not None else default_theta_dot_resolution
    
    # Update global parameters
    global grid_resolution, simulation_horizon, convergence_threshold, N_train, M_offline
    grid_resolution = theta_resolution  # Keep for backward compatibility
    simulation_horizon = args.horizon
    convergence_threshold = args.threshold
    N_train = args.N_train
    M_offline = args.M_offline
    degree = DEGREE
    regularization = args.regularization
    use_scaling = args.use_scaling

    print(f"=== {system_type.replace('_', ' ').title()} Convergence Region Analysis ===")
    print(f"System type: {system_type}")
    print(f"Seed: {args.seed}")
    print(f"Grid resolution: {theta_resolution}x{theta_dot_resolution} (theta x theta_dot)")
    print(f"Simulation horizon: {simulation_horizon}")
    print(f"Convergence threshold: {convergence_threshold}")
    print(f"Training samples: {N_train}")
    print(f"Offline pool size: {M_offline}")
    print(f"Polynomial feature degree: {degree}")
    print(f"Regularization parameter: {regularization}")
    print(f"Use scaling: {use_scaling}")
    print(f"Exclude u squared: {args.exclude_u_squared}")
    print(f"Use LQR rollouts: {args.use_lqr_rollouts}")
    print()
    
    # Get system configuration (matches bounded_lp_sweep.py)
    system_cfg = get_system_config(system_type)
    
    # Check if LQR rollouts are supported
    if args.use_lqr_rollouts and not system_cfg.get("supports_lqr_rollouts", False):
        print(f"LQR rollout sampling not available for {system_cfg['pretty_name']}. Using purely random samples.")
        use_lqr_rollouts = False
    else:
        use_lqr_rollouts = args.use_lqr_rollouts
    
    # Set grid bounds based on system type
    if system_type == "cartpole":
        theta_grid_bounds_local = THETA_BOUNDS
        theta_dot_grid_bounds_local = THETA_DOT_BOUNDS
        x_grid_bounds_local = None
    elif system_type == "single_pendulum":
        theta_grid_bounds_local = THETA_BOUNDS_SINGLE_PENDULUM_TEST
        theta_dot_grid_bounds_local = THETA_DOT_BOUNDS_SINGLE_PENDULUM_TEST
        x_grid_bounds_local = None
    elif system_type == "scalar_cubic":
        theta_grid_bounds_local = None
        theta_dot_grid_bounds_local = None
        x_grid_bounds_local = (-5.0, 5.0)  # Default range for scalar_cubic
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    # Train moment matching policy using same procedure as bounded_lp_sweep.py
    try:
        result = train_moment_matching_policy(
            system_cfg, args.seed, N_train, M_offline, degree, 
            regularization=regularization, 
            use_scaling=use_scaling, 
            exclude_u_squared=args.exclude_u_squared,
            use_lqr_rollouts=use_lqr_rollouts
        )
        if result[0] is None:
            print("✗ Moment matching training failed: Q_mm is None")
            return
        mm_policy, poly, scaler, system = result
        print("✓ Moment matching policy trained successfully")
        
        # Check value function positivity if policy was successfully trained
        if mm_policy is not None:
            print("\n=== Checking Value Function Positivity ===")
            # Note: We would need to store Q_mm from training to check positivity
            # For now, we'll skip this check as it requires access to the Q matrix
            print("Value function positivity check skipped (requires Q matrix access)")
    except Exception as e:
        print(f"✗ Moment matching training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Extract LQR policy (matches bounded_lp_sweep.py)
    try:
        # extractor was already created in train_moment_matching_policy, but we need it here
        # Create extractor with same parameters as used in training
        extractor = PolicyExtractor(
            degree=degree,
            **system_cfg["extractor_kwargs"]
        )
        lqr_policy, K_lqr, P_lqr = extractor.extract_lqr_policy(system)
        print("✓ LQR policy extracted successfully")
    except Exception as e:
        print(f"✗ LQR policy extraction failed: {e}")
        return
    
    # Create analyzer
    analyzer = ConvergenceAnalyzer(system, extractor, mm_policy, lqr_policy, convergence_threshold, system_type=system_type)

    # Create 2D/3D policy comparison plots (only for 2D systems)
    if system_type != "scalar_cubic":
        # Create 2D policy comparison plot
        print("\n=== Creating 2D Policy Comparison Plot ===")
        policy_2d_plot_path = args.plot_png.replace('.pdf', '_policy_comparison_2d.pdf')
        if system_type == "single_pendulum":
            fig_policy_2d = analyzer.plot_policy_comparison_2d(
                THETA_BOUNDS_SINGLE_PENDULUM, THETA_DOT_BOUNDS_SINGLE_PENDULUM,
                theta_resolution=theta_resolution, 
                theta_dot_resolution=theta_dot_resolution,
                save_path=policy_2d_plot_path
            )
        else:  # cartpole
            fig_policy_2d = analyzer.plot_policy_comparison_2d(
                THETA_BOUNDS, THETA_DOT_BOUNDS,
                theta_resolution=theta_resolution, 
                theta_dot_resolution=theta_dot_resolution,
                save_path=policy_2d_plot_path
            )
        print(f"2D policy comparison plot saved to {policy_2d_plot_path}")
        
        # Create 3D policy comparison plot
        print("\n=== Creating 3D Policy Comparison Plot ===")
        policy_3d_plot_path = args.plot_png.replace('.pdf', '_policy_comparison_3d.pdf')
        if system_type == "single_pendulum":
            fig_policy_3d = analyzer.plot_policy_comparison_3d(
                THETA_BOUNDS_SINGLE_PENDULUM, THETA_DOT_BOUNDS_SINGLE_PENDULUM,
                theta_resolution=theta_resolution, 
                theta_dot_resolution=theta_dot_resolution,
                save_path=policy_3d_plot_path
            )
        else:  # cartpole
            fig_policy_3d = analyzer.plot_policy_comparison_3d(
                THETA_BOUNDS, THETA_DOT_BOUNDS,
                theta_resolution=theta_resolution, 
                theta_dot_resolution=theta_dot_resolution,
                save_path=policy_3d_plot_path
            )
        print(f"3D policy comparison plot saved to {policy_3d_plot_path}")
    
    # Run convergence analysis
    print("\nStarting convergence analysis...")
    start_time = time.time()
    
    try:
        if system_type == "scalar_cubic":
            results = analyzer.analyze_convergence_grid(
                x_range=x_grid_bounds_local,
                x_resolution=theta_resolution if theta_resolution is not None else 100
            )
        else:
            results = analyzer.analyze_convergence_grid(
                theta_grid_bounds_local, theta_dot_grid_bounds_local, 
                theta_resolution=theta_resolution, 
                theta_dot_resolution=theta_dot_resolution
            )
        
        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Convergence analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print statistics
    stats = results['statistics']
    print("\n=== Convergence Statistics ===")
    print(f"Moment Matching Convergence Rate: {stats['mm_convergence_rate']:.3f}")
    print(f"LQR Convergence Rate: {stats['lqr_convergence_rate']:.3f}")
    print(f"Both Policies Converge: {stats['both_converge_rate']:.3f}")
    print(f"Either Policy Converges: {stats['either_converges_rate']:.3f}")
    print(f"Only MM Converges: {stats['only_mm_rate']:.3f}")
    print(f"Only LQR Converges: {stats['only_lqr_rate']:.3f}")
    
    # Save results
    # Convert numpy arrays to lists for JSON serialization
    if system_type == "scalar_cubic":
        results_json = {
            'x_vals': results['x_vals'].tolist(),
            'mm_convergence': results['mm_convergence'].tolist(),
            'lqr_convergence': results['lqr_convergence'].tolist(),
            'mm_success': results['mm_success'].tolist(),
            'lqr_success': results['lqr_success'].tolist(),
            'both_converge': results['both_converge'].tolist(),
            'either_converges': results['either_converges'].tolist(),
            'only_mm': results['only_mm'].tolist(),
            'only_lqr': results['only_lqr'].tolist(),
            'statistics': stats,
            'parameters': {
                'grid_resolution': theta_resolution if theta_resolution is not None else 100,
                'simulation_horizon': simulation_horizon,
                'convergence_threshold': convergence_threshold,
                'N_train': N_train,
                'M_offline': M_offline,
                'degree': degree,
                'regularization': regularization,
                'use_scaling': use_scaling,
                'x_grid_bounds': x_grid_bounds_local,
                'system_type': system_type,
            }
        }
    else:
        results_json = {
            'theta_vals': results['theta_vals'].tolist(),
            'theta_dot_vals': results['theta_dot_vals'].tolist(),
            'mm_convergence': results['mm_convergence'].tolist(),
            'lqr_convergence': results['lqr_convergence'].tolist(),
            'mm_success': results['mm_success'].tolist(),
            'lqr_success': results['lqr_success'].tolist(),
            'both_converge': results['both_converge'].tolist(),
            'either_converges': results['either_converges'].tolist(),
            'only_mm': results['only_mm'].tolist(),
            'only_lqr': results['only_lqr'].tolist(),
            'statistics': stats,
            'parameters': {
                'grid_resolution': grid_resolution,
                'simulation_horizon': simulation_horizon,
                'convergence_threshold': convergence_threshold,
                'N_train': N_train,
                'M_offline': M_offline,
                'degree': degree,
                'regularization': regularization,
                'use_scaling': use_scaling,
                'theta_grid_bounds': theta_grid_bounds_local,
                'theta_dot_grid_bounds': theta_dot_grid_bounds_local,
                'system_type': system_type,
            }
        }
    
    with open(args.out_json, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {args.out_json}")
    
    # Create and save plot
    fig = analyzer.plot_convergence_regions(results, args.plot_png)
    print(f"Plot saved to {args.plot_png}")
    
    # Create trajectory analysis plot
    trajectory_plot_path = args.plot_png.replace('.pdf', '_trajectories.pdf')
    fig_traj = analyzer.plot_trajectory_analysis(results, trajectory_plot_path)
    print(f"Trajectory analysis plot saved to {trajectory_plot_path}")
    
    # Create videos if requested
    if args.create_videos:
        print("\n=== Creating Cartpole Videos ===")
        
        # Parse initial state based on system type
        if args.video_initial_states is None:
            # Use defaults based on system type
            if system_type == "cartpole":
                initial_state = np.array([0.0, 0.0, 0.1, 0.0])
            elif system_type == "single_pendulum":
                initial_state = np.array([0.1, 0.0])
            elif system_type == "scalar_cubic":
                initial_state = np.array([0.5])
            else:
                raise ValueError(f"Unknown system type: {system_type}")
        else:
            if system_type == "cartpole" and len(args.video_initial_states) != 4:
                print("Warning: For cartpole, video_initial_states must have exactly 4 values [x, x_dot, theta, theta_dot]")
                print("Using default initial state: [0, 0, 0.1, 0]")
                initial_state = np.array([0.0, 0.0, 0.1, 0.0])
            elif system_type == "single_pendulum" and len(args.video_initial_states) != 2:
                print("Warning: For single_pendulum, video_initial_states must have exactly 2 values [theta, theta_dot]")
                print("Using default initial state: [0.1, 0.0]")
                initial_state = np.array([0.1, 0.0])
            elif system_type == "scalar_cubic" and len(args.video_initial_states) != 1:
                print("Warning: For scalar_cubic, video_initial_states must have exactly 1 value [x]")
                print("Using default initial state: [0.5]")
                initial_state = np.array([0.5])
            else:
                initial_state = np.array(args.video_initial_states)
        
        print(f"Initial state for videos: {initial_state}")
        
        # Create LQR video (without angle wrapping)
        try:
            lqr_video_path = analyzer.create_system_video(
                initial_state, lqr_policy, "LQR Policy", 
                duration=args.video_duration, fps=args.video_fps, wrap_angles=False
            )
            print(f"✓ LQR video created: {lqr_video_path}")
        except Exception as e:
            print(f"✗ LQR video creation failed: {e}")
        
        # Create Moment Matching video (with angle wrapping)
        try:
            mm_video_path = analyzer.create_system_video(
                initial_state, mm_policy, "LP Policy", 
                duration=args.video_duration, fps=args.video_fps, wrap_angles=True
            )
            print(f"✓ Moment Matching video created: {mm_video_path}")
        except Exception as e:
            print(f"✗ Moment Matching video creation failed: {e}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
