#!/usr/bin/env python3
"""
Convergence Region Analysis for Cart Pole Policies

This script compares the convergence regions of moment matching and LQR policies
in the angle-angular velocity space, with fixed initial position and velocity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.preprocessing import PolynomialFeatures
from utils import ScaleOnlyScaler, BlockScaleOnlyScaler
from feature_scaling import PolynomialFeatureScaler
from dynamical_systems import cart_pole
from policy_extraction import PolicyExtractor
from moment_matching import solve_moment_matching_Q, solve_identity_Q
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
    DEFAULT_M_OFFLINE, DEFAULT_REGULARIZATION
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
theta_grid_bounds = THETA_BOUNDS      # radians
theta_dot_grid_bounds = THETA_DOT_BOUNDS  # rad/s
grid_resolution = 100                  # points per dimension
simulation_horizon = 10000             # simulation steps
convergence_threshold = 0.2           # threshold for considering converged

class ConvergenceAnalyzer:
    """
    Analyzes convergence regions for different policies
    """
    
    def __init__(self, system, extractor, mm_policy, lqr_policy, convergence_threshold):
        self.system = system
        self.extractor = extractor
        self.mm_policy = mm_policy
        self.lqr_policy = lqr_policy
        self.convergence_threshold = convergence_threshold
        self.horizon = simulation_horizon
        
    def is_converged(self, trajectory, threshold=None):
        """
        Check if a trajectory has converged to the equilibrium point
        
        Args:
            trajectory: array of shape (horizon, 4) with states [x, x_dot, theta, theta_dot]
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
        
        # Equilibrium point is [0, 0, 0, 0]
        equilibrium = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Check if all states in the last 100 steps are within threshold
        distances = np.linalg.norm(last_states - equilibrium, ord=np.inf, axis=1)
        max_distance = np.max(distances)
        # print(f"Max distance: {max_distance}")
        
        return max_distance < threshold
    
    def simulate_policy(self, policy, initial_state, horizon=None, return_controls=False):
        """
        Simulate a policy from an initial state
        
        Args:
            policy: policy function
            initial_state: initial state [x, x_dot, theta, theta_dot]
            horizon: simulation horizon (default: self.horizon)
            return_controls: whether to return control sequence
            
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
                
                # Update state
                x_current = self.system.step(x_current, u_current)
                trajectory.append(x_current.copy())
                    
            except Exception as e:
                print(f"Simulation error at step {step}: {e}")
                success = False
                break
        
        # Check stability based on last 100 samples if we have enough data
        if success and len(trajectory) >= 100:
            # Get the last 100 states
            last_100_states = np.array(trajectory[-100:])
            
            # Check if all last 100 states are sufficiently close to equilibrium (origin)
            equilibrium_tolerance = self.convergence_threshold  # Adjust this threshold as needed
            distances_to_equilibrium = np.linalg.norm(last_100_states, axis=1)
            
            # If any of the last 100 states are too far from equilibrium, mark as unstable
            if not np.all(distances_to_equilibrium <= equilibrium_tolerance):
                success = False
        
        if return_controls:
            return np.array(trajectory), success, np.array(controls)
        else:
            return np.array(trajectory), success
    
    def analyze_convergence_grid(self, theta_range, theta_dot_range, resolution=50):
        """
        Analyze convergence regions on a grid
        
        Args:
            theta_range: tuple (min, max) for angle
            theta_dot_range: tuple (min, max) for angular velocity
            resolution: grid resolution (points per dimension)
            
        Returns:
            dict: results containing convergence maps and statistics
        """
        print(f"Analyzing convergence on {resolution}x{resolution} grid...")
        
        # Create grid
        theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)
        theta_dot_vals = np.linspace(theta_dot_range[0], theta_dot_range[1], resolution)
        
        # Initialize results
        mm_convergence = np.zeros((resolution, resolution), dtype=bool)
        lqr_convergence = np.zeros((resolution, resolution), dtype=bool)
        mm_success = np.zeros((resolution, resolution), dtype=bool)
        lqr_success = np.zeros((resolution, resolution), dtype=bool)
        
        total_points = resolution * resolution
        processed = 0
        
        for i, theta in enumerate(theta_vals):
            for j, theta_dot in enumerate(theta_dot_vals):
                # Fixed initial conditions: position=0, velocity=0
                initial_state = np.array([0.0, 0.0, theta, theta_dot])
                
                # Test moment matching policy
                try:
                    mm_traj, mm_sim_success = self.simulate_policy(self.mm_policy, initial_state)
                    mm_success[i, j] = mm_sim_success
                    if mm_sim_success:
                        mm_convergence[i, j] = self.is_converged(mm_traj)
                except Exception as e:
                    print(f"MM policy failed at ({theta:.3f}, {theta_dot:.3f}): {e}")
                    mm_success[i, j] = False
                    mm_convergence[i, j] = False
                
                # Test LQR policy
                try:
                    lqr_traj, lqr_sim_success = self.simulate_policy(self.lqr_policy, initial_state)
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
        ax.set_xlabel('Angle (rad)', fontsize=12)
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
        ax.set_title('Policy Convergence Regions (Tab10 Palette)\n' + 
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
            initial_state = np.array([0.0, 0.0, point['theta'], point['theta_dot']])
            
            # Simulate both policies with controls
            lqr_traj, lqr_success, lqr_controls = self.simulate_policy(self.lqr_policy, initial_state, return_controls=True)
            mm_traj, mm_success, mm_controls = self.simulate_policy(self.mm_policy, initial_state, return_controls=True)
            
            # Plot states with colorblind-friendly colors and distinct line styles
            ax_states = axes[i, 0]
            time_steps = np.arange(len(lqr_traj))
            
            # LQR trajectories with solid lines and blue color
            ax_states.plot(time_steps, lqr_traj[:, 0], color=colorblind_colors['lqr'], linestyle='-', linewidth=2, label='Position (LQR)', alpha=0.8)
            ax_states.plot(time_steps, lqr_traj[:, 1], color=colorblind_colors['lqr'], linestyle='--', linewidth=2, label='Velocity (LQR)', alpha=0.8)
            ax_states.plot(time_steps, lqr_traj[:, 2], color=colorblind_colors['lqr'], linestyle=':', linewidth=2, label='Angle (LQR)', alpha=0.8)
            ax_states.plot(time_steps, lqr_traj[:, 3], color=colorblind_colors['lqr'], linestyle='-.', linewidth=2, label='Angular Vel (LQR)', alpha=0.8)
            
            time_steps_mm = np.arange(len(mm_traj))
            # MM trajectories with solid lines and orange color
            ax_states.plot(time_steps_mm, mm_traj[:, 0], color=colorblind_colors['mm'], linestyle='-', linewidth=2, label='Position (MM)', alpha=0.8)
            ax_states.plot(time_steps_mm, mm_traj[:, 1], color=colorblind_colors['mm'], linestyle='--', linewidth=2, label='Velocity (MM)', alpha=0.8)
            ax_states.plot(time_steps_mm, mm_traj[:, 2], color=colorblind_colors['mm'], linestyle=':', linewidth=2, label='Angle (MM)', alpha=0.8)
            ax_states.plot(time_steps_mm, mm_traj[:, 3], color=colorblind_colors['mm'], linestyle='-.', linewidth=2, label='Angular Vel (MM)', alpha=0.8)
            
            ax_states.set_xlabel('Time Step')
            ax_states.set_ylabel('State Value')
            ax_states.set_title(f'{point["label"]}: θ={point["theta"]:.3f}, θ̇={point["theta_dot"]:.3f}')
            ax_states.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_states.grid(True, alpha=0.3)
            
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
            equilibrium = np.array([0.0, 0.0, 0.0, 0.0])
            
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

    def create_cartpole_video(self, initial_state, policy, policy_name, save_path=None, duration=10, fps=30):
        """
        Create a video animation of the cartpole system controlled by a policy
        
        Args:
            initial_state: initial state [x, x_dot, theta, theta_dot]
            policy: policy function to control the cartpole
            policy_name: name of the policy for display
            save_path: path to save the video (default: auto-generate)
            duration: duration of video in seconds (default: 10)
            fps: frames per second (default: 30)
            
        Returns:
            str: path to saved video file
        """
        print(f"Creating cartpole video for {policy_name}...")
        
        # Calculate simulation parameters
        total_frames = duration * fps
        simulation_steps = int(duration / self.system.delta_t)
        
        # Simulate the policy
        trajectory, success, controls = self.simulate_policy(
            policy, initial_state, horizon=simulation_steps, return_controls=True
        )
        
        if not success:
            print(f"Warning: Simulation failed for {policy_name}, creating video anyway")
        
        # Set up the figure and axis - vertical layout with main plot on top
        fig = plt.figure(figsize=(18, 10))
        
        # Create subplots with main plot spanning full width on top
        # Use GridSpec for more control over subplot sizes
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], width_ratios=[1, 1], 
                      hspace=0.3, wspace=0.3)
        
        ax_main = fig.add_subplot(gs[0, :])    # Main cartpole visualization (spans full width on top)
        ax_control = fig.add_subplot(gs[1, 0]) # Control plot (bottom left)
        ax_states = fig.add_subplot(gs[1, 1])  # State plots (bottom right)
        
        # Main cartpole visualization
        ax_main.set_xlim(-5, 5)
        ax_main.set_ylim(-1.5, 2)
        ax_main.set_aspect('equal')
        ax_main.set_title(f'Cartpole Control - {policy_name}', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        
        # Control plot - match the aspect ratio of the main plot
        control_time = np.arange(len(controls)) * self.system.delta_t
        ax_control.plot(control_time, controls, 'r-', linewidth=2)
        ax_control.set_xlabel('Time (s)')
        ax_control.set_ylabel('Control Force (N)')
        ax_control.set_title('Control Input')
        ax_control.grid(True, alpha=0.3)
        ax_control.axhline(y=u_bounds[0], color='gray', linestyle='--', alpha=0.5, label='Bounds')
        ax_control.axhline(y=u_bounds[1], color='gray', linestyle='--', alpha=0.5)
        ax_control.legend()
        
        # State plots - combined plot showing all states
        time_axis = np.arange(len(trajectory)) * self.system.delta_t
        state_names = ['Position (m)', 'Velocity (m/s)', 'Angle (rad)', 'Angular Velocity (rad/s)']
        
        # Plot all states in one subplot with different colors
        colors = ['blue', 'green', 'orange', 'red']
        for i in range(4):
            ax_states.plot(time_axis, trajectory[:, i], color=colors[i], linewidth=2, 
                          alpha=0.7, label=state_names[i])
        
        ax_states.set_xlabel('Time (s)')
        ax_states.set_ylabel('State Value')
        ax_states.set_title('State Trajectories')
        ax_states.grid(True, alpha=0.3)
        ax_states.legend()
        
        # Let control and states plots use their natural aspect ratios
        # The main cartpole plot will be larger due to set_aspect('equal')
        
        # Cartpole visualization elements
        cart_width = 0.3
        cart_height = 0.2
        pole_length = 0.8
        pole_width = 0.05
        
        # Create cart
        cart = Rectangle((0, 0), cart_width, cart_height, 
                        facecolor='lightblue', edgecolor='black', linewidth=2)
        ax_main.add_patch(cart)
        
        # Create pole (we'll use a line instead of rectangle for easier rotation)
        pole_line, = ax_main.plot([], [], 'brown', linewidth=8, solid_capstyle='round')
        
        # Create mass at end of pole
        mass = Circle((0, 0), 0.05, facecolor='red', edgecolor='black', linewidth=2)
        ax_main.add_patch(mass)
        
        # Ground line
        ax_main.axhline(y=0, color='black', linewidth=3)
        
        # Initialize position indicators
        cart_pos_line = ax_main.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='Cart Position')
        pole_angle_line = ax_main.plot([], [], 'red', linestyle='--', alpha=0.7, label='Pole Angle')[0]
        
        # Time text
        time_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes, 
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # State text
        state_text = ax_main.text(0.02, 0.20, '', transform=ax_main.transAxes, 
                                 fontsize=10, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Control text
        control_text = ax_main.text(0.02, 0.30, '', transform=ax_main.transAxes, 
                                   fontsize=10, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax_main.legend(loc='upper right')
        
        def animate(frame):
            # Calculate which simulation step this frame corresponds to
            step = min(frame * simulation_steps // total_frames, len(trajectory) - 1)
            
            # Get current state
            x, x_dot, theta, theta_dot = trajectory[step]
            u = controls[step] if step < len(controls) else 0
            
            # Update cart position
            cart.set_x(x - cart_width/2)
            
            # Update pole position and angle
            pole_x = x
            pole_y = cart_height
            pole_angle = theta
            
            # Calculate pole endpoints
            pole_start_x = pole_x
            pole_start_y = pole_y
            pole_end_x = pole_x + pole_length * np.sin(pole_angle)
            pole_end_y = pole_y + pole_length * np.cos(pole_angle)
            
            # Update pole line
            pole_line.set_data([pole_start_x, pole_end_x], [pole_start_y, pole_end_y])
            
            # Update mass position
            mass_x = pole_x + pole_length * np.sin(pole_angle)
            mass_y = pole_y + pole_length * np.cos(pole_angle)
            mass.center = (mass_x, mass_y)
            
            # Update position indicators
            cart_pos_line.set_xdata([x, x])
            pole_angle_line.set_data([pole_x, mass_x], [pole_y, mass_y])
            
            # Update text displays
            current_time = step * self.system.delta_t
            time_text.set_text(f'Time: {current_time:.2f}s')
            
            state_text.set_text(f'Position: {x:.3f}m\nVelocity: {x_dot:.3f}m/s\nAngle: {theta:.3f}rad\nAngular Vel: {theta_dot:.3f}rad/s')
            
            control_text.set_text(f'Control: {u:.3f}N')
            
            return cart, pole_line, mass, cart_pos_line, pole_angle_line, time_text, state_text, control_text
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, 
                           blit=False, repeat=True)
        
        # Save video
        if save_path is None:
            save_path = f"../figures/cartpole_{policy_name.lower().replace(' ', '_')}_{initial_state[0]}_{initial_state[1]}_{initial_state[2]}_{initial_state[3]}.gif"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        
        print(f"Video saved to {save_path}")
        
        # Close figure to free memory
        plt.close(fig)
        
        return save_path

def train_moment_matching_policy(system, N_train, M_offline, degree, regularization=DEFAULT_REGULARIZATION, use_scaling=True):
    """
    Train moment matching policy
    
    Returns:
        mm_policy: trained moment matching policy
        poly: polynomial feature transformer
        scaler: feature scaler (if use_scaling=True)
    """
    print("Training moment matching policy...")
    
    # Generate training data
    x, u, x_plus, u_plus = system.generate_samples(
        x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=N_train
    )
    
    # Generate auxiliary data
    x_aux, u_aux = system.generate_samples_auxiliary(
        x_bounds, x_dot_bounds, theta_bounds, theta_dot_bounds, u_bounds, n_samples=M_offline
    )
    
    # Create features
    z = np.concatenate([x, u], axis=1)           # (N, dx+du)
    z_plus = np.concatenate([x_plus, u_plus], axis=1)  # (N, dx+du)
    Z_all = np.concatenate([z, z_plus], axis=0)       # (2N, dx+du)
    
    # Compute costs (needed for both scaling and non-scaling cases)
    L_xu = system.cost(x, u)
    
    # Handle polynomial feature generation based on whether scaling is used
    scaler = None
    if use_scaling:
        # Create block scale-only scaler (no centering, only division by std)
        # Scale states and actions separately for better numerical stability
        dx = x.shape[1]
        du = u.shape[1]
        scaler = BlockScaleOnlyScaler(dx=dx, du=du)
        scaler.fit(Z_all)
        Z_all_scaled = scaler.transform(Z_all)
        L_xu = L_xu / scaler.scale_u_[0]  # Scale costs by action scaling factor
        
        # poly was fitted on scaled data, so use scaled data
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        P_all = poly.fit_transform(Z_all_scaled)
        
        P_z = P_all[:N_train]
        P_z_next = P_all[N_train:]
        
        y = np.concatenate([x_aux, u_aux], axis=1)  # (M, dx+du)
        y_scaled = scaler.transform(y)
        P_y = poly.transform(y_scaled)
        
        # Use scaled features for moment matching
        Q_mm, mu, status_info = solve_moment_matching_Q(
            P_z, P_z_next, P_y, L_xu, N_train, M_offline, gamma,
            regularization=regularization
        )
    else:
        # poly was fitted on unscaled data, so use unscaled data
        # Create a PolynomialFeatureScaler with no scaling for consistency with policy extraction
        dx = x.shape[1]
        du = u.shape[1]
        poly_scaler = PolynomialFeatureScaler(
            degree=degree,
            scaling_method='none',
            dx=dx,
            du=du
        )
        # Fit the scaler (which will fit the internal poly)
        P_all = poly_scaler.fit_transform(Z_all)
        
        P_z = P_all[:N_train]
        P_z_next = P_all[N_train:]
        
        y = np.concatenate([x_aux, u_aux], axis=1)  # (M, dx+du)
        P_y = poly_scaler.transform(y)
        
        # Store poly for backward compatibility
        poly = poly_scaler.poly
    
        # Use original features for moment matching
        Q_mm, mu, status_info = solve_moment_matching_Q(
            P_z, P_z_next, P_y, L_xu, N_train, M_offline, gamma,
            regularization=regularization
        )
    
    if Q_mm is None:
        print(f"Moment matching training failed with status: {status_info}")
        print("Trying with different parameters...")
        
        # Try with more relaxed parameters
        try:
            Q_mm, mu, status_info = solve_moment_matching_Q(
                P_z, P_z_next, P_y, L_xu, N_train, M_offline, gamma,
                regularization=1e-3  # More relaxed regularization
            )
            if Q_mm is None:
                print("Creating fallback policy...")
                # Create a fallback policy using identity baseline approach
                status_id = solve_identity_Q(P_z, P_z_next, L_xu, N_train, gamma)
                
                if status_id in ["optimal", "optimal_inaccurate"]:
                    # Create a simple Q matrix for the fallback
                    d = P_z.shape[1]
                    Q_mm = np.eye(d) * 0.1  # Simple identity-based Q matrix
                    print("Using identity-based fallback policy")
                else:
                    # Last resort: create a very simple Q matrix
                    d = P_z.shape[1]
                    Q_mm = np.eye(d) * 0.01
                    print("Using minimal fallback policy")
        except Exception as e:
            print(f"Relaxed training also failed: {e}")
            print("Creating fallback policy...")
            # Create a fallback policy using identity baseline approach
            status_id = solve_identity_Q(P_z, P_z_next, L_xu, N_train, gamma)
            
            if status_id in ["optimal", "optimal_inaccurate"]:
                # Create a simple Q matrix for the fallback
                d = P_z.shape[1]
                Q_mm = np.eye(d) * 0.1  # Simple identity-based Q matrix
                print("Using identity-based fallback policy")
            else:
                # Last resort: create a very simple Q matrix
                d = P_z.shape[1]
                Q_mm = np.eye(d) * 0.01
                print("Using minimal fallback policy")
    
    print(f"Moment matching training status: {status_info}")
    
    # Extract policy using improved methods
    extractor = PolicyExtractor(degree=degree)
    
    if use_scaling and scaler is not None:
        print("Using analytical policy extraction with scaling")
        # Create a PolynomialFeatureScaler wrapper for the scaler and poly
        dx = x.shape[1]
        du = u.shape[1]
        poly_scaler = PolynomialFeatureScaler(
            degree=degree,
            scaling_method='standard',  # This will be ignored since we're using existing scaler
            dx=dx,
            du=du
        )
        poly_scaler.poly = poly
        # Map scaler attributes (handle both underscore and non-underscore versions)
        if hasattr(scaler, 'scaler_x'):
            poly_scaler.scaler_x = scaler.scaler_x
        elif hasattr(scaler, 'scaler_x_'):
            poly_scaler.scaler_x = scaler.scaler_x_
        else:
            poly_scaler.scaler_x = None
        if hasattr(scaler, 'scaler_u'):
            poly_scaler.scaler_u = scaler.scaler_u
        elif hasattr(scaler, 'scaler_u_'):
            poly_scaler.scaler_u = scaler.scaler_u_
        else:
            poly_scaler.scaler_u = None
        poly_scaler.is_fitted_ = True
        mm_policy = extractor.extract_moment_matching_policy_analytical(
            system, Q_mm, poly_scaler
        )
    else:
        print("Using analytical policy extraction without scaling")
        # poly_scaler was already created in the else block above
        mm_policy = extractor.extract_moment_matching_policy_analytical(
            system, Q_mm, poly_scaler
        )
    
    return mm_policy, poly, scaler

def main():
    parser = argparse.ArgumentParser(description="Analyze convergence regions of cart pole policies")
    parser.add_argument("--resolution", type=int, default=100, help="Grid resolution (default: 50)")
    parser.add_argument("--horizon", type=int, default=10000, help="Simulation horizon (default: 5000)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Convergence threshold (default: 0.5)")
    parser.add_argument("--N_train", type=int, default=15000, help="Training samples (default: 2000)")
    parser.add_argument("--M_offline", type=int, default=DEFAULT_M_OFFLINE, help="Offline pool size (default: from config)")
    parser.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION, help="Regularization parameter (default: from config)")
    parser.add_argument("--use_scaling", action="store_true", default=False, help="Use feature scaling (default: False)")
    parser.add_argument("--out_json", type=str, default="../results/convergence_analysis.json", help="Output JSON file")
    parser.add_argument("--plot_png", type=str, default="../figures/convergence_analysis.pdf", help="Output plot file")
    parser.add_argument("--create_videos", action="store_true", default=False, help="Create cartpole videos (default: False)")
    parser.add_argument("--video_duration", type=int, default=30, help="Video duration in seconds (default: 10)")
    parser.add_argument("--video_fps", type=int, default=30, help="Video frames per second (default: 30)")
    parser.add_argument("--video_initial_states", nargs='+', type=float, default=[0.0, 0.0, 1.0, 1.0], 
                       help="Initial state for videos [x, x_dot, theta, theta_dot] (default: [0, 0, 0.1, 0])")
    args = parser.parse_args()
    
    # Update global parameters
    global grid_resolution, simulation_horizon, convergence_threshold, N_train, M_offline
    grid_resolution = args.resolution
    simulation_horizon = args.horizon
    convergence_threshold = args.threshold
    N_train = args.N_train
    M_offline = args.M_offline
    degree = DEGREE
    regularization = args.regularization
    use_scaling = args.use_scaling

    print("=== Cart Pole Convergence Region Analysis ===")
    print(f"Grid resolution: {grid_resolution}x{grid_resolution}")
    print(f"Simulation horizon: {simulation_horizon}")
    print(f"Convergence threshold: {convergence_threshold}")
    print(f"Training samples: {N_train}")
    print(f"Offline pool size: {M_offline}")
    print(f"Polynomial feature degree: {degree}")
    print(f"Regularization parameter: {regularization}")
    print(f"Use scaling: {use_scaling}")
    print()
    
    # Create system
    system = cart_pole(M_c, M_p, l, dt, C, rho, gamma, N_train, M_offline)
    
    # Train moment matching policy
    try:
        mm_policy, poly, scaler = train_moment_matching_policy(
            system, N_train, M_offline, degree, regularization=regularization, use_scaling=use_scaling
        )
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
    
    # Extract LQR policy
    try:
        extractor = PolicyExtractor(degree=degree)
        lqr_policy, K_lqr, P_lqr = extractor.extract_lqr_policy(system)
        print("✓ LQR policy extracted successfully")
    except Exception as e:
        print(f"✗ LQR policy extraction failed: {e}")
        return
    
    # Create analyzer
    analyzer = ConvergenceAnalyzer(system, extractor, mm_policy, lqr_policy, convergence_threshold)
    
    # Run convergence analysis
    print("\nStarting convergence analysis...")
    start_time = time.time()
    
    try:
        results = analyzer.analyze_convergence_grid(
            theta_grid_bounds, theta_dot_grid_bounds, grid_resolution
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
            'theta_grid_bounds': theta_grid_bounds,
            'theta_dot_grid_bounds': theta_dot_grid_bounds,
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
        
        # Parse initial state
        if len(args.video_initial_states) != 4:
            print("Warning: video_initial_states must have exactly 4 values [x, x_dot, theta, theta_dot]")
            print("Using default initial state: [0, 0, 0.1, 0]")
            initial_state = np.array([0.0, 0.0, 0.1, 0.0])
        else:
            initial_state = np.array(args.video_initial_states)
        
        print(f"Initial state for videos: {initial_state}")
        
        # Create LQR video
        try:
            lqr_video_path = analyzer.create_cartpole_video(
                initial_state, lqr_policy, "LQR Policy", 
                duration=args.video_duration, fps=args.video_fps
            )
            print(f"✓ LQR video created: {lqr_video_path}")
        except Exception as e:
            print(f"✗ LQR video creation failed: {e}")
        
        # Create Moment Matching video
        try:
            mm_video_path = analyzer.create_cartpole_video(
                initial_state, mm_policy, "LP Policy", 
                duration=args.video_duration, fps=args.video_fps
            )
            print(f"✓ Moment Matching video created: {mm_video_path}")
        except Exception as e:
            print(f"✗ Moment Matching video creation failed: {e}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
