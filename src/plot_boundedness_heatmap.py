#!/usr/bin/env python3
"""
Create 2D color plots showing boundedness percentages for moment matching and identity covariance
as a function of dx (state dimension) and N (number of samples).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def load_percentages_data():
    """Load all bounded_vs_dim_percentages_N_*.json files and extract data."""
    src_dir = Path(__file__).parent
    
    # N values to look for (250, 500, 750, ..., 2750)
    N_values = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    
    # Also check for other N values that might exist
    all_files = list(src_dir.glob("bounded_vs_dim_percentages_N_*.json"))
    found_N_values = []
    for f in all_files:
        try:
            N = int(f.stem.split("_N_")[1])
            if np.min(N_values) <= N <= np.max(N_values):
                if N not in found_N_values:
                    found_N_values.append(N)
        except:
            continue
    
    # Use found N values, sorted
    N_values = sorted(found_N_values)
    print(f"Found N values: {N_values}")
    
    # Collect data: {N: {dx: {moment_pct, identity_pct}}}
    data = {}
    dx_values = set()
    
    for N in N_values:
        fname = src_dir / f"bounded_vs_dim_percentages_N_{N}.json"
        if not fname.exists():
            print(f"Warning: {fname} not found, skipping")
            continue
        
        try:
            with open(fname, 'r') as f:
                records = json.load(f)
            
            data[N] = {}
            for record in records:
                dx = record['dx']
                dx_values.add(dx)
                data[N][dx] = {
                    'moment_bounded_pct': record.get('moment_bounded_pct', 0.0),
                    'identity_bounded_pct': record.get('identity_bounded_pct', 0.0)
                }
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
    
    dx_values = sorted(dx_values)
    N_values = sorted([N for N in N_values if N in data])
    
    return data, dx_values, N_values

def create_heatmap_data(data, dx_values, N_values):
    """Create 2D arrays for heatmap plotting."""
    moment_matrix = np.zeros((len(dx_values), len(N_values)))
    identity_matrix = np.zeros((len(dx_values), len(N_values)))
    
    for i, dx in enumerate(dx_values):
        for j, N in enumerate(N_values):
            if N in data and dx in data[N]:
                moment_matrix[i, j] = data[N][dx]['moment_bounded_pct']
                identity_matrix[i, j] = data[N][dx]['identity_bounded_pct']
            else:
                moment_matrix[i, j] = np.nan
                identity_matrix[i, j] = np.nan
    
    return moment_matrix, identity_matrix

def plot_heatmaps(moment_matrix, identity_matrix, dx_values, N_values, output_file):
    """Create two side-by-side heatmaps with shared color scale."""
    # Try different colormaps - options: 'viridis', 'plasma', 'inferno', 'magma', 
    # 'YlOrRd', 'YlGnBu', 'Blues', 'Greens', 'RdYlGn', 'coolwarm', 'seismic'
    # Using 'YlGnBu' (Yellow-Green-Blue) for a clean, professional look
    # Alternative: 'viridis' for better colorblind accessibility
    colormap = 'RdYlGn'  # Change to 'viridis', 'plasma', or 'YlOrRd' for different looks
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Common color scale settings - explicitly set to 0-100 for both plots
    vmin = 0.0
    vmax = 100.0
    
    # Determine if we should show only even dimension labels (if too many dimensions)
    show_only_even_dx = len(dx_values) > 15
    if show_only_even_dx:
        # Create labels showing only even dimensions
        dx_labels = [str(dx) if dx % 2 == 0 else '' for dx in dx_values]
    else:
        dx_labels = [str(dx) for dx in dx_values]
    
    # Plot 1: Moment Matching
    im1 = axes[0].imshow(moment_matrix, aspect='auto', cmap=colormap, 
                        vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0].invert_yaxis()  # Reverse the dimension axis
    axes[0].set_title('Moment Matching Boundedness (%)', fontsize=24, fontweight='bold', pad=15)
    axes[0].set_xlabel('N (Number of Samples)', fontsize=24)
    axes[0].set_ylabel('dx (State Dimension)', fontsize=24)
    
    # Set ticks with better formatting
    axes[0].set_xticks(range(len(N_values)))
    axes[0].set_xticklabels(N_values, rotation=45, ha='right', fontsize=20)
    axes[0].set_yticks(range(len(dx_values)))
    axes[0].set_yticklabels(dx_labels, fontsize=20)
    
    # Add grid for better readability
    axes[0].set_xticks([x - 0.5 for x in range(len(N_values) + 1)], minor=True)
    axes[0].set_yticks([y - 0.5 for y in range(len(dx_values) + 1)], minor=True)
    axes[0].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add text annotations for each cell, but skip 0 and 100 to reduce clutter
    for i in range(len(dx_values)):
        for j in range(len(N_values)):
            value = moment_matrix[i, j]
            if not np.isnan(value) and value != 0 and value != 100:
                # Use white text for darker backgrounds, black for lighter
                text_color = "white" if value < 50 else "black"
                text = axes[0].text(j, i, f'{value:.0f}',
                                  ha="center", va="center", 
                                  color=text_color,
                                  fontsize=14, fontweight='bold')
    
    # Plot 2: Identity Covariance - use same color scale
    im2 = axes[1].imshow(identity_matrix, aspect='auto', cmap=colormap,
                        vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].invert_yaxis()  # Reverse the dimension axis
    axes[1].set_title('Identity Covariance Boundedness (%)', fontsize=24, fontweight='bold', pad=15)
    axes[1].set_xlabel('N (Number of Samples)', fontsize=24)
    axes[1].set_ylabel('dx (State Dimension)', fontsize=24)
    
    # Set ticks with better formatting
    axes[1].set_xticks(range(len(N_values)))
    axes[1].set_xticklabels(N_values, rotation=45, ha='right', fontsize=20)
    axes[1].set_yticks(range(len(dx_values)))
    axes[1].set_yticklabels(dx_labels, fontsize=20)
    
    # Add grid for better readability
    axes[1].set_xticks([x - 0.5 for x in range(len(N_values) + 1)], minor=True)
    axes[1].set_yticks([y - 0.5 for y in range(len(dx_values) + 1)], minor=True)
    axes[1].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar with same explicit limits
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Boundedness (%)', fontsize=20, fontweight='bold')
    cbar2.set_ticks([0, 20, 40, 60, 80, 100])
    cbar2.ax.tick_params(labelsize=20)
    
    # Add text annotations for each cell, but skip 0 and 100 to reduce clutter
    for i in range(len(dx_values)):
        for j in range(len(N_values)):
            value = identity_matrix[i, j]
            if not np.isnan(value) and value != 0 and value != 100:
                # Use white text for darker backgrounds, black for lighter
                text_color = "white" if value < 50 else "black"
                text = axes[1].text(j, i, f'{value:.0f}',
                                  ha="center", va="center",
                                  color=text_color,
                                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_file}")
    plt.close()

def main():
    try:
        # Load data
        print("Loading data from JSON files...")
        data, dx_values, N_values = load_percentages_data()
        
        if not data:
            print("Error: No data files found!")
            return
        
        print(f"Found data for dx values: {dx_values}")
        print(f"Found data for N values: {N_values}")
        
        # Create heatmap matrices
        print("Creating heatmap matrices...")
        moment_matrix, identity_matrix = create_heatmap_data(data, dx_values, N_values)
        
        # Create output directory if needed
        figures_dir = Path(__file__).parent.parent / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Plot heatmaps
        output_file = figures_dir / "boundedness_heatmap_dx_vs_N.pdf"
        print(f"Creating heatmaps...")
        plot_heatmaps(moment_matrix, identity_matrix, dx_values, N_values, output_file)
        
        print("Done!")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
