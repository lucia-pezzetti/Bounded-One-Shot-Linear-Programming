#!/usr/bin/env python3
"""
Create 2D color plots showing boundedness percentages for moment matching and identity covariance
as a function of dx (state dimension) and N (number of samples) for NONLINEAR systems.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def load_nonlinear_data():
    """Load all bounded_vs_dim_results_nonlinear_N_*.json files and compute percentages."""
    src_dir = Path(__file__).parent
    
    # N values to look for
    N_values = [1000, 3000, 5000, 7000]
    
    # Check which files actually exist
    found_N_values = []
    for N in N_values:
        fname = src_dir / f"bounded_vs_dim_results_nonlinear_N_{N}.json"
        if fname.exists():
            found_N_values.append(N)
    
    print(f"Found N values: {found_N_values}")
    
    # Collect data: {N: {dx: {moment_pct, identity_pct}}}
    data = {}
    dx_values = set()
    
    for N in found_N_values:
        fname = src_dir / f"bounded_vs_dim_results_nonlinear_N_{N}.json"
        
        try:
            with open(fname, 'r') as f:
                records = json.load(f)
            
            # Group by dx and compute percentages
            dx_data = {}
            for record in records:
                dx = record['dx']
                dx_values.add(dx)
                
                if dx not in dx_data:
                    dx_data[dx] = {'moment_bounded': [], 'identity_bounded': []}
                
                dx_data[dx]['moment_bounded'].append(record.get('moment_matching_bounded', False))
                dx_data[dx]['identity_bounded'].append(record.get('identity_bounded', False))
            
            # Compute percentages
            data[N] = {}
            for dx, values in dx_data.items():
                moment_pct = 100.0 * np.mean(values['moment_bounded'])
                identity_pct = 100.0 * np.mean(values['identity_bounded'])
                data[N][dx] = {
                    'moment_bounded_pct': moment_pct,
                    'identity_bounded_pct': identity_pct
                }
                
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
    
    dx_values = sorted(dx_values)
    N_values = sorted([N for N in found_N_values if N in data])
    
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
    colormap = 'RdYlGn'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Common color scale settings - explicitly set to 0-100 for both plots
    vmin = 0.0
    vmax = 100.0
    
    # Font sizes
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 18
    ANNOTATION_SIZE = 16
    
    # Labels show dx + du where du = dx/2
    dx_du_labels = [str(int(dx + dx // 2)) for dx in dx_values]
    
    # Plot 1: Moment Matching
    im1 = axes[0].imshow(moment_matrix, aspect='auto', cmap=colormap, 
                        vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0].invert_yaxis()  # Reverse the dimension axis
    axes[0].set_title('Moment Matching Boundedness (%)', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    axes[0].set_xlabel('N (Number of Samples)', fontsize=LABEL_SIZE)
    axes[0].set_ylabel('dx + du (state + input dimension)', fontsize=LABEL_SIZE)
    
    # Set ticks with better formatting
    axes[0].set_xticks(range(len(N_values)))
    axes[0].set_xticklabels(N_values, rotation=45, ha='right', fontsize=TICK_SIZE)
    axes[0].set_yticks(range(len(dx_values)))
    axes[0].set_yticklabels(dx_du_labels, fontsize=TICK_SIZE)
    
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
                axes[0].text(j, i, f'{value:.0f}',
                            ha="center", va="center", 
                            color=text_color,
                            fontsize=ANNOTATION_SIZE, fontweight='bold')
    
    # Plot 2: Identity Covariance - use same color scale
    im2 = axes[1].imshow(identity_matrix, aspect='auto', cmap=colormap,
                        vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].invert_yaxis()  # Reverse the dimension axis
    axes[1].set_title('Identity Covariance Boundedness (%)', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    axes[1].set_xlabel('N (Number of Samples)', fontsize=LABEL_SIZE)
    axes[1].set_ylabel('dx + du (state + input dimension)', fontsize=LABEL_SIZE)
    
    # Set ticks with better formatting
    axes[1].set_xticks(range(len(N_values)))
    axes[1].set_xticklabels(N_values, rotation=45, ha='right', fontsize=TICK_SIZE)
    axes[1].set_yticks(range(len(dx_values)))
    axes[1].set_yticklabels(dx_du_labels, fontsize=TICK_SIZE)
    
    # Add grid for better readability
    axes[1].set_xticks([x - 0.5 for x in range(len(N_values) + 1)], minor=True)
    axes[1].set_yticks([y - 0.5 for y in range(len(dx_values) + 1)], minor=True)
    axes[1].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar with same explicit limits
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Boundedness (%)', fontsize=LABEL_SIZE, fontweight='bold')
    cbar2.set_ticks([0, 20, 40, 60, 80, 100])
    cbar2.ax.tick_params(labelsize=TICK_SIZE)
    
    # Add text annotations for each cell, but skip 0 and 100 to reduce clutter
    for i in range(len(dx_values)):
        for j in range(len(N_values)):
            value = identity_matrix[i, j]
            if not np.isnan(value) and value != 0 and value != 100:
                # Use white text for darker backgrounds, black for lighter
                text_color = "white" if value < 50 else "black"
                axes[1].text(j, i, f'{value:.0f}',
                            ha="center", va="center",
                            color=text_color,
                            fontsize=ANNOTATION_SIZE, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_file}")
    plt.close()

def main():
    try:
        # Load data
        print("Loading nonlinear data from JSON files...")
        data, dx_values, N_values = load_nonlinear_data()
        
        if not data:
            print("Error: No data files found!")
            return
        
        print(f"Found data for dx values: {dx_values}")
        print(f"Found data for N values: {N_values}")
        
        # Print the percentages table
        print("\nBoundedness Percentages:")
        print("-" * 60)
        print(f"{'dx':<6} {'N':<8} {'Moment (%)':<15} {'Identity (%)':<15}")
        print("-" * 60)
        for dx in dx_values:
            for N in N_values:
                if N in data and dx in data[N]:
                    m_pct = data[N][dx]['moment_bounded_pct']
                    i_pct = data[N][dx]['identity_bounded_pct']
                    print(f"{dx:<6} {N:<8} {m_pct:<15.1f} {i_pct:<15.1f}")
        print("-" * 60)
        
        # Create heatmap matrices
        print("\nCreating heatmap matrices...")
        moment_matrix, identity_matrix = create_heatmap_data(data, dx_values, N_values)
        
        # Create output directory if needed
        figures_dir = Path(__file__).parent.parent / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Plot heatmaps
        output_file = figures_dir / "boundedness_heatmap_nonlinear_dx_vs_N.pdf"
        print(f"Creating heatmaps...")
        plot_heatmaps(moment_matrix, identity_matrix, dx_values, N_values, output_file)
        
        print("Done!")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
