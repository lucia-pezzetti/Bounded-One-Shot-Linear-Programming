"""
Utility module for loading generated controllable systems.

Usage:
    from data.load_systems import load_systems, load_system, get_available_dimensions
    
    # Load all 50 systems for dimension 10
    systems = load_systems(n=10)
    A, B, C = systems[0]['A'], systems[0]['B'], systems[0]['C']
    
    # Load a specific system (dimension 10, system index 5)
    A, B, C = load_system(n=10, idx=5)
    
    # Get list of available dimensions
    dims = get_available_dimensions()
"""

import json
import os
import numpy as np
from typing import Tuple, List, Dict, Optional

# Directory containing the system files
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def get_available_dimensions() -> List[int]:
    """
    Get list of dimensions for which system files exist.
    
    Returns:
        List of available dimensions (sorted).
    """
    dims = []
    for filename in os.listdir(DATA_DIR):
        if filename.startswith("dx_") and filename.endswith("_du_2_systems_v2.json"):
            # Extract dimension from filename like "dx_10_du_2_systems.json"
            parts = filename.split("_")
            try:
                n = int(parts[1])
                dims.append(n)
            except (IndexError, ValueError):
                continue
    return sorted(dims)


def load_systems(n: int) -> List[Dict]:
    """
    Load all systems for a given dimension.
    
    Args:
        n: State dimension.
        
    Returns:
        List of dictionaries, each containing 'A', 'B', 'C' (as lists) and 'seed'.
        
    Raises:
        FileNotFoundError: If no file exists for the given dimension.
    """
    filename = os.path.join(DATA_DIR, f"dx_{n}_du_2_systems.json")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No systems file found for dimension {n}. "
                                f"Available dimensions: {get_available_dimensions()}")
    
    with open(filename, "r") as f:
        data = json.load(f)
    
    return data["systems"]


def load_system(n: int, idx: int = 0, as_numpy: bool = True) -> Tuple:
    """
    Load a specific system by dimension and index.
    
    Args:
        n: State dimension.
        idx: System index (0-49).
        as_numpy: If True, return numpy arrays. If False, return lists.
        
    Returns:
        Tuple (A, B, C) - system matrices.
        
    Raises:
        FileNotFoundError: If no file exists for the given dimension.
        IndexError: If idx is out of range.
    """
    systems = load_systems(n)
    
    if idx < 0 or idx >= len(systems):
        raise IndexError(f"System index {idx} out of range. "
                         f"Available indices: 0-{len(systems)-1}")
    
    system = systems[idx]
    A, B, C = system["A"], system["B"], system["C"]
    
    if as_numpy:
        return np.array(A), np.array(B), np.array(C)
    return A, B, C


def load_all_systems_as_numpy(n: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load all systems for a dimension as numpy arrays.
    
    Args:
        n: State dimension.
        
    Returns:
        List of tuples (A, B, C) as numpy arrays.
    """
    systems = load_systems(n)
    return [(np.array(s["A"]), np.array(s["B"]), np.array(s["C"])) for s in systems]


def get_system_seed(n: int, idx: int) -> int:
    """
    Get the random seed used to generate a specific system.
    
    Args:
        n: State dimension.
        idx: System index (0-49).
        
    Returns:
        The seed value.
    """
    systems = load_systems(n)
    return systems[idx]["seed"]


# Convenience: preload commonly used dimensions on import (optional)
if __name__ == "__main__":
    # Demo usage
    print("Available dimensions:", get_available_dimensions())
    
    for n in [2, 5, 10]:
        if n in get_available_dimensions():
            A, B, C = load_system(n, idx=0)
            print(f"\nDimension {n}:")
            print(f"  A shape: {A.shape}")
            print(f"  B shape: {B.shape}")
            print(f"  C shape: {C.shape}")
