
import numpy as np
from scipy.ndimage import uniform_filter

def simulate_filter(window_size, threshold):
    print(f"--- Window: {window_size}, Threshold: {threshold} ---")
    
    # 1. Simulate a 3x3 "Random" blob in a sea of wafer (0.5)
    # Grid 10x10
    grid = np.full((10, 10), 0.5)
    # Add 3x3 blob of 2s
    grid[3:6, 3:6] = 2.0
    
    mean_map = uniform_filter(grid, size=window_size, mode="constant", cval=0.5)
    
    # Check center of blob
    center_val = grid[4, 4]
    center_mean = mean_map[4, 4]
    survives = center_mean >= threshold
    print(f"3x3 Blob Center Mean: {center_mean:.3f} -> {'Survives' if survives else 'Filtered'}")

    # 2. Simulate a "Near-full" region (solid 2s)
    grid_full = np.full((10, 10), 2.0)
    mean_map_full = uniform_filter(grid_full, size=window_size, mode="constant", cval=0.5)
    center_mean_full = mean_map_full[4, 4]
    survives_full = center_mean_full >= threshold
    print(f"Solid Region Center Mean: {center_mean_full:.3f} -> {'Survives' if survives_full else 'Filtered'}")
    
    # 3. Simulate a "Sparse Random" (single pixel)
    grid_single = np.full((10, 10), 0.5)
    grid_single[4, 4] = 2.0
    mean_map_single = uniform_filter(grid_single, size=window_size, mode="constant", cval=0.5)
    center_mean_single = mean_map_single[4, 4]
    survives_single = center_mean_single >= threshold
    print(f"Single Pixel Mean: {center_mean_single:.3f} -> {'Survives' if survives_single else 'Filtered'}")

simulate_filter(3, 1.25)
simulate_filter(5, 1.25)
simulate_filter(7, 1.25)
