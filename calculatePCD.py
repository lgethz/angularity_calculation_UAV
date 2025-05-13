import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from OverlapAnalysisFunctions import load_outline, find_boundaries, read_npy_file
import os
from pathlib import Path
import csv
import sys

# Remove IPython dependency completely for CLI usage
def print_header(text):
    """Print a section header"""
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)

def calculate_pcd_from_boundary(boundary_points):
    """
    Calculate the Pixel Count Diameter (PCD) for a grain using its boundary points.
    """
    # Calculate pairwise distances between all boundary points
    distances = distance.pdist(boundary_points)
    
    # Find the maximum distance (diameter)
    max_distance_idx = np.argmax(distances)
    max_distance = distances[max_distance_idx]
    
    # Convert condensed distance matrix index to square matrix indices
    # This uses scipy's squareform to get the full distance matrix
    # and then np.unravel_index to find the indices of the maximum value
    square_distances = distance.squareform(distances)
    i, j = np.unravel_index(np.argmax(square_distances), square_distances.shape)
    
    # Return the PCD and the points that define it
    return max_distance, (boundary_points[i], boundary_points[j])

def calculate_pcd_for_all_grains(outline_path, output_folder=None, show_plots=False):
    """
    Calculate the PCD for all grains in an outline image
    """
    # Load the outline data
    print_header(f"Loading outline from {outline_path}")
    
    # Check file extension and use appropriate loading function
    if outline_path.lower().endswith('.npy'):
        outline = read_npy_file(outline_path)
        print("Loaded .npy file format")
    else:
        outline = load_outline(outline_path)
        print("Loaded .tif file format")
    
    # Get grain boundaries
    print_header("Finding grain boundaries")
    grain_boundaries = find_boundaries(outline)
    print(f"Found {len(grain_boundaries)-1} grains (excluding background)")
    
    # Calculate PCD for each grain
    print_header("Calculating PCD for each grain")
    pcd_results = {}
    pcd_points = {}
    
    for grain_id, contours in grain_boundaries.items():
        if grain_id == 0:  # Skip background
            continue
        
        # Use the longest contour if there are multiple
        max_length = 0
        longest_contour = None
        for contour in contours:
            if len(contour) > max_length:
                max_length = len(contour)
                longest_contour = contour
        
        # Calculate PCD
        pcd, max_points = calculate_pcd_from_boundary(longest_contour)
        pcd_results[grain_id] = pcd
        pcd_points[grain_id] = max_points
        
        # Show progress for large datasets
        if grain_id % 20 == 0:
            print(f"Processed {grain_id} grains...")
    
    print_header(f"Completed PCD calculation for {len(pcd_results)} grains")
    
    # Display statistics
    pcd_values = list(pcd_results.values())
    if pcd_values:
        print_header("PCD Statistics:")
        print(f"Min: {min(pcd_values):.2f} pixels")
        print(f"Max: {max(pcd_values):.2f} pixels")
        print(f"Mean: {np.mean(pcd_values):.2f} pixels")
        print(f"Median: {np.median(pcd_values):.2f} pixels")
    
    # Visualize results if requested
    if show_plots:
        print_header("Generating PCD Visualization")
        plt.figure(figsize=(12, 12))
        plt.imshow(outline, cmap='viridis', alpha=0.7)
        
        for grain_id, points in pcd_points.items():
            point1, point2 = points
            plt.plot([point1[1], point2[1]], [point1[0], point2[0]], 'r-', linewidth=1)
            
        plt.title("PCD Visualization")
        plt.colorbar(label='Grain ID')
        plt.show()
    
    # Save results if output folder provided
    if output_folder:
        print_header(f"Saving results to {output_folder}")
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Save results to CSV
        csv_path = os.path.join(output_folder, "pcd_results.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Grain_ID', 'PCD_pixels'])
            for grain_id, pcd in pcd_results.items():
                writer.writerow([grain_id, pcd])
        
        # Save visualization as an image
        plt.figure(figsize=(12, 12))
        plt.imshow(outline, cmap='viridis', alpha=0.7)
        
        for grain_id, points in pcd_points.items():
            point1, point2 = points
            plt.plot([point1[1], point2[1]], [point1[0], point2[0]], 'r-', linewidth=1)
            
        plt.title("PCD Visualization")
        plt.colorbar(label='Grain ID')
        plt.savefig(os.path.join(output_folder, "pcd_visualization.png"))
        plt.close()
        
        print(f"Results saved to {output_folder}")
        
    return pcd_results

# This section runs when the script is executed directly
if __name__ == "__main__":

    outline_path =  "/Users/.../outline.tif" # Replace with your path
    output_folder = "./pcd_output"
    show_plots = True
    
    # Fix file extension if needed
    if outline_path.endswith('.tify'):
        outline_path = outline_path.replace('.tify', '.tif')
        print(f"Fixed file extension: {outline_path}")
        
    # Calculate PCDs
    print(f"Processing: {outline_path}")
    print(f"Output folder: {output_folder}")
    print(f"Show plots: {show_plots}")
    
    pcd_results = calculate_pcd_for_all_grains(outline_path, output_folder, show_plots)
    
    # Print summary of results
    print_header("PCD Results Summary")
    print(f"Total grains processed: {len(pcd_results)}")
    
    # Optional: print first few results
    print("\nFirst 5 results (if available):")
    for i, (grain_id, pcd) in enumerate(list(pcd_results.items())[:5]):
        print(f"Grain ID: {grain_id}, PCD: {pcd:.2f} pixels")
    
    print("\n--- PCD Calculation Complete ---")
    print("\nComplete! Check the output folder for full results.")