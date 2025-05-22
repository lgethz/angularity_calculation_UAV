import numpy as np
import matplotlib.pyplot as plt
import rasterio
import scipy.ndimage as ndimage
from skimage import measure
import warnings
from rasterio.errors import NotGeoreferencedWarning
import os
import csv
from skimage.io import imsave
from scipy import ndimage
import glob
from pathlib import Path
from scipy.spatial import distance
import re

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

def load_dem(file_path: str):
    """
    Loads a Digital Elevation Model (DEM) from a GeoTIFF file. Reads the first band of the raster dataset.

    Parameters: Path to the DEM GeoTIFF file    
    Returns: NumPy array containing elevation values
    """
    try:
        with rasterio.open(file_path) as dem_src:
            dem = dem_src.read(1)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    except rasterio.errors.RasterioIOError as e:
        raise rasterio.errors.RasterioIOError(f"Error reading file: {file_path}. Error: {e}")
    
    except ValueError as e:
        raise ValueError(f"ERROR: Invalid raster format: {str(e)}")

    return dem


def show_dem(dem, output_path=None, show = True, close_after = False):
    """
    Visualizes the Digital Elevation Model using matplotlib with a rainbow colormap.

    Parameters: 
        dem: NumPy array containing the DEM data
        output_path: If provided, saves the visualization to this path
        show: Whether to display the plot (default=True)
        close_after: Whether to close the plot after showing (default=False)

    Visualization: Creates a figure with colorbar showing elevation values
    """
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(dem, cmap='rainbow')  # or 'jet'
        plt.colorbar()
        plt.title("Digital Elevation Model")
        plt.gcf().canvas.toolbar.zoom()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved DEM visualization to {output_path}")
        
        if show:
            plt.show()

        if close_after:
            plt.close()

    except dem is None:
        raise ValueError("ERROR: DEM data is None. Please check the file path or format.")

    except not isinstance(dem, np.ndarray):
        raise ValueError("ERROR: DEM data is not a valid NumPy array. Please check the file format.")
    
    except dem.size == 0:
        raise ValueError("ERROR: DEM data is empty. Please check the file path or format.")


def load_outline(file_path: str, show_ids=True):
    """
    Loads a pre-analyzed image from ImageGrains that contains grain segmentation data.

    Parameters: 
        file_path: Path to the grain segmentation GeoTIFF file
        show_ids: If True, displays a plot with grain IDs labeled
    Returns: 
        NumPy array where each pixel value represents a grain ID (0 = background)    
    """
    try:
        with rasterio.open(file_path) as outline_src:
            outline = outline_src.read(1)
        
        unique_ids = np.unique(outline)
        unique_ids = unique_ids[unique_ids > 0]  
        print(f"Found {len(unique_ids)} grains with IDs: {unique_ids[:10]}{'...' if len(unique_ids) > 10 else ''}")
        
        if show_ids:
            plt.figure(figsize=(12, 12))
            plt.imshow(outline, cmap='viridis')
            plt.colorbar(label='Grain ID')
            
            for grain_id in unique_ids:
                grain_mask = (outline == grain_id).astype(int)
                
                if np.sum(grain_mask) < 20:
                    continue
                
                centroid = ndimage.center_of_mass(grain_mask)
                centroid_y, centroid_x = int(centroid[0]), int(centroid[1])
                
                plt.text(centroid_x, centroid_y, str(grain_id), 
                         color='white', fontsize=8, ha='center', va='center',
                         bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
            
            plt.title(f"Grain Segmentation with ID Labels ({len(unique_ids)} grains)")
            plt.tight_layout()
            plt.show()
        
        return outline

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    except rasterio.errors.RasterioIOError as e:
        raise rasterio.errors.RasterioIOError(f"Error reading file: {file_path}. Error: {e}")
    
    except ValueError as e:
        raise ValueError(f"ERROR: Invalid raster format: {str(e)}")

def remove_edge_grains(outline, show_comparison=False, close_after = False):
    """
    Identifies and removes grains that touch the edge of the image.
    
    Parameters:
        outline: NumPy array where each pixel value represents a grain ID (0 = background)
        show_comparison: If True, displays a before/after comparison visualization
    
    Returns:
        filtered_outline: NumPy array with edge-touching grains removed (set to 0)
    """
    if not isinstance(outline, np.ndarray):
        raise ValueError("ERROR: Outline data is not a valid NumPy array.")
    
    if outline.size == 0:
        raise ValueError("ERROR: Outline data is empty.")
    
    height, width = outline.shape
    
    top_border = outline[0, :]
    bottom_border = outline[height-1, :]
    left_border = outline[:, 0]
    right_border = outline[:, width-1]
    
    all_border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
    edge_grain_ids = np.unique(all_border_pixels)
    edge_grain_ids = edge_grain_ids[edge_grain_ids > 0] 
    
    filtered_outline = outline.copy()
    
    original_grain_ids = np.unique(outline)
    original_grain_ids = original_grain_ids[original_grain_ids > 0]
    
    for grain_id in edge_grain_ids:
        filtered_outline[filtered_outline == grain_id] = 0
    
    remaining_grain_ids = np.unique(filtered_outline)
    remaining_grain_ids = remaining_grain_ids[remaining_grain_ids > 0]
    
    removed_count = len(edge_grain_ids)
    total_count = len(original_grain_ids)
    remaining_count = len(remaining_grain_ids)
    
    print(f"Found {removed_count} grains touching the image edge out of {total_count} total grains")
    print(f"Removed IDs: {edge_grain_ids[:10]}{'...' if len(edge_grain_ids) > 10 else ''}")
    print(f"Remaining grains: {remaining_count}")
    
    if show_comparison:
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(outline, cmap='viridis')
        plt.title(f"Original Segmentation\n({total_count} grains)")
        plt.colorbar(label='Grain ID')
        
        plt.subplot(1, 2, 2)
        plt.imshow(filtered_outline, cmap='viridis')
        plt.title(f"Clasts Remaining\n({remaining_count} clasts)")
        plt.colorbar(label='Grain ID')
        
        removed_mask = np.zeros_like(outline, dtype=bool)
        for grain_id in edge_grain_ids:
            removed_mask = removed_mask | (outline == grain_id)
        
        plt.subplot(1, 2, 1)
        plt.imshow(removed_mask, cmap='autumn', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        if close_after:
            plt.close()
    
    return filtered_outline


def remove_small_grains(outline, min_pcd=None, show_comparison=False, close_after = False):
    """
    Removes grains that are smaller than a specified pixel count diameter (PCD).
    
    Parameters:
        outline: NumPy array where each pixel value represents a grain ID (0 = background)
        min_pcd: Minimum PCD to keep a grain, if None will prompt user
        show_comparison: If True, displays a before/after comparison visualization
    
    Returns:
        filtered_outline: NumPy array with small grains removed (set to 0)
    """
    if not isinstance(outline, np.ndarray):
        raise ValueError("ERROR: Outline data is not a valid NumPy array.")
    
    if outline.size == 0:
        raise ValueError("ERROR: Outline data is empty.")
    
    grain_boundaries = find_boundaries(outline)

    pcd_results = {}
    pcd_points = {} 
    
    for grain_id, contours in grain_boundaries.items():
        if grain_id == 0:
            continue
        
        max_length = 0
        longest_contour = None
        for contour in contours:
            if len(contour) > max_length:
                max_length = len(contour)
                longest_contour = contour
        
        if longest_contour is not None:
            distances = distance.pdist(longest_contour)
            
            if len(distances) > 0:
                square_distances = distance.squareform(distances)
                i, j = np.unravel_index(np.argmax(square_distances), square_distances.shape)
                max_distance = square_distances[i, j]
                
                pcd_results[grain_id] = max_distance
                pcd_points[grain_id] = (longest_contour[i], longest_contour[j])
    
    grain_ids = list(pcd_results.keys())
    pcd_values = list(pcd_results.values())
    
    if len(pcd_values) > 0:
        print(f"PCD Statistics:")
        print(f"  Min: {min(pcd_values):.2f} pixels")
        print(f"  Max: {max(pcd_values):.2f} pixels")
        print(f"  Mean: {np.mean(pcd_values):.2f} pixels")
        print(f"  Median: {np.median(pcd_values):.2f} pixels")
    
    if min_pcd is None:
        min_pcd = float(input(f"Enter minimum PCD to keep (in pixels): "))
    
    small_grain_ids = [grain_id for grain_id, pcd in pcd_results.items() if pcd < min_pcd]
    
    filtered_outline = outline.copy()
    
    for grain_id in small_grain_ids:
        filtered_outline[filtered_outline == grain_id] = 0
    
    original_grain_ids = np.unique(outline)
    original_grain_ids = original_grain_ids[original_grain_ids > 0]
    
    remaining_grain_ids = np.unique(filtered_outline)
    remaining_grain_ids = remaining_grain_ids[remaining_grain_ids > 0]
    
    removed_count = len(small_grain_ids)
    total_count = len(original_grain_ids)
    remaining_count = len(remaining_grain_ids)
    
    print(f"Removed {removed_count} grains smaller than {min_pcd:.2f} pixels PCD")
    print(f"Original grains: {total_count}")
    print(f"Remaining grains: {remaining_count}")
    
    if show_comparison:
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(outline, cmap='viridis')
        plt.title(f"Original Segmentation\n({total_count} grains)")
        plt.colorbar(label='Grain ID')
        
        for grain_id, points in pcd_points.items():
            point1, point2 = points
            plt.plot([point1[1], point2[1]], [point1[0], point2[0]], 'r-', linewidth=1)
        
        plt.subplot(1, 2, 2)
        plt.imshow(filtered_outline, cmap='viridis')
        plt.title(f"Grains Remaining\n({remaining_count} grains, min PCD={min_pcd:.2f})")
        plt.colorbar(label='Grain ID')
        
        removed_mask = np.zeros_like(outline, dtype=bool)
        for grain_id in small_grain_ids:
            removed_mask = removed_mask | (outline == grain_id)
        
        plt.subplot(1, 2, 1)
        plt.imshow(removed_mask, cmap='autumn', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        if close_after:
            plt.close()
    
    return filtered_outline


def show_outline(outline, output_path=None, show=True, close_after=False):
    """ 
    Visualizes the grain segmentation image using matplotlib with a rainbow colormap.

    Parameters: 
        outline: NumPy array containing grain segmentation data
        output_path: If provided, saves the visualization to this path
        show: Whether to display the plot (default=True)
        close_after: Whether to close the plot after showing (default=False)
    """
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(outline, cmap='rainbow')  # or 'jet'
        plt.colorbar(label='Grain ID')
        plt.title("Grain segmentation data")
        plt.gcf().canvas.toolbar.zoom()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved outline visualization to {output_path}")
            
        if show:
            plt.show()
            
        if close_after:
            plt.close()
    
    except Exception as e:
        raise ValueError(f"ERROR visualizing outline: {str(e)}")


def create_coordinate_transformer(dem, outline):
    """
    Creates coordinate transformation functions between DEM and outline datasets with different resolutions.
    
    Parameters:
        dem: NumPy array of the DEM
        outline: NumPy array of the grain outline/segmentation
    
    Returns:
        dict: Dictionary with transformation functions and metadata
    """
    dem_height, dem_width = dem.shape
    outline_height, outline_width = outline.shape
    
    # Calculate scaling factors
    height_ratio = outline_height / dem_height
    width_ratio = outline_width / dem_width
    
    # Make sure scaling is consistent in both dimensions
    if abs(height_ratio - width_ratio) > 0.01:
        print(f"WARNING: Inconsistent scaling - height ratio: {height_ratio}, width ratio: {width_ratio}")
    
    scaling_factor = (height_ratio + width_ratio) / 2
    print(f"Detected scaling factor: {scaling_factor}")
    
    # Create transformation functions
    def outline_to_dem(y, x):
        """Convert outline coordinates to DEM coordinates"""
        dem_y = int(y / scaling_factor)
        dem_x = int(x / scaling_factor)
        # Ensure coordinates are within bounds
        dem_y = max(0, min(dem_y, dem_height - 1))
        dem_x = max(0, min(dem_x, dem_width - 1))
        return dem_y, dem_x
    
    def dem_to_outline(y, x):
        """Convert DEM coordinates to outline coordinates"""
        outline_y = int(y * scaling_factor)
        outline_x = int(x * scaling_factor)
        outline_y = max(0, min(outline_y, outline_height - 1))
        outline_x = max(0, min(outline_x, outline_width - 1))
        return outline_y, outline_x
    
    return {
        'scaling_factor': scaling_factor,
        'outline_to_dem': outline_to_dem,
        'dem_to_outline': dem_to_outline,
        'dem_shape': dem.shape,
        'outline_shape': outline.shape
    }


def find_boundaries(outline):
     """
     Detects the edges/boundaries of every grain in the segmented image.

    Parameters: 
        Outline array with grain IDs
    Algorithm:
        Finds unique grain IDs (excluding background)
        For each grain ID, creates a binary mask
        Uses measure.find_contours() to extract grain boundaries
    Returns: 
        Dictionary mapping grain IDs to lists of contour points
     """
     if not isinstance(outline, np.ndarray):
         raise ValueError("ERROR: Outline data is not a valid NumPy array. Please check the file format.")
     
     if outline.size == 0:
        raise ValueError("ERROR: Outline data is empty. Please check the file path or format.")
    
     unique_values = np.unique(outline)
    
     grain_boundaries = {}

     # Loop through all the different grains and find all the edges
     for grain_id in unique_values:
        if grain_id == 0: # skip background
            continue

        else:
            # Create a binary mask for the current grain
            binary_mask = (outline == grain_id).astype(np.uint8)

            # Use sci-kit image to find countours
            contours = measure.find_contours(binary_mask, level = 0.5)
            grain_boundaries[grain_id] = contours
    
     return grain_boundaries


def plot_boundaries_on_dem(dem, grain_boundaries, outline = None, output_path = None, show = True, close_after = False):
    """
    Plots the grain boundaries on top of the DEM for visualization.

    Parameters: 
        dem: DEM array
        grain_boundaries: Dictionary of grain contours
        outline: Optional segmentation data for coordinate transformation
        output_path: If provided, saves the visualization to this path
        show: Whether to display the plot (default=True)
        close_after: Whether to close the plot after showing (default=False)

    Visualization: 
        Creates a plot with DEM as background and grain boundaries overlaid
    """
    if not isinstance(dem, np.ndarray):
        raise ValueError("ERROR: DEM data is not a valid NumPy array. Please check the file format.")
    
    if dem.size == 0:
        raise ValueError("ERROR: DEM data is empty. Please check the file path or format.")
        
    if not isinstance(grain_boundaries, dict):
        raise TypeError("ERROR: grain_boundaries must be a dictionary")
    
    if len(grain_boundaries) == 0:
        raise ValueError("ERROR: grain_boundaries dictionary is empty")

    if not isinstance(dem, np.ndarray):
        raise ValueError("ERROR: DEM data is not a valid NumPy array.")
    
    # Create transformer if outline is provided
    transformer = None
    if outline is not None:
        transformer = create_coordinate_transformer(dem, outline)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(dem, cmap='rainbow', alpha=0.5)
    
    for grain_id, contours in grain_boundaries.items():
        for contour in contours:
            # Transform coordinates if needed
            if transformer:
                # Create transformed contour points
                transformed_contour = np.zeros_like(contour)
                for i, (y, x) in enumerate(contour):
                    dem_y, dem_x = transformer['outline_to_dem'](y, x)
                    transformed_contour[i, 0] = dem_y
                    transformed_contour[i, 1] = dem_x
                plt.plot(transformed_contour[:, 1], transformed_contour[:, 0], 'w-', linewidth=1)
            else:
                # Use original coordinates (assuming they match DEM)
                plt.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=1)
    
    plt.title("Grain Boundaries on DEM")
    plt.colorbar(label='Elevation')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved boundaries visualization to {output_path}")
        
    if show:
        plt.show()
        
    if close_after:
        plt.close()


def analyse_overlap(grain_boundaries, DEM, outline, range_check: int = 5, VCO: int = 0.002):
    """
    Core function that analyzes potential overlapping regions between grains.

    Parameters:
        grain_boundaries: Dictionary of grain contours
        DEM: Digital Elevation Model data
        range_check: Maximum distance (in pixels) to check for overlaps
        VCO (Vertical Clast Offset): Minimum elevation difference to consider as overlap
        outline: Segmentation image with grain IDs
    Algorithm:
        For each grain, creates a binary mask
        For each point along the grain boundary:
        Computes normal vector to the boundary
        Ensures normal points outward from grain
        Checks points along normal vector for elevation differences
        If elevation at a point exceeds boundary elevation by VCO, marks as overlap
        Accounts for different scales between DEM and outline
    Returns: 
        Dictionary mapping grain IDs to lists of overlap points
    """
    if not isinstance(grain_boundaries, dict):
        raise TypeError("grain_boundaries must be a dictionary")
    
    if len(grain_boundaries) == 0:
        raise ValueError("grain_boundaries dictionary is empty")
        
    if not isinstance(DEM, np.ndarray):
        raise TypeError("DEM must be a NumPy array")
        
    if DEM.size == 0:
        raise ValueError("DEM array is empty")
        
    if not isinstance(range_check, int) or range_check <= 0:
        raise ValueError(f"range_check must be a positive integer, got {range_check}")
        
    if not isinstance(VCO, (int, float)) or VCO <= 0:
        raise ValueError(f"VCO must be a positive number, got {VCO}")
    
    transformer = create_coordinate_transformer(DEM, outline)
    outline_to_dem = transformer['outline_to_dem']

    overlap_results = {}
    image_data = outline

    for grain_id in grain_boundaries.keys(): # Loop going through every grain and 
        if grain_id == 0: # skip background
            continue

        else:
            binary_mask = (image_data == grain_id).astype(np.uint8) # Create a binary mask for the current grain, it is used later

            for contour in grain_boundaries[grain_id]: # Loop through the outlines of a grain with the same grain id (if it has holes or disconnected areas it can have several outlines)
                n_points = len(contour)

                for i, outline_point in enumerate(contour): # Loop through the single points in a grain
                    y, x = outline_point

                    prev_idx = (i-1) % n_points # The % n_points ensures that the index uses the last/first point also at the start and beginning
                    next_idx = (i+1) % n_points
                    prev_point = contour[prev_idx]
                    next_point = contour[next_idx]

                    dy = next_point[0] - prev_point[0] # difference in y between the next and previous point
                    dx = next_point[1] - prev_point[1]

                    length = np.sqrt(dy**2 + dx**2)

                    if length > 1e-10: # avoid division by zero
                        dx = dx / length # normalize to unit length, length of the vector with dy and dx is one
                        dy = dy / length
                
                        normal_x, normal_y = -dy, dx # 90° rotation of the vector to get the normal vector

                        inside_y = int(round(y - normal_y)) # take a step in the direction of the normal vector, inside should be on the inside of the grain and outside on the outside
                        inside_x = int(round(x - normal_x))
                        outside_y = int(round(y + normal_y))
                        outside_x = int(round(x + normal_x))

                        h, w = binary_mask.shape
                        if not (0 <= inside_y < h and 0 <= inside_x < w and 0 <= outside_y < h and 0 <= outside_x < w): # check if the used points to decide the overlap are inside the image
                            continue

                        else:
                            if binary_mask[inside_y, inside_x] < binary_mask[outside_y, outside_x]: # check if normal vector points outside of the grain (in a binary_mask the value of the grain is 1 and the background is 0)
                                normal_x = -normal_x
                                normal_y = -normal_y

                            dem_y, dem_x = outline_to_dem(y,x) # transform the coordinates to the DEM coordinates
                            boundary_elevation = DEM[int(dem_y), int(dem_x)] # get the elevation of the boundary point

                            for d in range(1, range_check+1): # check for overlap in the range of the normal vector (1 to range_check away from the point in the direction of the normal vector)
                                sample_outline_y = int(round(y + d * normal_y))
                                sample_outline_x = int(round(x + d * normal_x))
                                sample_dem_y, sample_dem_x = outline_to_dem(sample_outline_y, sample_outline_x) # transform the coordinates to the DEM coordinates
                                
                                if (0 <= sample_outline_y < outline.shape[0] and 
                                    0 <= sample_outline_x < outline.shape[1]): # Check if the sample point is inside the image

                                    if (0 <= sample_dem_y < DEM.shape[0] and 
                                            0 <= sample_dem_x < DEM.shape[1]): # Check if the sample point is inside the DEM

                                        sample_elevation = DEM[sample_dem_y, sample_dem_x]
                                        elevation_difference = sample_elevation - boundary_elevation

                                        if (elevation_difference > VCO and outline[sample_outline_y, sample_outline_x] != 0 and outline[sample_outline_y, sample_outline_x] != grain_id):
                                                overlapped_grain_id = outline[sample_outline_y, sample_outline_x]
                                                
                                                if grain_id != overlapped_grain_id:
                                                    if grain_id not in overlap_results:
                                                            overlap_results[grain_id] = []
                                                        
                                                    if (x,y) not in overlap_results[grain_id]:
                                                            overlap_results[grain_id].append((x,y))
                                                    
                                                    break
    return overlap_results


def visualize_overlap(overlap_results, grain_boundaries, outline, output_path = None, show = True, close_after = False):
    """
    Creates a visualization showing grain boundaries and detected overlap points.

    Parameters:
        overlap_results: Dictionary of overlap points by grain ID
        grain_boundaries: Dictionary of grain contours
    Visualization:
        Grain segmentation as background
        White lines for grain boundaries
        Red dots for overlap points
        Legend and colorbar
    """
    if not isinstance(grain_boundaries, dict):
        raise TypeError("grain_boundaries must be a dictionary")
        
    if not isinstance(overlap_results, dict):
        raise TypeError("overlap_results must be a dictionary")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(outline, cmap='viridis', alpha=0.7)  

    # Plot all grain boundaries
    first_line = None
    for grain_id in grain_boundaries:
        for contour in grain_boundaries[grain_id]:
            lines = plt.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=1)
            if first_line is None:
                first_line = lines[0]
    
    # Plot all overlap points - simpler structure now
    first_point = None
    for grain_id, points in overlap_results.items():
        for point in points:  # Direct iteration over points
            pts = plt.plot(point[0], point[1], 'ro', markersize=1)
            if first_point is None:
                first_point = pts[0]
    
    # Add legend if we have data
    if first_line is not None and first_point is not None:
        plt.legend([first_line, first_point], 
                  ['Grain Boundaries', 'Overlap Points'],
                  loc='upper right')
    
    plt.title("Overlap Points on Grain Boundaries")
    plt.colorbar(label='Grain ID')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved overlap visualization to {output_path}")
        
    if show:
        plt.show()
        
    if close_after:
        plt.close()


def save_grain_images_and_overlap_csv(overlap_results, outline, output_folder : str = "./overlap_output"):
    """
    Saves each grain as a separate JPG file and all overlap points to a CSV file.

    Parameters:
        overlap_results: Dictionary of overlap points
        outline: NumPy array containing grain segmentation data
        output_folder: Folder where JPGs and CSV will be saved
    Output: 
        - Images subfolder with one JPG file per grain (white grain on black background)
        - CSV subfolder with one CSV file of all overlap points (MATLAB compatible)
    """
    if not isinstance(overlap_results, dict):
        raise TypeError("overlap_results must be a dictionary")

    if len(overlap_results) == 0:
        print("WARNING: No overlap points to save. Output will be empty.")
    
    # Create main output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Create subdirectories for images and CSV files
    images_folder = os.path.join(output_folder, "images")
    csv_folder = os.path.join(output_folder, "csv")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    # Process grain images
    grain_ids = np.unique(outline)
    grain_ids = grain_ids[grain_ids > 0]   # Get the unique grain IDs, excluding background (0)
        
    print(f"Processing {len(grain_ids)} grains...")

    for grain_id in grain_ids:
        grain_mask = 255 - ((outline == grain_id).astype(np.uint8) * 255) # convert it into a binary mask and scale it to 0-255 for a 8 bit grayscale image with high contrast
        output_jpg = os.path.join(images_folder, f"grain_{grain_id}.jpg")
        imsave(output_jpg, grain_mask)

    # Save overlap points to CSV file
    csv_path = os.path.join(csv_folder, "overlap_points.csv")
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Grain ID", "X", "Y"])
        point_count = 0
        for grain_id, points in overlap_results.items():
            for point in points:
                x, y = point
                csv_writer.writerow([grain_id, x, y])
                point_count += 1

    print(f"Saved {point_count} overlap points to {csv_path}")
    print(f"Saved {len(grain_ids)} grain images to {images_folder}")
    print(f"Output structure created at {output_folder}")


def read_npy_file(npy_file_path: str, dem_file_path=None):
    """
    Reads a segmentation data NPY file and converts it to a labeled image where each
    pixel value represents a grain ID.
    
    Parameters:
        npy_file_path: Path to the NPY file containing segmentation data
        dem_file_path: Optional path to DEM file for determining dimensions
                      (used if dimensions cannot be determined from segmentation)
    
    Returns:
        outline: 2D numpy array where each pixel value is a grain ID (0 = background)
    """
    try:
        data = np.load(npy_file_path, allow_pickle = True)
        print(f"Successfully loaded NPY file: {npy_file_path}")
        print(f"Type of loaded data: {type(data)}")
        print(f"Shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")

        if isinstance(data, np.ndarray) and data.size == 1 and data.dtype == "object":
            data = data.item()
            print(f"Extracted item from singleton array. Type: {type(data)}")
            
            if isinstance(data,dict):
                print("Keys in dictionary:", data.keys())

        if dem_file_path:
            try:
                with rasterio.open(dem_file_path) as dem_src:
                    height, width = dem_src.shape
                    print(f"Loaded DEM dimensions: {height} x {width}")
                
            except Exception as e:
                print(f"Error loading DEM file: {e}")
                if isinstance(data, dict) and 'masks' in data:
                    masks = data['masks']
                    if isinstance(masks, np.ndarray) and len(masks.shape) == 2:
                        height, width = masks.shape
                    elif hasattr(masks, '__iter__') and len(masks) > 0:
                        if isinstance(masks[0], np.ndarray):
                            height, width = masks[0].shape
                        else:
                            raise ValueError("Cannot determine dimensions from data")
                else:
                    raise ValueError("Cannot determine dimensions from data")
        
        else:
            if isinstance(data, dict) and 'masks' in data:
                masks = data['masks']
                if isinstance(masks, np.ndarray) and len(masks.shape) == 2:
                    height, width = masks.shape
                    print(f"Using mask dimensions: {height}x{width}")
                elif hasattr(masks, '__iter__') and len(masks) > 0 and isinstance(masks[0], np.ndarray):
                    height, width = masks[0].shape
                    print(f"Using first mask dimensions: {height}x{width}")
                else:
                    raise ValueError("Cannot determine dimensions from data")
            else:
                raise ValueError("Cannot determine dimensions and no DEM file provided")
            
        
        outline = np.zeros((height, width), dtype=np.int32)

        if isinstance(data,dict) and "masks" in data:
            masks = data["masks"]
            print(f"Found masks. Type: {type(masks)}")
            
            if isinstance(masks, np.ndarray) and len(masks.shape) == 2:
                outline = masks.copy().astype(np.int32)
                print("Using masks directly as label image")
            
            elif hasattr(masks, '__iter__') and not isinstance(masks, (str, bytes)):
                print(f"Processing {len(masks)} individual masks")
                for i, mask in enumerate(masks):
                    grain_id = i + 1

                    if isinstance(mask, np.ndarray) and mask.dtype == bool:
                        outline[mask] = grain_id
                        print(f"Grain {i}: Boolean mask")
                    elif isinstance(mask, tuple) and len(mask) == 2:
                        outline[mask] = grain_id
                        print(f"Grain {i}: Coordinate tuple")
                    else:
                        print(f"Grain {i}: Unknown format {type(mask)}")
            else:
                raise ValueError("No valid masks found in data")

        else:
            raise ValueError("No valid masks found in data")
        
        print(f"Successfully created labeled image with {np.max(outline)} grains")
        return outline

    except Exception as e:
        raise Exception(f"Error reading NPY file: {e}")
    

def visualize_normal_vectors(grain_boundaries, outline, sample_rate=10):
    """
    Visualizes the normal vectors to grain boundaries to help debug overlap detection.
    
    Parameters:
        grain_boundaries: Dictionary of grain contours
        outline: NumPy array containing grain segmentation data
        sample_rate: Only show every nth vector to avoid overcrowding (default=10)
    
    Returns:
        None (displays a matplotlib figure)
    """
    if not isinstance(grain_boundaries, dict):
        raise TypeError("grain_boundaries must be a dictionary")
    
    plt.figure(figsize=(12, 12))
    plt.imshow(outline, cmap='viridis', alpha=0.7)
    
    # For storing all normal vectors for visualization
    all_points = []
    all_normals = []
    all_grain_ids = []
    
    # Process each grain
    for grain_id in grain_boundaries.keys():
        if grain_id == 0:  # skip background
            continue
            
        binary_mask = (outline == grain_id).astype(np.uint8)
        
        for contour in grain_boundaries[grain_id]:
            n_points = len(contour)
            
            # Plot the contour
            plt.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=1)
            
            # Calculate normals for selected points
            for i in range(0, n_points, sample_rate):  # Sample every nth point
                y, x = contour[i]
                
                # Get neighboring points for tangent calculation
                prev_idx = (i-1) % n_points
                next_idx = (i+1) % n_points
                prev_point = contour[prev_idx]
                next_point = contour[next_idx]
                
                # Calculate tangent vector
                dy = next_point[0] - prev_point[0]
                dx = next_point[1] - prev_point[1]
                
                length = np.sqrt(dy**2 + dx**2)
                
                if length > 1e-10:  # avoid division by zero
                    dx = dx / length
                    dy = dy / length
                    
                    normal_x, normal_y = -dy, dx # Calculate initial normal vector (90° rotation)
                    
                    # Check if normal points outward and correct if needed
                    inside_y = int(round(y - normal_y))
                    inside_x = int(round(x - normal_x))
                    outside_y = int(round(y + normal_y))
                    outside_x = int(round(x + normal_x))
                    
                    h, w = binary_mask.shape
                    if 0 <= inside_y < h and 0 <= inside_x < w and 0 <= outside_y < h and 0 <= outside_x < w:
                        if binary_mask[inside_y, inside_x] < binary_mask[outside_y, outside_x]:
                            normal_x = -normal_x
                            normal_y = -normal_y

                        all_points.append((x, y))
                        scale = 10
                        all_normals.append((normal_x * scale, normal_y * scale))
                        all_grain_ids.append(grain_id)
    
    unique_ids = np.unique(all_grain_ids)
    cmap = plt.cm.get_cmap('tab10', len(unique_ids))
    
    for i, (point, normal, grain_id) in enumerate(zip(all_points, all_normals, all_grain_ids)):
        color_idx = np.where(unique_ids == grain_id)[0][0]
        color = cmap(color_idx)
        
        plt.arrow(point[0], point[1], normal[0], normal[1], 
                  head_width=2, head_length=2, fc=color, ec=color, 
                  length_includes_head=True)
    
    plt.title("Grain Boundaries with Normal Vectors")
    plt.colorbar(label='Grain ID')
    plt.tight_layout()
    plt.show()


def process_folder(dem_folder, outline_folder, range_check, VCO, 
                  matlab_tol=0.05, matlab_factor=0.98, matlab_span=0.07, 
                  matlab_exclusion_range=5, output_base="./overlap_output",
                  show_visualizations=False, min_pcd=None):
    """
    Process all DEM and outline files in the specified folders.
    
    Parameters:
        dem_folder: Path to folder containing DEM files
        outline_folder: Path to folder containing outline/segmentation files
        range_check: Range to check for overlaps in pixels
        VCO: Elevation VCO for overlap detection
        output_base: Base output directory
        show_visualizations: Whether to display images during processing (default=False)
        min_pcd: Minimum PCD to keep grains (if None, will prompt user)
    """
    dem_files = sorted(glob.glob(os.path.join(dem_folder, "*.tif")))
    outline_files_npy = sorted(glob.glob(os.path.join(outline_folder, "*.npy")))
    outline_files_tif = sorted(glob.glob(os.path.join(outline_folder, "*.tif")))
    outline_files = outline_files_npy + outline_files_tif

    print(f"Found {len(dem_files)} DEM files")
    print(f"Found {len(outline_files)} outline files")

    num_pairs = min(len(dem_files), len(outline_files))
    print(f"Will process {num_pairs} file pairs")

    user_input = input("Do you want to exclude edge clasts? (yes/no): ").strip().lower()
    
    if min_pcd is None:
        user_input_min_grains = input("Do you want to remove small clasts? (yes/no): ").strip().lower()
        if user_input_min_grains == "yes":
            min_pcd = float(input("Enter minimum PCD to keep (in pixels): "))
        else:
            min_pcd = None
    
    for i in range(num_pairs):
        dem_path = dem_files[i]
        outline_path = outline_files[i]
        
        dem_basename = os.path.basename(dem_path)
        outline_basename = os.path.basename(outline_path)
        dem_name = os.path.splitext(dem_basename)[0]
        
        print(f"\n\n===== Processing pair {i+1}/{num_pairs} =====")
        print(f"DEM file: {dem_basename}")
        print(f"Outline file: {outline_basename}")
        
        # Determine file format based on extension
        if outline_path.lower().endswith('.npy'):
            file_format = "NPY"
        elif outline_path.lower().endswith('.tif'):
            file_format = "TIF"
        else:
            print(f"Unknown outline file format for {outline_basename}, skipping...")
            continue
        
        print(f"Using {file_format} file: {outline_basename}")
        image_output_folder = os.path.join(output_base, dem_name)
        os.makedirs(image_output_folder, exist_ok=True)

        vis_folder = os.path.join(image_output_folder, "visualizations")
        os.makedirs(vis_folder, exist_ok=True)
        
        images_folder = os.path.join(image_output_folder, "images")
        csv_folder = os.path.join(image_output_folder, "csv")
        csv_path = os.path.join(csv_folder, "overlap_points.csv")
        
        print(f"\n\n===== Processing {dem_name} and outline =====")
        
        print("=== Running Python Overlap Analysis ===")
        dem = load_dem(dem_path)
        dem_vis_path = os.path.join(vis_folder, "dem.png")
        show_dem(dem, output_path=dem_vis_path, show=show_visualizations, close_after=True)
        
        if file_format == "NPY":
            outline = read_npy_file(outline_path)
            outline_vis_path = os.path.join(vis_folder, "original_outline.png")
            show_outline(outline, output_path=outline_vis_path, show=show_visualizations, close_after=True)
            
            if user_input == "yes":
                outline = remove_edge_grains(outline, show_comparison=show_visualizations, close_after=True)
                filtered_outline_vis_path = os.path.join(vis_folder, "filtered_outline_excluded_edge_clasts.png")
                show_outline(outline, output_path=filtered_outline_vis_path, show=show_visualizations, close_after=True)

                if user_input_min_grains == "yes":
                    outline = remove_small_grains(outline, min_pcd=min_pcd, show_comparison=show_visualizations, close_after=True)
                    filtered_outline_vis_path = os.path.join(vis_folder, "filtered_outline_excluded_small_clasts.png")
                    show_outline(outline, output_path=filtered_outline_vis_path, show=show_visualizations, close_after=True)
        
        elif file_format == "TIF":
            outline = load_outline(outline_path, show_ids = False)
            outline_vis_path = os.path.join(vis_folder, "original_outline.png")
            show_outline(outline, output_path=outline_vis_path, show=show_visualizations, close_after=True)
            
            if user_input == "yes":
                outline = remove_edge_grains(outline, show_comparison=show_visualizations, close_after=True)
                filtered_outline_vis_path = os.path.join(vis_folder, "filtered_outline_excluded_edge_clasts.png")
                show_outline(outline, output_path=filtered_outline_vis_path, show=show_visualizations, close_after=True)

                if user_input_min_grains == "yes":
                    outline = remove_small_grains(outline, min_pcd=min_pcd, show_comparison=show_visualizations, close_after=True)
                    filtered_outline_vis_path = os.path.join(vis_folder, "filtered_outline_excluded_small_clasts.png")
                    show_outline(outline, output_path=filtered_outline_vis_path, show=show_visualizations, close_after=True)

        grain_boundaries = find_boundaries(outline)
        boundaries_vis_path = os.path.join(vis_folder, "boundaries_on_dem.png")
        plot_boundaries_on_dem(dem, grain_boundaries, outline, 
                              output_path=boundaries_vis_path, show=show_visualizations, close_after=True)
        
        overlap_results = analyse_overlap(grain_boundaries, dem, outline, range_check, VCO)
        overlap_vis_path = os.path.join(vis_folder, "overlap_points.png")
        visualize_overlap(overlap_results, grain_boundaries, outline,
                         output_path=overlap_vis_path, show=show_visualizations, close_after=True)
        
        
        print(f"\n=== Saving grain images and overlap points to {image_output_folder} ===")
        save_grain_images_and_overlap_csv(overlap_results, outline, image_output_folder)

        print(f"Verifying output files...")
        print(f"  Images folder: {images_folder} - Exists: {os.path.exists(images_folder)}")
        print(f"  CSV file: {csv_path} - Exists: {os.path.exists(csv_path)}")

        if not os.path.exists(images_folder):
            print(f"ERROR: Images folder not created at {images_folder}")
            continue 
        elif not os.path.exists(csv_path):
            print(f"ERROR: CSV file not created at {csv_path}")
            continue  
        else:
            image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
            print(f"  Found {len(image_files)} image files in {images_folder}")
            
            if len(image_files) == 0:
                print(f"ERROR: No image files found in {images_folder}")
                continue 

            print(f"Output verification complete - continuing to MATLAB analysis")
        
        print("\n=== Starting MATLAB Angularity Analysis ===")
        try:
            import matlab.engine
            print("MATLAB engine imported successfully")

            print("Starting MATLAB engine...")
            eng = matlab.engine.start_matlab()

            matlab_script_dir = os.path.abspath("./MatlabZheng2015")
            images_folder_abs = os.path.abspath(images_folder)
            csv_path_abs = os.path.abspath(csv_path)

            print(f"Running Wadell_roundness with:")
            print(f"  - Images folder: {images_folder_abs}")
            print(f"  - Overlap CSV: {csv_path_abs}")
            
            eng.cd(matlab_script_dir)
            
            eng.Wadell_roundness(images_folder_abs, csv_path_abs, 
                                float(matlab_tol), 
                                float(matlab_factor), 
                                float(matlab_span), 
                                float(matlab_exclusion_range), 
                                nargout=0)

            analysis_folder = os.path.join(image_output_folder, "analysis_results")
            if os.path.exists(analysis_folder):
                result_files = os.listdir(analysis_folder)
                print(f"\nGenerated {len(result_files)} output files in {analysis_folder}:")
                
                csv_files = [f for f in result_files if f.endswith('.csv')]
                png_files = [f for f in result_files if f.endswith('.png')]
                
                if csv_files:
                    print(f"  - Results CSV: {csv_files[0]}")
                if png_files:
                    print(f"  - Grain analysis images: {len(png_files)} files")
            
            eng.quit()
            
        except Exception as e:
            print(f"ERROR during MATLAB processing: {e}")
        
        print(f"===== Completed analysis for {dem_name} =====\n")


def process_single_picture(dem_path, outline_path, range_check: int = 5, VCO: float = 0.002, 
                          matlab_tol: float = 0.05, matlab_factor: float = 0.98, 
                          matlab_span: float = 0.07, matlab_exclusion_range: int = 5, 
                          output_folder: str = "./overlap_output"):    
    """
    Process a single DEM and outline file.
    
    Parameters:
        dem_path: Path to the DEM file
        outline_path: Path to the outline/segmentation file
        range_check: Range to check for overlaps in pixels (default=5)
        VCO: Elevation VCO for overlap detection (default=0.002)
        output_folder: Base output directory for results
    """
    dem_basename = os.path.basename(dem_path)
    dem_name = os.path.splitext(dem_basename)[0]
    
    image_output_folder = os.path.join(output_folder, dem_name)
    os.makedirs(image_output_folder, exist_ok=True)

    vis_folder = os.path.join(image_output_folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)
    
    images_folder = os.path.join(image_output_folder, "images")
    csv_folder = os.path.join(image_output_folder, "csv")
    csv_path = os.path.join(csv_folder, "overlap_points.csv")
    
    print("=== Running Python Overlap Analysis ===")
    dem = load_dem(dem_path)
    dem_vis_path = os.path.join(vis_folder, "dem.png")
    show_dem(dem, output_path=dem_vis_path)
    
    if outline_path.lower().endswith('.npy'):
            outline = read_npy_file(outline_path)
            # Save outline visualization
            outline_vis_path = os.path.join(vis_folder, "outline.png")
            show_outline(outline, output_path=outline_vis_path)
            
            user_input_npy = input(f"Would you like to exclude the grains that touch the image edge? (yes/no): ").strip().lower()
            if user_input_npy == "yes":
                outline = remove_edge_grains(outline, show_comparison=True)
                # Save filtered outline
                filtered_vis_path = os.path.join(vis_folder, "filtered_outline.png")
                show_outline(outline, output_path=filtered_vis_path)

            input_user_min_grains = input("Do you want to remove small grains? (yes/no): ").strip().lower()
            if input_user_min_grains == "yes":
                min_pcd = float(input("Enter minimum PCD to keep (in pixels): "))
                outline = remove_small_grains(outline, min_pcd=min_pcd, show_comparison=True)
                small_filtered_vis_path = os.path.join(vis_folder, "filtered_outline_small_clasts.png")
                show_outline(outline, output_path=small_filtered_vis_path)


    elif outline_path.lower().endswith('.tif'):
            outline = load_outline(outline_path)
            # Save outline visualization
            outline_vis_path = os.path.join(vis_folder, "outline.png")
            show_outline(outline, output_path=outline_vis_path)
            user_input_npy = input(f"Would you like to exclude the grains that touch the image edge? (yes/no): ").strip().lower()

            if user_input_npy == "yes":
                outline = remove_edge_grains(outline, show_comparison=True)
                filtered_vis_path = os.path.join(vis_folder, "filtered_outline.png")
                show_outline(outline, output_path=filtered_vis_path)
            
            input_user_min_grains = input("Do you want to remove small grains? (yes/no): ").strip().lower()
            if input_user_min_grains == "yes":
                min_pcd = float(input("Enter minimum PCD to keep (in pixels): "))
                outline = remove_small_grains(outline, min_pcd=min_pcd, show_comparison=True)
                small_filtered_vis_path = os.path.join(vis_folder, "filtered_outline_small_clasts.png")
                show_outline(outline, output_path=small_filtered_vis_path)

    
    grain_boundaries = find_boundaries(outline)
    boundaries_vis_path = os.path.join(vis_folder, "boundaries_on_dem.png")
    plot_boundaries_on_dem(dem, grain_boundaries, outline, output_path=boundaries_vis_path)
    
    overlap_results = analyse_overlap(grain_boundaries, dem, outline, range_check, VCO)
    overlap_vis_path = os.path.join(vis_folder, "overlap_points.png")
    visualize_overlap(overlap_results, grain_boundaries, outline, output_path=overlap_vis_path)
    
    user_input = input("\nDo you want to proceed with saving and MATLAB analysis? (yes/no): ").strip().lower()
    if user_input != "yes":
        print("Analysis stopped by user.")
        return
    
    print(f"\n=== Saving grain images and overlap points to {image_output_folder} ===")
    save_grain_images_and_overlap_csv(overlap_results, outline, image_output_folder)

    print(f"Verifying output files...")
    print(f"  Images folder: {images_folder} - Exists: {os.path.exists(images_folder)}")
    print(f"  CSV file: {csv_path} - Exists: {os.path.exists(csv_path)}")

    if not os.path.exists(images_folder):
        print(f"ERROR: Images folder not created at {images_folder}")
        return
    elif not os.path.exists(csv_path):
        print(f"ERROR: CSV file not created at {csv_path}")
        return
    else:
        image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
        print(f"  Found {len(image_files)} image files in {images_folder}")
        
        if len(image_files) == 0:
            print(f"ERROR: No image files found in {images_folder}")
            return
        
        print(f"Output verification complete - continuing to MATLAB analysis")
    
    print("\n=== Starting MATLAB Angularity Analysis ===")
    try:
        import matlab.engine
        print("MATLAB engine imported successfully")
        
        print("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        
        matlab_script_dir = os.path.abspath("./MatlabZheng2015")
        images_folder_abs = os.path.abspath(images_folder)
        csv_path_abs = os.path.abspath(csv_path)
        
        print(f"Running Wadell_roundness with:")
        print(f"  - Images folder: {images_folder_abs}")
        print(f"  - Overlap CSV: {csv_path_abs}")
        
        eng.cd(matlab_script_dir)
        
        eng.Wadell_roundness(images_folder_abs, csv_path_abs, 
                            float(matlab_tol), 
                            float(matlab_factor), 
                            float(matlab_span), 
                            float(matlab_exclusion_range), 
                            nargout=0)
        
        analysis_folder = os.path.join(image_output_folder, "analysis_results")
        if os.path.exists(analysis_folder):
            result_files = os.listdir(analysis_folder)
            print(f"\nGenerated {len(result_files)} output files in {analysis_folder}:")
            
            csv_files = [f for f in result_files if f.endswith('.csv')]
            png_files = [f for f in result_files if f.endswith('.png')]
            
            if csv_files:
                print(f"  - Results CSV: {csv_files[0]}")
            if png_files:
                print(f"  - Grain analysis images: {len(png_files)} files")
        
        eng.quit()
    
    except ImportError:
        print("ERROR: Could not import MATLAB engine. Make sure it's properly installed.")
        print("Install using: cd /Applications/MATLAB_R2024a.app/extern/engines/python && python setup.py install")
    
    except Exception as e:
        print(f"ERROR during MATLAB processing: {e}")
    
    print(f"===== Completed analysis for {dem_basename} =====\n")


if __name__ == "__main__":
    # Example usage
    DEM = load_dem("Path_to_your_DEM.tif")
    show_dem(DEM)

    # If you have a NPY file, you can load it like this:
    # outline = read_npy_file("Path_to_your_segmentation.npy", dem_file_path="Path_to_your_DEM.tif")
    outline = load_outline("Path_to_your_outline.tif")
    show_outline(outline)


    grain_boundaries = find_boundaries(outline)
    plot_boundaries_on_dem(DEM, grain_boundaries, outline)

    overlap_results = analyse_overlap(grain_boundaries, DEM, outline, 4, 0.001)
    visualize_overlap(overlap_results, grain_boundaries, outline)

    save_grain_images_and_overlap_csv(overlap_results, outline)
    

