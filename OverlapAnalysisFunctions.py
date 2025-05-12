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


def show_dem(dem):
    """
    Visualizes the Digital Elevation Model using matplotlib with a rainbow colormap.

    Parameters: NumPy array containing the DEM data
    Visualization: Creates a figure with colorbar showing elevation values
    """
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(dem, cmap='rainbow')  # or 'jet'
        plt.colorbar()
        plt.title("Digital Elevation Model")
        plt.gcf().canvas.toolbar.zoom()
        plt.show()
    
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

def remove_edge_grains(outline, show_comparison=False):
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
    
    # Get dimensions of the image
    height, width = outline.shape
    
    # Extract all four borders
    top_border = outline[0, :]
    bottom_border = outline[height-1, :]
    left_border = outline[:, 0]
    right_border = outline[:, width-1]
    
    # Combine all borders and find unique grain IDs
    all_border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
    edge_grain_ids = np.unique(all_border_pixels)
    edge_grain_ids = edge_grain_ids[edge_grain_ids > 0]  # Exclude background (0)
    
    # Create a filtered copy of the outline
    filtered_outline = outline.copy()
    
    # Count the grains before filtering
    original_grain_ids = np.unique(outline)
    original_grain_ids = original_grain_ids[original_grain_ids > 0]
    
    # Remove edge-touching grains
    for grain_id in edge_grain_ids:
        filtered_outline[filtered_outline == grain_id] = 0
    
    # Count the grains after filtering
    remaining_grain_ids = np.unique(filtered_outline)
    remaining_grain_ids = remaining_grain_ids[remaining_grain_ids > 0]
    
    # Print statistics
    removed_count = len(edge_grain_ids)
    total_count = len(original_grain_ids)
    remaining_count = len(remaining_grain_ids)
    
    print(f"Found {removed_count} grains touching the image edge out of {total_count} total grains")
    print(f"Removed IDs: {edge_grain_ids[:10]}{'...' if len(edge_grain_ids) > 10 else ''}")
    print(f"Remaining grains: {remaining_count}")
    
    # Show comparison if requested
    if show_comparison:
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(outline, cmap='viridis')
        plt.title(f"Original Segmentation\n({total_count} grains)")
        plt.colorbar(label='Grain ID')
        
        plt.subplot(1, 2, 2)
        plt.imshow(filtered_outline, cmap='viridis')
        plt.title(f"Edge Grains Removed\n({remaining_count} grains)")
        plt.colorbar(label='Grain ID')
        
        # Highlight removed grains in the original image
        removed_mask = np.zeros_like(outline, dtype=bool)
        for grain_id in edge_grain_ids:
            removed_mask = removed_mask | (outline == grain_id)
        
        plt.subplot(1, 2, 1)
        plt.imshow(removed_mask, cmap='autumn', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return filtered_outline


def show_outline(outline):
    """ 
    Visualizes the grain segmentation image using matplotlib with a rainbow colormap.

    Parameters: NumPy array containing grain segmentation data
    Visualization: Creates a figure with colorbar showing grain IDs
    """
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(outline, cmap='rainbow')  # or 'jet'
        plt.colorbar()
        plt.title("Analysed Image with ImageGrains")
        plt.gcf().canvas.toolbar.zoom()
        plt.show()
    
    except outline is None:
        raise ValueError("ERROR: DEM data is None. Please check the file path or format.")

    except not isinstance(outline, np.ndarray):
        raise ValueError("ERROR: DEM data is not a valid NumPy array. Please check the file format.")
    
    except outline.size == 0:
        raise ValueError("ERROR: DEM data is empty. Please check the file path or format.")


def check_dimension(dem, outline):
    """
    Compares the dimensions of the DEM and outline arrays to ensure they match.

    Parameters: DEM array and outline array
    Output: Warning message if dimensions don't match
    """
    if dem.shape != outline.shape:
        print(f"WARNING: DEM shape {dem.shape} doesn't match outline shape {outline.shape}")
        print("This may cause misalignment issues!")


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


def plot_boundaries_on_dem(dem, grain_boundaries):
    """
    Plots the grain boundaries on top of the DEM for visualization.

    Parameters: 
        DEM array and grain boundaries dictionary
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

    plt.figure(figsize=(10, 10))
    plt.imshow(dem, cmap='rainbow', alpha=0.5)  # or 'jet'
    
    for grain_id, contours in grain_boundaries.items():
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=1)
    
    plt.title("Grain Boundaries on DEM")
    plt.colorbar(label='Elevation')
    plt.show()


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

                            for d in range(1, range_check): # check for overlap in the range of the normal vector (1 to range_check away from the point in the direction of the normal vector)
                                sample_y = int(round(y + d * normal_y))
                                sample_x = int(round(x + d * normal_x))

                                if (0 <= sample_y < DEM.shape[0] and 0 <= sample_x < DEM.shape[1]): # Check if sample is within DEM bounds
                                    sample_elevation = DEM[sample_y, sample_x]
                                    boundary_elevation = DEM[int(round(y)),int(round(x))]
                                    elevation_difference = sample_elevation - boundary_elevation


                                    if (elevation_difference > VCO and outline[sample_y, sample_x] != 0 and outline[sample_y, sample_x] != grain_id):
                                        overlapped_grain_id = outline[sample_y, sample_x]
                                        
                                        if grain_id != overlapped_grain_id:
                                            if grain_id not in overlap_results:
                                                overlap_results[grain_id] = []
                                            
                                            if (x,y) not in overlap_results[grain_id]:
                                                overlap_results[grain_id].append((x,y))
                                        
                                            break
    return overlap_results


def visualize_overlap(overlap_results, grain_boundaries, outline):
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
    plt.show()


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


def process_folder(dem_folder, outline_folder, range_check, VCO, output_base="./overlap_output"):
    """
    Process all DEM and outline files in the specified folders.
    
    Parameters:
        dem_folder: Path to folder containing DEM files
        outline_folder: Path to folder containing outline/segmentation files
        range_check: Range to check for overlaps in pixels
        VCO: Elevation VCO for overlap detection
        output_base: Base output directory
    """
    dem_files = glob.glob(os.path.join(dem_folder, "*.tif"))
    print(f"Found {len(dem_files)} DEM files to process")
    user_input = input("Do you want to exclude edge clasts? (yes/no): ").strip().lower()
    
    for dem_path in dem_files:
        dem_basename = os.path.basename(dem_path)
        dem_name = os.path.splitext(dem_basename)[0]
        
        outline_pattern_npy = os.path.join(outline_folder, f"{dem_name}*.npy")
        outline_pattern_tif = os.path.join(outline_folder, f"{dem_name}*.tif")
        
        outline_matches_npy = glob.glob(outline_pattern_npy)
        outline_matches_tif = glob.glob(outline_pattern_tif)
        
        if outline_matches_npy:
            outline_path = outline_matches_npy[0]
            file_format = "NPY"
            if len(outline_matches_npy) > 1:
                print(f"Multiple NPY matches found for {dem_basename}, using: {os.path.basename(outline_path)}")
        elif outline_matches_tif:
            outline_path = outline_matches_tif[0]
            file_format = "TIF"
            if len(outline_matches_tif) > 1:
                print(f"Multiple TIF matches found for {dem_basename}, using: {os.path.basename(outline_path)}")
        else:
            print(f"No matching outline file found for {dem_basename}, skipping...")
            continue
        
        print(f"Using {file_format} file for {dem_basename}: {os.path.basename(outline_path)}")
        
        image_output_folder = os.path.join(output_base, dem_name)
        os.makedirs(image_output_folder, exist_ok=True)
        
        images_folder = os.path.join(image_output_folder, "images")
        csv_folder = os.path.join(image_output_folder, "csv")
        csv_path = os.path.join(csv_folder, "overlap_points.csv")
        
        print(f"\n\n===== Processing {dem_name} =====")
        
        print("=== Running Python Overlap Analysis ===")
        dem = load_dem(dem_path)
        show_dem(dem)
        
        if file_format == "NPY":
            outline = read_npy_file(outline_path)
            if user_input == "yes":
                outline = remove_edge_grains(outline, show_comparison=False)

        else: 
            outline = load_outline(outline_path)
            if user_input == "yes":
                outline = remove_edge_grains(outline, show_comparison=False)
            
        check_dimension(dem, outline)
        
        grain_boundaries = find_boundaries(outline)
        plot_boundaries_on_dem(dem, grain_boundaries)
        
        overlap_results = analyse_overlap(grain_boundaries, dem, outline, range_check, VCO)
        visualize_overlap(overlap_results, grain_boundaries, outline)
        
        user_input = input(f"\nDo you want to proceed with analysis of {dem_name}? (yes/no): ").strip().lower()
        if user_input != "yes":
            print(f"Analysis of {dem_name} skipped by user.")
            continue
        
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
            
            eng.Wadell_roundness(images_folder_abs, csv_path_abs, nargout=0)

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


def process_single_picture(dem_path, outline_path, range_check=5, VCO=0.002, output_folder="./overlap_output"):
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
    
    images_folder = os.path.join(image_output_folder, "images")
    csv_folder = os.path.join(image_output_folder, "csv")
    csv_path = os.path.join(csv_folder, "overlap_points.csv")
    
    print("=== Running Python Overlap Analysis ===")
    dem = load_dem(dem_path)
    show_dem(dem)
    
    if outline_path.lower().endswith('.npy'):
        outline = read_npy_file(outline_path)
        user_input_npy = input(f"Would you like to exclude the grains that touch the image edge? (yes/no): ").strip().lower()
        if user_input_npy == "yes":
            outline = remove_edge_grains(outline, show_comparison=True)
        if user_input_npy == "no":
            show_outline(outline)

    else:
        outline = load_outline(outline_path)
        user_input_tif = input(f"Would you like to exclude the grains that touch the image edge? (yes/no): ").strip().lower()
        if user_input_tif == "yes":
            outline = remove_edge_grains(outline, show_comparison=True)
        if user_input_tif == "no":
            show_outline(outline)
        
    check_dimension(dem, outline)
    
    grain_boundaries = find_boundaries(outline)
    plot_boundaries_on_dem(dem, grain_boundaries)
    
    overlap_results = analyse_overlap(grain_boundaries, dem, outline, range_check, VCO)
    visualize_overlap(overlap_results, grain_boundaries, outline)
    
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
        
        eng.Wadell_roundness(images_folder_abs, csv_path_abs, nargout=0)
        
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

    check_dimension(DEM, outline)

    grain_boundaries = find_boundaries(outline)
    plot_boundaries_on_dem(DEM, grain_boundaries)

    overlap_results = analyse_overlap(grain_boundaries, DEM, outline, 4, 0.001)
    visualize_overlap(overlap_results, grain_boundaries, outline)

    save_grain_images_and_overlap_csv(overlap_results, outline)
    

