from OverlapAnalysisFunctions import process_single_picture, process_folder
import os
import datetime
import glob

# Define optimized parameters for 2000x2000 pixel images
VCO = 1000  # Recommended VCO for overlap analysis in meters
range_check = 4    # Recommended range for overlap analysis in pixels

# MATLAB parameters 
matlab_tol = 0.015      # Forming straight lines to the boundary
matlab_factor = 0.98   # Fitting small circles 
matlab_span = 0.07     # Nonparametric fitting
matlab_exclusion_range = 3  # Radius for filtering convex points near overlaps


# Create timestamped output folder to keep analyses separate
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_base = f"./overlap_output_{timestamp}"

# Define default paths (can be modified by user input)
input_dem_path = "/Users/.../DEM.tif"
input_outline_path = "/Users/.../Outline.npy"
input_dem_folder = "/Users/.../DEM"
input_outline_folder = "/Users/.../NPY"

try:
    # Print header
    print("=" * 80)
    print("GRAIN OVERLAP AND ANGULARITY ANALYSIS")
    print("=" * 80)
    print(f"Output will be saved to: {output_base}")
    print("Parameters:")
    print(f"  - Elevation VCO: {VCO}")
    print(f"  - Range check: {range_check} pixels")
    print("=" * 80)
    
    # User choice for processing mode
    print("\nSelect processing mode:")
    print("1. Process a single image")
    print("2. Process all images in a folder")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Single image processing
        print("\n--- SINGLE IMAGE PROCESSING ---")
        
        # Process the single image with minimal output
        process_single_picture(input_dem_path, input_outline_path, 
                            range_check, VCO, 
                            matlab_tol, matlab_factor, matlab_span, matlab_exclusion_range,
                            output_base)
        
    elif choice == "2":
        # Folder processing
        print("\n--- FOLDER PROCESSING ---")
        
        # Confirm processing
        print(f"\nProcessing all images in:")
        print(f"  DEM folder: {input_dem_folder}")
        print(f"  Outline folder: {input_outline_folder}")
        confirmation = input("Continue? (yes/no): ").strip().lower()
        
        if confirmation == "yes":
            # Get all files from each folder
            dem_files = sorted(glob.glob(os.path.join(input_dem_folder, "*.tif")))
            outline_files = sorted(glob.glob(os.path.join(input_outline_folder, "*.npy"))) + \
               sorted(glob.glob(os.path.join(input_outline_folder, "*.tif")))
            print(f"Found {len(dem_files)} DEM files")
            print(f"Found {len(outline_files)} outline files")
            
            # Process the minimum number of pairs available
            num_pairs = min(len(dem_files), len(outline_files))
            print(f"Will process {num_pairs} file pairs")
            
            # With this single function call:
            process_folder(
                input_dem_folder, 
                input_outline_folder, 
                range_check, 
                VCO,
                matlab_tol=matlab_tol,
                matlab_factor=matlab_factor,
                matlab_span=matlab_span,
                matlab_exclusion_range=matlab_exclusion_range,
                output_base=output_base,
                show_visualizations=False  
)
                
            print(f"\nCompleted processing {num_pairs} image pairs")
        else:
            print("Operation cancelled by user.")
            
    else:
        print("Invalid choice. Please enter 1 or 2.")
        
except ImportError:
    print("ERROR: Could not import MATLAB engine. Make sure it's properly installed.")
    print("Install using: cd /Applications/MATLAB_R2024a.app/extern/engines/python && python setup.py install")
    
except Exception as e:
    print(f"ERROR during processing: {e}")
    
print("\n=== Analysis Complete ===")
print(f"Results saved to: {output_base}")