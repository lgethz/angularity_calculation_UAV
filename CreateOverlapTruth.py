import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button, RadioButtons
from matplotlib.path import Path
import pandas as pd
import os
from OverlapAnalysisFunctions import load_dem, load_outline, read_npy_file, find_boundaries

def main():
    """
    Main function to run the BoundarySelector tool.
    
    Parameters:
        None
    """
    #dem_path = "/Users/larsgubeli/Library/CloudStorage/OneDrive-ETHZurich/97_Bachelorarbeit/06_ImageGrainsAnalysis/DataforOverlapCalculation2000x2000/DEM/Image02_DEM.tif"
    ortho_path = "/Users/larsgubeli/Library/CloudStorage/OneDrive-ETHZurich/97_Bachelorarbeit/06_ImageGrainsAnalysis/DataforOverlapCalculation2000x2000/Orthomosaics/Image04_Orthomosaic.jpg"
    
    # Specify your outlines directory or file
    outlines_path = "/Users/larsgubeli/Library/CloudStorage/OneDrive-ETHZurich/97_Bachelorarbeit/06_ImageGrainsAnalysis/DataforOverlapCalculation2000x2000/ImageGrains/Image04_Orthomosaic_IG2coarse_pred.tif"
    
    # Use orthomosaic for better visualization with outlines
    selector = create_ground_truth(ortho_path, outlines_path)


# Enable matplotlib toolbar by default
plt.rcParams['toolbar'] = 'toolbar2'

class BoundarySelector:
    def __init__(self, image, outlines):
        """
        Interactive tool to select overlap regions from outlines
        
        Parameters:
            image: Background image or DEM
            outlines: Dictionary of outlines data keyed by grain ID
        """
        self.image = image
        self.original_outlines = outlines
        self.outlines = outlines.copy()
        self.selected_points = {}  # Dictionary to store selected points by grain ID
        self.rotation = 0  # Current rotation (0, 90, 180, 270)
        self.flip_h = False  # Horizontal flip
        self.flip_v = False  # Vertical flip
        self.selected_grain = None  # Currently selected grain
        self.selection_mode = "grain"  # 'grain' or 'lasso'
        self.selection_history = []  # Track lasso selection operations for undo
        
        # Get image dimensions
        if len(image.shape) == 3:
            self.height, self.width, _ = image.shape
        else:
            self.height, self.width = image.shape
        
        # Setup the figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Display the image
        self.ax.imshow(self.image, cmap='gray')
        
        # Plot all outlines
        self.plot_outlines()
        
        # Create the lasso selector (disabled initially)
        self.lasso = LassoSelector(self.ax, self.on_select, useblit=True)
        self.lasso.set_active(False)
        
        # Add click event for grain selection
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add save and clear buttons
        self.ax_save = plt.axes([0.8, 0.01, 0.1, 0.04])
        self.ax_clear = plt.axes([0.65, 0.01, 0.1, 0.04])
        self.btn_save = Button(self.ax_save, 'Save CSV')
        self.btn_clear = Button(self.ax_clear, 'Clear')
        self.btn_save.on_clicked(self.save_csv)
        self.btn_clear.on_clicked(self.clear_selection)
        
        # Add undo button
        self.ax_undo = plt.axes([0.35, 0.01, 0.1, 0.04])
        self.btn_undo = Button(self.ax_undo, 'Undo')
        self.btn_undo.on_clicked(self.undo_last_selection)
        
        # Add grain selection toggle button
        self.ax_mode = plt.axes([0.5, 0.01, 0.1, 0.04])
        self.btn_mode = Button(self.ax_mode, 'Mode: Grain')
        self.btn_mode.on_clicked(self.toggle_selection_mode)
        
        # Add rotation buttons
        self.ax_rot_label = plt.figtext(0.01, 0.98, "Rotation:", size=10)
        self.ax_rot90 = plt.axes([0.01, 0.93, 0.1, 0.04])
        self.ax_rot180 = plt.axes([0.01, 0.88, 0.1, 0.04])
        self.ax_rot270 = plt.axes([0.01, 0.83, 0.1, 0.04])
        self.btn_rot90 = Button(self.ax_rot90, 'Rotate 90°')
        self.btn_rot180 = Button(self.ax_rot180, 'Rotate 180°')
        self.btn_rot270 = Button(self.ax_rot270, 'Rotate 270°')
        self.btn_rot90.on_clicked(lambda event: self.rotate_outlines(90))
        self.btn_rot180.on_clicked(lambda event: self.rotate_outlines(180))
        self.btn_rot270.on_clicked(lambda event: self.rotate_outlines(270))
        
        # Add flip buttons
        self.ax_flip_h = plt.axes([0.01, 0.78, 0.1, 0.04])
        self.ax_flip_v = plt.axes([0.01, 0.73, 0.1, 0.04])
        self.btn_flip_h = Button(self.ax_flip_h, 'Flip Horiz')
        self.btn_flip_v = Button(self.ax_flip_v, 'Flip Vert')
        self.btn_flip_h.on_clicked(self.flip_horizontal)
        self.btn_flip_v.on_clicked(self.flip_vertical)
        
        # Add reset button
        self.ax_reset = plt.axes([0.01, 0.68, 0.1, 0.04])
        self.btn_reset = Button(self.ax_reset, 'Reset Transform')
        self.btn_reset.on_clicked(self.reset_transform)
        
        # Add clear grain selection button
        self.ax_clear_grain = plt.axes([0.01, 0.63, 0.1, 0.04])
        self.btn_clear_grain = Button(self.ax_clear_grain, 'Clear Grain')
        self.btn_clear_grain.on_clicked(self.clear_grain_selection)
        
        # Add status display
        self.status_text = self.fig.text(0.01, 0.57, 
                                        "Mode: Grain Selection\nTransform: None", 
                                        bbox=dict(facecolor='white', alpha=0.5))
        
        # Instructions
        plt.suptitle('First select a grain, then use lasso to select overlap regions', fontsize=14)
        
        plt.show()
    
    def toggle_selection_mode(self, event):
        """Toggle between grain selection and lasso selection modes"""
        if self.selection_mode == "grain":
            self.selection_mode = "lasso"
            self.btn_mode.label.set_text('Mode: Lasso')
            if self.selected_grain is not None:
                self.lasso.set_active(True)
        else:
            self.selection_mode = "grain"
            self.btn_mode.label.set_text('Mode: Grain')
            self.lasso.set_active(False)
            # Note: We don't clear the grain selection here, allowing user to switch modes for the same grain
        
        self.update_status_text()
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        """Handle mouse clicks for grain selection"""
        if not event.inaxes or self.selection_mode != "grain":
            return
        
        # Disable the lasso temporarily if active
        lasso_was_active = self.lasso.active
        if lasso_was_active:
            self.lasso.set_active(False)
        
        click_point = np.array([event.xdata, event.ydata])
        
        # First try to find grain that contains the click point
        grain_found = False
        for grain_id, outline in self.outlines.items():
            # Check if click point is inside the grain outline
            path = Path(outline)
            if path.contains_point(click_point):
                self.selected_grain = grain_id
                grain_found = True
                print(f"Selected grain (inside): {self.selected_grain}")
                break
        
        # If no grain contains the point, find the closest outline
        if not grain_found:
            min_dist = float('inf')
            closest_grain = None
            
            for grain_id, outline in self.outlines.items():
                # Calculate distances from click point to all points in outline
                distances = np.linalg.norm(outline - click_point, axis=1)
                min_point_dist = np.min(distances)
                
                if min_point_dist < min_dist:
                    min_dist = min_point_dist
                    closest_grain = grain_id
            
            # If click is close enough to an outline, select that grain
            if min_dist < 20:  # Threshold in pixels
                self.selected_grain = closest_grain
                print(f"Selected grain (nearby): {self.selected_grain}")
                grain_found = True
        
        if grain_found:
            # Enable lasso for the selected grain but keep in grain selection mode
            # This allows selecting another grain if needed
            self.lasso.set_active(True)
        else:
            print("No grain selected")
        
        self.update_display()

    def clear_grain_selection(self, event):
        """Clear the current grain selection"""
        self.selected_grain = None
        self.lasso.set_active(False)
        self.selection_mode = "grain"
        self.btn_mode.label.set_text('Mode: Grain')
        self.update_display()
    
    def plot_outlines(self):
        """Plot all outlines with optional highlighting for selected grain"""
        for grain_id, outline in self.outlines.items():
            if grain_id == self.selected_grain:
                # Highlight selected grain
                self.ax.plot(outline[:, 0], outline[:, 1], 'b-', linewidth=2)
            else:
                self.ax.plot(outline[:, 0], outline[:, 1], 'r-', linewidth=1)
    
    def update_status_text(self):
        """Update the transform status text"""
        mode_text = f"Mode: {'Lasso Selection' if self.selection_mode == 'lasso' else 'Grain Selection'}"
        if self.selected_grain is not None:
            mode_text += f"\nSelected Grain: {self.selected_grain}"
        
        transform_text = f"Rotation: {self.rotation}°"
        if self.flip_h:
            transform_text += ", Flipped H"
        if self.flip_v:
            transform_text += ", Flipped V"
        
        self.status_text.set_text(f"{mode_text}\n{transform_text}")
    
    def rotate_outlines(self, degrees):
        """Rotate all outlines by the specified degrees"""
        self.rotation = (self.rotation + degrees) % 360
        self.apply_transforms()
        self.update_display()
    
    def flip_horizontal(self, event):
        """Flip outlines horizontally"""
        self.flip_h = not self.flip_h
        self.apply_transforms()
        self.update_display()
    
    def flip_vertical(self, event):
        """Flip outlines vertically"""
        self.flip_v = not self.flip_v
        self.apply_transforms()
        self.update_display()
    
    def reset_transform(self, event):
        """Reset all transformations"""
        self.rotation = 0
        self.flip_h = False
        self.flip_v = False
        self.outlines = self.original_outlines.copy()
        self.update_display()
    
    def apply_transforms(self):
        """Apply all current transformations to the original outlines"""
        # Start with copies of the original outlines
        self.outlines = {id: outline.copy() for id, outline in self.original_outlines.items()}
        
        # Apply rotation if needed
        if self.rotation != 0:
            for grain_id, outline in self.outlines.items():
                # For each rotation of 90 degrees, swap x and y and adjust coordinates
                if self.rotation == 90:
                    # (x, y) -> (y, width - x)
                    new_points = np.zeros_like(outline)
                    new_points[:, 0] = outline[:, 1]
                    new_points[:, 1] = self.width - outline[:, 0]
                    self.outlines[grain_id] = new_points
                elif self.rotation == 180:
                    # (x, y) -> (width - x, height - y)
                    new_points = np.zeros_like(outline)
                    new_points[:, 0] = self.width - outline[:, 0]
                    new_points[:, 1] = self.height - outline[:, 1]
                    self.outlines[grain_id] = new_points
                elif self.rotation == 270:
                    # (x, y) -> (height - y, x)
                    new_points = np.zeros_like(outline)
                    new_points[:, 0] = self.height - outline[:, 1]
                    new_points[:, 1] = outline[:, 0]
                    self.outlines[grain_id] = new_points
        
        # Apply horizontal flip if needed
        if self.flip_h:
            for grain_id, outline in self.outlines.items():
                outline[:, 0] = self.width - outline[:, 0]
        
        # Apply vertical flip if needed
        if self.flip_v:
            for grain_id, outline in self.outlines.items():
                outline[:, 1] = self.height - outline[:, 1]
    
    def on_select(self, verts):
        """When lasso selection is made"""
        if self.selected_grain is None:
            print("Please select a grain first")
            return
        
        # Create a Path from the vertices
        p = Path(verts)
        
        # Get the outline for the selected grain
        outline = self.outlines.get(self.selected_grain)
        if outline is None:
            return
        
        # Find points inside lasso
        inside_mask = p.contains_points(outline)
        
        # Store the selected points
        points_added = []  # Track points added in this operation for undo functionality
        
        if np.any(inside_mask):
            selected = outline[inside_mask]
            
            if self.selected_grain not in self.selected_points:
                self.selected_points[self.selected_grain] = []
            
            # Add new points to existing selected points
            for point in selected:
                # Check if point already exists
                point_exists = False
                for existing_point in self.selected_points[self.selected_grain]:
                    if np.array_equal(point, existing_point):
                        point_exists = True
                        break
                
                if not point_exists:
                    self.selected_points[self.selected_grain].append(point)
                    points_added.append(point)  # Track this point for undo
        
        # Add to history only if points were actually added
        if points_added:
            self.selection_history.append((self.selected_grain, points_added))
            print(f"Added {len(points_added)} points to grain {self.selected_grain}")
        
        # Update the display
        self.update_display()
    
    def undo_last_selection(self, event):
        """Remove the most recent lasso selection"""
        if not self.selection_history:
            print("Nothing to undo")
            return
            
        # Get the last selection
        grain_id, points_added = self.selection_history.pop()
        print(f"Undoing last selection for grain {grain_id} ({len(points_added)} points)")
        
        # Remove these points from the selection
        if grain_id in self.selected_points:
            # Convert to list for easier manipulation
            current_points = list(self.selected_points[grain_id])
            
            # Remove each point that was added in the last selection
            for point in points_added:
                for i, existing_point in enumerate(current_points):
                    if np.array_equal(point, existing_point):
                        current_points.pop(i)
                        break
            
            # Update the selected points
            if current_points:
                self.selected_points[grain_id] = current_points
            else:
                # If no points left, remove the grain from selected_points
                del self.selected_points[grain_id]
        
        # Update the display
        self.update_display()
    
    def update_display(self):
        """Update the display to show selected regions"""
        # Store current axis limits before clearing
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Clear the current display
        self.ax.clear()
        
        # Show the image
        self.ax.imshow(self.image, cmap='gray')
        
        # Plot all outlines
        self.plot_outlines()
        
        # Plot selected points as green dots
        for grain_id, points in self.selected_points.items():
            if points:
                points_array = np.array(points)
                self.ax.plot(points_array[:, 0], points_array[:, 1], 'go', markersize=3)
        
        # Restore previous zoom level
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
        # Update status text
        self.update_status_text()
        
        self.fig.canvas.draw_idle()
    
    def clear_selection(self, event):
        """Clear the current selection"""
        self.selected_points = {}
        self.selection_history = []  # Clear history when clearing all selections
        self.update_display()
    
    def save_csv(self, event):
        """Save the selected overlap points to a CSV file"""
        filename = input("Enter filename to save the overlap points (will be saved as .csv): ")
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Prepare data for CSV
        data = []
        for grain_id, points in self.selected_points.items():
            for point in points:
                data.append([grain_id, point[0], point[1]])
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data, columns=['Grain_ID', 'X', 'Y'])
        df.to_csv(filename, index=False)
        print(f"Overlap points saved to {filename}")


def create_ground_truth(image_path, outlines_path=None):
    """
    Load an image and create ground truth for overlaps by selecting parts of outlines
    
    Parameters:
        image_path: Path to the DEM or image
        outlines_path: Path to a directory containing outline files (csv or npy)
    """
    # Load the image
    if image_path.endswith('.npy'):
        image = np.load(image_path)
    elif image_path.endswith('.tif'):
        # Use load_dem for tif files
        dem_data = load_dem(image_path)
        image = dem_data
    else:
        image = plt.imread(image_path)
    
    # Load outlines
    outlines = {}
    
    if outlines_path:
        if os.path.isdir(outlines_path):
            # If it's a directory, load all outline files
            for file in os.listdir(outlines_path):
                if file.endswith('.npy') or file.endswith('.tif'):
                    file_path = os.path.join(outlines_path, file)
                    try:
                        grain_id = int(file.split('_')[1].split('.')[0])  # Extract grain ID from filename
                        
                        # Load outline based on file type
                        if file.endswith('.npy'):
                            outline_array = read_npy_file(file_path)
                        else:  # TIF format
                            outline_array = load_outline(file_path, show_ids=False)
                            
                        # Get boundaries (contours) for this grain
                        grain_boundaries = find_boundaries(outline_array)
                        
                        # Check if this grain ID exists in boundaries
                        if grain_id in grain_boundaries:
                            outlines[grain_id] = grain_boundaries[grain_id][0]  # Use first contour
                    except Exception as e:
                        print(f"Error loading file {file}: {e}")
        else:
            # Direct file path to a single outlines file
            if outlines_path.endswith('.npy'):
                outline_array = read_npy_file(outlines_path)
            else:  # TIF format
                outline_array = load_outline(outlines_path, show_ids=False)
                
            # Get boundaries for all grains
            grain_boundaries = find_boundaries(outline_array)

            for grain_id, contours in grain_boundaries.items():
                if contours:
                    # Option 1: Use all contours instead of just the first one
                    if len(contours) > 1:
                        # Combine all contours into one array
                        all_points = np.vstack(contours)
                        outlines[grain_id] = all_points
                    else:
                        outlines[grain_id] = contours[0]
                        
                    # Option 2: Print information to debug
                    print(f"Grain {grain_id}: Found {len(contours)} contours with {sum(len(c) for c in contours)} total points")
    
    # Create the selector interface
    selector = BoundarySelector(image, outlines)
    
    return selector


# Example usage with your specific files:
if __name__ == "__main__":
    main()