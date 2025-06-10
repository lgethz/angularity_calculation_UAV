import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm
from OverlapAnalysisFunctions import load_dem, load_outline, read_npy_file, find_boundaries, analyse_overlap
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    """
    Performs multi-dataset parameter optimization for clast overlap detection.
    
    This function:
    1. Loads DEM, outline, and ground truth data from specified folders
    2. Identifies matching datasets using filename identifiers (format: X-Y)
    3. Tests multiple VCO and range_check parameter combinations on each dataset
    4. Evaluates performance using confusion matrices (TP, FP, TN, FN)
    5. Calculates accuracy, precision, recall, and F1 scores for each combination
    6. Generates individual heatmaps and performance metrics for each dataset
    7. Combines results across datasets to find globally optimal parameters
    8. Creates visualizations of average metrics across all datasets
    9. Saves comprehensive results to CSV files and images
    
    The function automatically creates a timestamped output directory containing:
    - A folder for each dataset with individual results and visualizations
    - An average_results.csv file with aggregated metrics
    - An average_f1_score_heatmap.png showing performance across parameters
    - A best_parameters.json file with optimal parameter values
    - A datasets_info.txt file documenting processed datasets
    
    Input parameters are defined at the top of the function:
    - dem_folder: Path to directory containing DEM files (.tif)
    - outline_folder: Path to directory containing clast outlines (.tif/.npy)
    - truth_folder: Path to directory containing ground truth files (.csv)
    - VCOs: List of VCO values to test
    - range_checks: List of range_check values to test
    
    Returns:
        None: Results are saved to disk rather than returned
    """
    # Define folders containing data - modify these paths as needed
    dem_folder = "/Users/.../DEM"
    outline_folder = "/Users/.../NPY"
    truth_folder = "/Users/.../ManuallyMarked"
    
    # Create a results directory with timestamp
    import datetime
    import re
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"parameter_analysis_results_{timestamp}"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Define parameter ranges to test
    VCOs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    range_checks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Find all files in each folder
    dem_files = [f for f in os.listdir(dem_folder) if f.endswith('.tif')]
    outline_files = [f for f in os.listdir(outline_folder) if f.endswith('.tif') or f.endswith('.npy')]
    truth_files = [f for f in os.listdir(truth_folder) if f.endswith('.csv')]
    
    print(f"Found {len(dem_files)} DEM files")
    print(f"Found {len(outline_files)} outline files")
    print(f"Found {len(truth_files)} ground truth files")
    
    # Extract identifiers from filenames
    # Pattern looking for X-Y format (like 0-2, 13-1, etc.)
    pattern = r'(\d+-\d+)'
    
    # Create dictionaries to store files by their identifier
    dem_dict = {}
    for f in dem_files:
        match = re.search(pattern, f)
        if match:
            identifier = match.group(1)
            dem_dict[identifier] = f
    
    outline_dict = {}
    for f in outline_files:
        match = re.search(pattern, f)
        if match:
            identifier = match.group(1)
            outline_dict[identifier] = f
    
    truth_dict = {}
    for f in truth_files:
        match = re.search(pattern, f)
        if match:
            identifier = match.group(1)
            truth_dict[identifier] = f
    
    # Create datasets by matching common identifiers
    datasets = []
    for identifier in dem_dict.keys():
        if identifier in outline_dict and identifier in truth_dict:
            datasets.append({
                'name': f"dataset_{identifier}",
                'dem': os.path.join(dem_folder, dem_dict[identifier]),
                'outline': os.path.join(outline_folder, outline_dict[identifier]),
                'truth': os.path.join(truth_folder, truth_dict[identifier])
            })
    
    print(f"Successfully matched {len(datasets)} complete datasets")
    
    # Display matched datasets
    for i, dataset in enumerate(datasets):
        print(f"\nDataset {i+1}: {dataset['name']}")
        print(f"  DEM: {os.path.basename(dataset['dem'])}")
        print(f"  Outline: {os.path.basename(dataset['outline'])}")
        print(f"  Ground Truth: {os.path.basename(dataset['truth'])}")
    
    # Rest of your function remains the same...
    
    # Dictionary to store results from all datasets
    all_datasets_results = []
    dataset_info = []
    
    # Process each matched dataset
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset['name']}")
        print(f"{'='*50}")
        
        dem_path = dataset['dem']
        outline_path = dataset['outline']
        ground_truth_path = dataset['truth']
        
        # Save dataset info
        dataset_info.append({
            'name': dataset['name'],
            'dem_file': os.path.basename(dem_path),
            'outline_file': os.path.basename(outline_path),
            'ground_truth_file': os.path.basename(ground_truth_path)
        })
        
        # Run analysis for this dataset
        try:
            results = analyze_parameters(
                dem_path=dem_path,
                outline_path=outline_path,
                ground_truth_path=ground_truth_path,
                VCOs=VCOs,
                range_checks=range_checks,
                verbose=False  # Reduce output verbosity for multiple datasets
            )
            
            # Add dataset identifier to results
            results['dataset'] = dataset['name']
            
            # Save individual results
            dataset_results_dir = os.path.join(results_dir, dataset['name'])
            os.makedirs(dataset_results_dir, exist_ok=True)
            
            # Save individual dataset results
            results.to_csv(os.path.join(dataset_results_dir, 'parameter_analysis_results.csv'), index=False)
            
            # Create individual heatmap
            visualize_f1_heatmap(
                results,
                save_path=os.path.join(dataset_results_dir, 'f1_score_heatmap.png'),
                highlight_best=True,
                show_plot=False
            )
            
            # Store results for aggregation
            all_datasets_results.append(results)
            
        except Exception as e:
            print(f"Error processing dataset {dataset['name']}: {str(e)}")
            continue
    
    # If no datasets were processed successfully, exit
    if not all_datasets_results:
        print("No datasets were processed successfully. Exiting.")
        return
    
    # Combine all results
    combined_results = pd.concat(all_datasets_results)
    
    # Calculate average F1 scores for each parameter combination
    avg_results = combined_results.groupby(['VCO', 'range_check']).agg({
        'TP': 'mean',
        'FP': 'mean',
        'TN': 'mean',
        'FN': 'mean',
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean'
    }).reset_index()
    
    # Save the averaged results
    avg_results.to_csv(os.path.join(results_dir, 'average_results.csv'), index=False)
    print(f"Average results saved to {os.path.join(results_dir, 'average_results.csv')}")
    
    # Save dataset information for reference
    with open(os.path.join(results_dir, 'datasets_info.txt'), 'w') as f:
        f.write(f"Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Number of datasets analyzed: {len(all_datasets_results)}\n\n")
        
        for i, info in enumerate(dataset_info):
            f.write(f"Dataset {i+1}: {info['name']}\n")
            f.write(f"  - DEM file: {info['dem_file']}\n")
            f.write(f"  - Outline file: {info['outline_file']}\n")
            f.write(f"  - Ground truth file: {info['ground_truth_file']}\n\n")
        
        f.write(f"Parameter ranges:\n")
        f.write(f"  - VCO values: {VCOs}\n")
        f.write(f"  - Range check values: {range_checks}\n")
    
    # Create heatmap of average F1 scores
    plt.figure(figsize=(14, 12))
    
    # Create pivot table for average F1 scores
    pivot_table = avg_results.pivot_table(
        values='f1_score', 
        index='VCO', 
        columns='range_check'
    )
    
    # Find best parameters
    best_idx = avg_results['f1_score'].idxmax()
    best_row = avg_results.loc[best_idx]
    best_threshold = best_row['VCO']
    best_range_check = best_row['range_check']
    best_f1 = best_row['f1_score']
    
    # Create the heatmap
    ax = sns.heatmap(
        pivot_table,
        annot=True, 
        cmap='viridis',
        fmt='.3f',
        linewidths=0.5,
        cbar_kws={'label': 'Average F1 Score'}
    )
    
    # Highlight the best parameter combination
    idx_y = list(pivot_table.index).index(best_threshold)
    idx_x = list(pivot_table.columns).index(best_range_check)
    ax.add_patch(plt.Rectangle((idx_x, idx_y), 1, 1, fill=False, edgecolor='red', lw=3))
    
    # Title and labels
    ax.set_title('Average F1 Score by VCO and Range Check (Across All Datasets)', fontsize=18, fontweight='bold')
    ax.set_ylabel('VCO', fontsize=14, fontweight='bold')
    ax.set_xlabel('Range Check', fontsize=14, fontweight='bold')
    
    # Add annotation for best parameters
    plt.figtext(0.5, 0.01, 
               f'Best Average F1 Score: {best_f1:.4f} (VCO={best_threshold}, Range Check={best_range_check})',
               ha='center', fontsize=14, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    
    # Save the average F1 score heatmap
    avg_heatmap_path = os.path.join(results_dir, 'average_f1_score_heatmap.png')
    plt.savefig(avg_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Average F1 score heatmap saved to {avg_heatmap_path}")
    
    plt.show()
    
    # Save best parameters
    best_params = {
        'VCO': float(best_threshold),
        'range_check': int(best_range_check),
        'performance': {
            'f1_score': float(best_f1),
            'accuracy': float(best_row['accuracy']),
            'precision': float(best_row['precision']),
            'recall': float(best_row['recall']),
        }
    }
    
    import json
    with open(os.path.join(results_dir, 'best_parameters.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"\nBest average performance across {len(all_datasets_results)} datasets:")
    print(f"  VCO: {best_threshold}")
    print(f"  Range Check: {best_range_check}")
    print(f"  Average F1 Score: {best_f1:.4f}")

    

# Function to load ground truth overlap points from CSV
def load_ground_truth(csv_path):
    """
    Load ground truth overlap points from CSV file.
    
    Parameters:
        csv_path: Path to CSV file with ground truth overlap points
    
    Returns:
        Dictionary mapping grain IDs to lists of overlap points
    """
    ground_truth = {}
    
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        grain_id = int(row['Grain_ID'])
        x = float(row['X'])
        y = float(row['Y'])
        
        if grain_id not in ground_truth:
            ground_truth[grain_id] = []
        
        ground_truth[grain_id].append((x, y))
    
    return ground_truth


def analyze_parameters(dem_path, outline_path, ground_truth_path, 
                       VCOs=None, range_checks=None, verbose=True):
    """
    Analyze different combinations of VCO and range_check parameters.
    
    Parameters:
        dem_path: Path to the DEM file
        outline_path: Path to the outline file
        ground_truth_path: Path to the ground truth CSV file
        VCOs: List of VCO values to test (default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        range_checks: List of range_check values to test (default: [1, 2, 3, 4, 5])
        verbose: Whether to print detailed results for each combination (default: True)
        
    Returns:
        DataFrame with results for each parameter combination
    """
    # Default parameter values if not specified
    if VCOs is None:
        VCOs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    if range_checks is None:
        range_checks = [1, 2, 3, 4, 5]
    
    # Load data
    print(f"Loading DEM from {dem_path}")
    dem_data = load_dem(dem_path)
    
    print(f"Loading outlines from {outline_path}")
    if outline_path.endswith('.tif'):
        outline_data = load_outline(outline_path)
    else:
        outline_data = read_npy_file(outline_path)
        
    print(f"Loading ground truth from {ground_truth_path}")
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Get grain outlines
    print("Extracting grain boundaries")
    grain_boundaries = find_boundaries(outline_data)
    grain_outlines = grain_boundaries
    
    # Initialize results storage
    results = []
    
    # Test all parameter combinations
    total_combinations = len(VCOs) * len(range_checks)
    print(f"Testing {total_combinations} parameter combinations")
    
    with tqdm(total=total_combinations, desc="Analyzing parameters") as pbar:
        for VCO in VCOs:
            for range_check in range_checks:
                # Analyze overlaps with current parameters
                overlaps = analyse_overlap(grain_boundaries, dem_data, outline_data, range_check=range_check, VCO=VCO)
                # Calculate confusion matrix metrics
                metrics = calculate_confusion_matrix(overlaps, ground_truth, grain_outlines)
                
                # Store results
                results.append({
                    'VCO': VCO,
                    'range_check': range_check,
                    'TP': metrics['TP'],
                    'FP': metrics['FP'],
                    'TN': metrics['TN'],
                    'FN': metrics['FN'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score']
                })
                
                # Print confusion matrix metrics for each combination if verbose
                if verbose:
                    print(f"\nParameters: VCO={VCO}, range_check={range_check}")
                    print(f"  Confusion Matrix: TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
                    print(f"  Metrics: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
                          f"Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
                
                pbar.update(1)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    return results_df


def visualize_results(results_df, metric='f1_score', save_path=None, show_plot=True):
    """
    Visualize the performance of different parameter combinations.
    
    Parameters:
        results_df: DataFrame with results from analyze_parameters
        metric: Which metric to visualize ('accuracy', 'precision', 'recall', 'f1_score')
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: True)
    """
    plt.figure(figsize=(12, 10))
    
    # Create a pivot table for the heatmap
    pivot_table = results_df.pivot_table(
        values=metric, 
        index='VCO', 
        columns='range_check'
    )
    
    # Create a heatmap
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        cmap='viridis', 
        fmt='.3f',
        linewidths=.5
    )
    
    # Set labels
    ax.set_title(f'{metric.capitalize()} by VCO and Range Check', fontsize=16)
    ax.set_ylabel('VCO', fontsize=14)
    ax.set_xlabel('Range Check', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def find_best_parameters(results_df, metric: str ='f1_score'):
    """
    Find the best combination of parameters.
    
    Parameters:
        results_df: DataFrame with results from analyze_parameters
        metric: Which metric to use for finding the best combination
        
    Returns:
        Dictionary with best parameters and their performance
    """
    # Find the row with the highest metric value
    best_idx = results_df[metric].idxmax()
    best_row = results_df.loc[best_idx]
    
    print(f"Best Parameters (optimizing for {metric}):")
    print(f"  VCO: {best_row['VCO']}")
    print(f"  Range Check: {best_row['range_check']}")
    print(f"  {metric}: {best_row[metric]:.4f}")
    print(f"  Accuracy: {best_row['accuracy']:.4f}")
    print(f"  Precision: {best_row['precision']:.4f}")
    print(f"  Recall: {best_row['recall']:.4f}")
    print(f"  F1 Score: {best_row['f1_score']:.4f}")
    print(f"  TP: {best_row['TP']}, FP: {best_row['FP']}, TN: {best_row['TN']}, FN: {best_row['FN']}")
    
    return {
        'VCO': best_row['VCO'],
        'range_check': best_row['range_check'],
        'performance': {
            'accuracy': best_row['accuracy'],
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'f1_score': best_row['f1_score'],
            'confusion_matrix': {
                'TP': best_row['TP'],
                'FP': best_row['FP'],
                'TN': best_row['TN'],
                'FN': best_row['FN']
            }
        }
    }


def calculate_confusion_matrix(predicted_overlaps, ground_truth_overlaps, grain_outlines):
    """Calculate confusion matrix metrics for overlap predictions."""

    # Define a VCO for point matching
    point_match_threshold = 0.3

    # Confusion matrix counters
    tp = 0  # Ground truth overlap matched by prediction
    fp = 0  # Prediction where there is no ground truth overlap
    fn = 0  # Ground truth overlap with no matching prediction
    tn = 0  # Contour points with no overlap in either prediction or ground truth

    # Set of all grain IDs to process
    all_grains = set(ground_truth_overlaps.keys()) | set(predicted_overlaps.keys())

    for grain_id in all_grains:
        gt_points = np.array(ground_truth_overlaps.get(grain_id, [])).reshape(-1, 2)
        pred_points = np.array(predicted_overlaps.get(grain_id, [])).reshape(-1, 2)
        contours = grain_outlines.get(grain_id, [])

        # --- True Positives and False Negatives ---
        for gt_point in gt_points:
            if pred_points.size > 0:
                distances = np.linalg.norm(pred_points - gt_point, axis=1)
                if np.any(distances < point_match_threshold):
                    tp += 1
                else:
                    fn += 1
            else:
                fn += 1

        # --- False Positives ---
        for pred_point in pred_points:
            if gt_points.size > 0:
                distances = np.linalg.norm(gt_points - pred_point, axis=1)
                if not np.any(distances < point_match_threshold):
                    fp += 1
            else:
                fp += 1

        # --- True Negatives ---
        for contour in contours:
            for point in contour:
                point_array = np.array(point).reshape(1, 2)

                in_gt = gt_points.size > 0 and np.any(np.linalg.norm(gt_points - point_array, axis=1) < point_match_threshold)
                in_pred = pred_points.size > 0 and np.any(np.linalg.norm(pred_points - point_array, axis=1) < point_match_threshold)

                if not in_gt and not in_pred:
                    tn += 1

    # --- Final Metrics ---
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }



def visualize_confusion_matrix(results_df, metric='TP', save_path=None, show_plot=True):
    """
    Visualize confusion matrix components across parameter combinations.
    
    Parameters:
        results_df: DataFrame with results from analyze_parameters
        metric: Which confusion matrix component to visualize ('TP', 'TN', 'FP', 'FN')
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: True)
    """
    plt.figure(figsize=(12, 10))
    
    # Create a pivot table for the heatmap
    pivot_table = results_df.pivot_table(
        values=metric, 
        index='VCO', 
        columns='range_check'
    )
    
    # Create a heatmap
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        cmap='Blues' if metric in ['TP', 'TN'] else 'Reds',  # Different color schemes for positive/negative
        fmt='.0f',  # Integer format
        linewidths=.5
    )
    
    # Set labels
    ax.set_title(f'{metric} by VCO and Range Check', fontsize=16)
    ax.set_ylabel('VCO', fontsize=14)
    ax.set_xlabel('Range Check', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix visualization saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_f1_heatmap(results_df, save_path=None, show_plot=True, highlight_best=True):
    """
    Create an enhanced heatmap visualization specifically for F1 scores.
    
    Parameters:
        results_df: DataFrame with results from analyze_parameters
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: True)
        highlight_best: Whether to highlight the best parameter combination (default: True)
    """
    plt.figure(figsize=(14, 12))
    
    # Create pivot table for F1 scores
    pivot_table = results_df.pivot_table(
        values='f1_score', 
        index='VCO', 
        columns='range_check'
    )
    
    # Find best parameters
    best_idx = results_df['f1_score'].idxmax()
    best_row = results_df.loc[best_idx]
    best_threshold = best_row['VCO']
    best_range_check = best_row['range_check']
    best_f1 = best_row['f1_score']
    
    # Create custom colormap with better contrast
    cmap = plt.cm.get_cmap('viridis')
    
    # Create the heatmap with enhanced formatting
    ax = sns.heatmap(
        pivot_table,
        annot=True, 
        cmap=cmap,
        fmt='.3f',  # Show 3 decimal places
        linewidths=0.5,
        cbar_kws={'label': 'F1 Score'}
    )
    
    # Highlight the best parameter combination if requested
    if highlight_best:
        # Find the position in the heatmap
        idx_y = list(pivot_table.index).index(best_threshold)
        idx_x = list(pivot_table.columns).index(best_range_check)
        # Add a red rectangle around the best value
        ax.add_patch(plt.Rectangle((idx_x, idx_y), 1, 1, fill=False, edgecolor='red', lw=3))
    
    # Enhance the title and labels
    ax.set_title('F1 Score by VCO and Range Check Parameters', fontsize=18, fontweight='bold')
    ax.set_ylabel('VCO', fontsize=14, fontweight='bold')
    ax.set_xlabel('Range Check', fontsize=14, fontweight='bold')
    
    # Add annotation for best parameters
    plt.figtext(0.5, 0.01, 
               f'Best F1 Score: {best_f1:.4f} (VCO={best_threshold}, Range Check={best_range_check})',
               ha='center', fontsize=14, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add annotations for TP, FP, FN at best parameters
    plt.figtext(0.5, 0.04, 
               f'At best parameters: TP={int(best_row["TP"])}, FP={int(best_row["FP"])}, FN={int(best_row["FN"])}',
               ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"F1 score heatmap saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return pivot_table  # Return the pivot table for further analysis if needed

if __name__ == "__main__":
    main()