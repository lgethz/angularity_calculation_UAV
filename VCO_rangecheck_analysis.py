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
    Single Dataset Parameter Optimization for Overlap Detection
    
    This script systematically evaluates how different combinations of VCO (Vertical Cutoff 
    Optimization) and range_check parameters affect overlap detection performance on a single 
    dataset. It compares algorithm predictions against manually marked ground truth data to 
    find optimal parameters based on various performance metrics.
    
    Workflow:
    1. Loads a single DEM, outline/segmentation file, and ground truth overlap points
    2. Tests a grid of parameter combinations (multiple VCO and range_check values)
    3. For each combination:
       - Runs overlap detection algorithm with those parameters
       - Compares detected points with ground truth using spatial proximity matching
       - Calculates confusion matrix components (TP, FP, TN, FN)
       - Derives performance metrics (accuracy, precision, recall, F1 score)
    4. Generates comprehensive visualizations:
       - Heatmaps for each performance metric
       - Enhanced F1 score heatmap with best parameters highlighted
       - Individual heatmaps for confusion matrix components
    5. Outputs multiple result files:
       - CSV with all parameter combinations and their performance metrics
       - Text file with analysis setup and parameters tested
       - JSON file with best parameters by different optimization targets
       - PNG visualizations for all metrics and confusion matrix components
    
    Required inputs:
    - dem_path: Path to DEM file (.tif format)
    - outline_path: Path to clast outline/segmentation file (.tif or .npy)
    - ground_truth_path: Path to CSV file with manually marked ground truth overlap points
      CSV format should have columns: clastID, X, Y
    
    Note: For multi-dataset analysis that finds globally optimal parameters across multiple
    samples, use VCO_rangecheck_analysis_dataset.py instead.
    """
    dem_path = "Users/folder/DEM/"
    outline_path = "Users/folder/Outline/"
    ground_truth_path = "Users/folder/ground_truth.csv"
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"parameter_analysis_results_{timestamp}"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    VCOS = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008]
    range_checks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    results = analyze_parameters(
        dem_path=dem_path,
        outline_path=outline_path,
        ground_truth_path=ground_truth_path,
        VCOS=VCOS,
        range_checks=range_checks,
        verbose=True 
    )
    
    csv_path = os.path.join(results_dir, 'parameter_analysis_results.csv')
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    with open(os.path.join(results_dir, 'analysis_info.txt'), 'w') as f:
        f.write(f"Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"DEM file: {dem_path}\n")
        f.write(f"Outline file: {outline_path}\n")
        f.write(f"Ground truth file: {ground_truth_path}\n\n")
        f.write(f"VCO values: {VCOS}\n")
        f.write(f"Range check values: {range_checks}\n")
    
    best_params = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        print("\n" + "="*50)
        best_param = find_best_parameters(results, metric=metric)
        best_params[metric] = best_param
        
        visualize_results(
            results,
            metric=metric,
            save_path=os.path.join(results_dir, f'visualization_{metric}.png')
        )

    visualize_f1_heatmap(
        results,
        save_path=os.path.join(results_dir, 'f1_score_heatmap.png'),
        highlight_best=True
    )

    for cm_metric in ['TP', 'TN', 'FP', 'FN']:
        visualize_confusion_matrix(
        results,
        metric=cm_metric,
        save_path=os.path.join(results_dir, f'confusion_matrix_{cm_metric}.png')
    )
    
    import json
    with open(os.path.join(results_dir, 'best_parameters.json'), 'w') as f:
        json.dump(best_params, f, indent=4)

    


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
                       VCOS=None, range_checks=None, verbose=True):
    """
    Analyze different combinations of VCO and range_check parameters.
    
    Parameters:
        dem_path: Path to the DEM file
        outline_path: Path to the outline file
        ground_truth_path: Path to the ground truth CSV file
        VCOS: List of VCO values to test (default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        range_checks: List of range_check values to test (default: [1, 2, 3, 4, 5])
        verbose: Whether to print detailed results for each combination (default: True)
        
    Returns:
        DataFrame with results for each parameter combination
    """
    if VCOS is None:
        VCOS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    if range_checks is None:
        range_checks = [1, 2, 3, 4, 5]
    
    print(f"Loading DEM from {dem_path}")
    dem_data = load_dem(dem_path)
    
    print(f"Loading outlines from {outline_path}")
    if outline_path.endswith('.tif'):
        outline_data = load_outline(outline_path)
    else:
        outline_data = read_npy_file(outline_path)
        
    print(f"Loading ground truth from {ground_truth_path}")
    ground_truth = load_ground_truth(ground_truth_path)
    
    print("Extracting grain boundaries")
    grain_boundaries = find_boundaries(outline_data)
    grain_outlines = grain_boundaries
    
    results = []
    
    total_combinations = len(VCOS) * len(range_checks)
    print(f"Testing {total_combinations} parameter combinations")
    
    with tqdm(total=total_combinations, desc="Analyzing parameters") as pbar:
        for VCO in VCOS:
            for range_check in range_checks:
                overlaps = analyse_overlap(grain_boundaries, dem_data, outline_data, range_check=range_check, VCO=VCO)
                metrics = calculate_confusion_matrix(overlaps, ground_truth, grain_outlines)
                
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
                
                if verbose:
                    print(f"\nParameters: VCO={VCO}, range_check={range_check}")
                    print(f"  Confusion Matrix: TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
                    print(f"  Metrics: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
                          f"Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
                
                pbar.update(1)
    
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
    
    pivot_table = results_df.pivot_table(
        values=metric, 
        index='VCO', 
        columns='range_check'
    )
    
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        cmap='viridis', 
        fmt='.3f',
        linewidths=.5
    )
    
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
    import numpy as np

    point_match_VCO = 0.3

    tp = 0  # Ground truth overlap matched by prediction
    fp = 0  # Prediction where there is no ground truth overlap
    fn = 0  # Ground truth overlap with no matching prediction
    tn = 0  # Contour points with no overlap in either prediction or ground truth

    all_grains = set(ground_truth_overlaps.keys()) | set(predicted_overlaps.keys())

    for grain_id in all_grains:
        gt_points = np.array(ground_truth_overlaps.get(grain_id, [])).reshape(-1, 2)
        pred_points = np.array(predicted_overlaps.get(grain_id, [])).reshape(-1, 2)
        contours = grain_outlines.get(grain_id, [])

        # --- True Positives and False Negatives ---
        for gt_point in gt_points:
            if pred_points.size > 0:
                distances = np.linalg.norm(pred_points - gt_point, axis=1)
                if np.any(distances < point_match_VCO):
                    tp += 1
                else:
                    fn += 1
            else:
                fn += 1

        # --- False Positives ---
        for pred_point in pred_points:
            if gt_points.size > 0:
                distances = np.linalg.norm(gt_points - pred_point, axis=1)
                if not np.any(distances < point_match_VCO):
                    fp += 1
            else:
                fp += 1

        # --- True Negatives ---
        for contour in contours:
            for point in contour:
                point_array = np.array(point).reshape(1, 2)

                in_gt = gt_points.size > 0 and np.any(np.linalg.norm(gt_points - point_array, axis=1) < point_match_VCO)
                in_pred = pred_points.size > 0 and np.any(np.linalg.norm(pred_points - point_array, axis=1) < point_match_VCO)

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
    
    pivot_table = results_df.pivot_table(
        values=metric, 
        index='VCO', 
        columns='range_check'
    )
    
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        cmap='Blues' if metric in ['TP', 'TN'] else 'Reds',  
        fmt='.0f',  
        linewidths=.5
    )
    
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
    
    pivot_table = results_df.pivot_table(
        values='f1_score', 
        index='VCO', 
        columns='range_check'
    )
    
    best_idx = results_df['f1_score'].idxmax()
    best_row = results_df.loc[best_idx]
    best_VCO = best_row['VCO']
    best_range_check = best_row['range_check']
    best_f1 = best_row['f1_score']
    
    cmap = plt.cm.get_cmap('viridis')
    
    ax = sns.heatmap(
        pivot_table,
        annot=True, 
        cmap=cmap,
        fmt='.3f',  # Show 3 decimal places
        linewidths=0.5,
        cbar_kws={'label': 'F1 Score'}
    )
    
    if highlight_best:
        idx_y = list(pivot_table.index).index(best_VCO)
        idx_x = list(pivot_table.columns).index(best_range_check)
        ax.add_patch(plt.Rectangle((idx_x, idx_y), 1, 1, fill=False, edgecolor='red', lw=3))
    
    ax.set_title('F1 Score by VCO and Range Check Parameters', fontsize=18, fontweight='bold')
    ax.set_ylabel('VCO', fontsize=14, fontweight='bold')
    ax.set_xlabel('Range Check', fontsize=14, fontweight='bold')
    
    plt.figtext(0.5, 0.01, 
               f'Best F1 Score: {best_f1:.4f} (VCO={best_VCO}, Range Check={best_range_check})',
               ha='center', fontsize=14, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
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
    
    return pivot_table  

if __name__ == "__main__":
    main()