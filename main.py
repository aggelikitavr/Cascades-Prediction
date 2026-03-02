from src.parser import parse_line
from experiments.train import train_at_k

def load_cascades(path, limit=None):
    """Loads and parses cascades from the specified file path."""
    print(f"Loading data from {path}...")
    cascades = []
    try:
        with open(path, "r") as f:
            for line in f:
                if not line.strip(): 
                    continue
                cascades.append(parse_line(line))
                if limit is not None and len(cascades) >= limit: 
                    break
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return []
    return cascades

def run_comparative_experiment(cascades, k_values, models):
    """Executes experiments for different models and observation thresholds (k)."""
    print(f"\n{'='*95}")
    print(f"{'Comparative Cascade Prediction Experiment (10-fold Cross Validation)':^95}")
    print(f"{'='*95}")
    print(f"{'Model':>12} | {'k':>4} | {'Avg AUC':>10} | {'Avg Acc':>10} | {'Pos %':>8} | {'Top Feature'}")
    print(f"{'-'*95}")

    all_results = {}

    for model_type in models:
        all_results[model_type] = {}
        for k in k_values:
            results = train_at_k(cascades, k=k, model_type=model_type)
            
            if results is None:
                continue
                
            # Identifying the most influential feature
            feat_imp = results['feature_importance']
            top_feat = max(feat_imp, key=lambda x: abs(feat_imp[x]))
            
            # Converting positive ratio to percentage
            pos_pct = results['positive_ratio'] * 100
            
            print(f"{model_type:>12} | {k:>4} | {results['auc']:>10.4f} | {results['accuracy']:>10.4f} | {pos_pct:>7.1f}% | {top_feat}")
            
            all_results[model_type][k] = results

    return all_results

def print_final_comparison(all_results):
    """Prints a summary of the top-performing model for each k value."""
    print(f"\nSummary of Best Models per k:")
    print(f"{'-'*40}")
    
    sample_model = list(all_results.keys())[0]
    k_values = all_results[sample_model].keys()

    for k in k_values:
        best_model = None
        best_auc = -1
        for model_type in all_results:
            current_auc = all_results[model_type][k]['auc']
            if current_auc > best_auc:
                best_auc = current_auc
                best_model = model_type
        print(f"k={k:2} : Best Model is {best_model:10} with AUC: {best_auc:.4f}")

if __name__ == "__main__":
    # Dataset loading configuration
    dataset_path = "data/weibo_dataset.txt"
    cascades = load_cascades(dataset_path, limit=13000)

    if cascades:
        print(f"Successfully loaded {len(cascades)} cascades.")

        # Experimental parameters
        k_test_values = [5, 10, 25, 50]
        model_types = ['logistic', 'rf', 'svm']

        # Executing comparative analysis
        results_data = run_comparative_experiment(cascades, k_test_values, model_types)

        # Final summary and paper verification
        print_final_comparison(results_data)