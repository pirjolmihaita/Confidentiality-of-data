
import pandas as pd
import sys
import os

# Add parent directory to path to find src if needed (though this script doesn't import src, it's good practice in this repo)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def analyze_results():
    try:
    try:
        results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'results_wide.csv') # Using wide results as default
        if not os.path.exists(results_path):
             # Fallback or try results.csv
             results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'results.csv')
        
        df = pd.read_csv(results_path)
        print(f"Loaded {results_path} with shape:", df.shape)
        
        # Ensure we have the expected columns
        # dataset, task, model, mode, epsilon, data_norm, accuracy, precision, recall, f1_score, etc.
        
        # 1. Main Table: Compare Baseline vs DP (at different epsilons)
        # We want columns: Dataset | Model | Metric | Baseline | DP (eps=0.1) | DP (eps=0.5) | ...
        
        print("\n" + "="*50)
        print("SUMMARY: BASELINE VS DIFFERENTIAL PRIVACY")
        print("="*50)
        
        # Filter for relevant columns
        metrics = ['accuracy', 'f1_score']
        
        # Create a pivot table for Accuracy
        pivot_acc = df[df['mode'].isin(['baseline', 'dp'])].pivot_table(
            index=['dataset', 'model'], 
            columns='epsilon', 
            values='accuracy',
            aggfunc='first' # Should verify uniqueness, but 'first' is safe if no dupes
        )
        
        # Baseline epsilon is NaN. We want it as a column.
        # Pivot puts NaN epsilon in a specific way or excludes it. 
        # Let's handle baseline separately.
        
        baseline_df = df[df['mode'] == 'baseline'][['dataset', 'model', 'accuracy', 'f1_score', 'precision', 'recall']]
        baseline_df = baseline_df.rename(columns={c: f"{c}_baseline" for c in ['accuracy', 'f1_score', 'precision', 'recall']})
        
        dp_df = df[df['mode'] == 'dp']
        
        # Merge DP with Baseline
        merged = pd.merge(dp_df, baseline_df, on=['dataset', 'model'], how='left')
        
        # Calculate Drops
        merged['accuracy_drop'] = merged['accuracy_baseline'] - merged['accuracy']
        
        # Select columns for a nice view
        view_cols = ['dataset', 'model', 'epsilon', 'data_norm', 'accuracy_baseline', 'accuracy', 'accuracy_drop']
        print(merged[view_cols].sort_values(['dataset', 'model', 'epsilon']).to_string(index=False))
        
        # 2. HE Analysis
        print("\n" + "="*50)
        print("SUMMARY: HOMOMORPHIC ENCRYPTION OVERHEAD")
        print("="*50)
        
        he_df = df[df['mode'] == 'dp+he']
        if not he_df.empty:
            # We want to compare HE inference time with NORMAL inference time?
            # Or just show the HE time.
            he_cols = ['dataset', 'model', 'epsilon', 'n_samples', 'he_inference_time_sec', 'same_prediction_ratio', 'notes']
            print(he_df[he_cols].to_string(index=False))
        else:
            print("No DP+HE results found.")
            
        print("\nDone. You can copy these tables or use this script to generate CSVs.")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    analyze_results()
