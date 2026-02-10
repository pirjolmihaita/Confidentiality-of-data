import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_encryption_cost():
    # 1. Load Results
    file_path = 'results_wide.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # 2. Select Relevant Metrics
    # We compare Baseline vs PHE vs Concrete ML
    # We'll use a specific setting for comparison, e.g., Eps=1.0, Norm=10 (Standard)
    
    eps = 1.0
    norm = 10
    
    # Columns to look for
    base_col = 'Baseline_InfTime'
    phe_col = f'PHE_Time_Eps{eps}_Norm{norm}'
    conc_col = f'Concrete_Time_Eps{eps}_Norm{norm}'
    
    summary_data = []
    
    # 3. Iterate through datasets/models to build plot data
    plot_data = []
    
    for i, row in df.iterrows():
        dataset = row['Dataset']
        model = row['Model']
        
        # Baseline Time
        base_time = row[base_col]
        plot_data.append({'Dataset': dataset, 'Model': model, 'Method': 'Baseline', 'Time (s)': base_time})
        
        row_summary = {
            'Dataset': dataset,
            'Model': model,
            'Baseline_Time_s': base_time,
            'PHE_Time_s': None,
            'Concrete_Time_s': None,
            'PHE_Slowdown': None,
            'Concrete_Slowdown': None
        }
        
        # PHE Time (Only for LR usually)
        if pd.notna(row.get(phe_col)):
            phe_time = row[phe_col]
            plot_data.append({'Dataset': dataset, 'Model': model, 'Method': 'PHE (Paillier)', 'Time (s)': phe_time})
            row_summary['PHE_Time_s'] = phe_time
            if base_time > 0:
                row_summary['PHE_Slowdown'] = phe_time / base_time
        
        # Concrete Time
        if pd.notna(row.get(conc_col)):
            conc_time = row[conc_col]
            plot_data.append({'Dataset': dataset, 'Model': model, 'Method': 'Concrete (FHE)', 'Time (s)': conc_time})
            row_summary['Concrete_Time_s'] = conc_time
            if base_time > 0:
                row_summary['Concrete_Slowdown'] = conc_time / base_time
                
        summary_data.append(row_summary)

    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    summary_df = pd.DataFrame(summary_data)
    
    # 4. Generate Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # We use Log Scale because FHE is much slower
    g = sns.barplot(data=plot_df, x='Dataset', y='Time (s)', hue='Method', palette='viridis')
    g.set_yscale("log")
    
    plt.title(f'Inference Time Cost (Log Scale) \n Comparison at Epsilon={eps}', fontsize=14)
    plt.ylabel('Inference Time (Seconds) - Log Scale', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.legend(title='Privacy Method')
    plt.tight_layout()
    
    # Save Plot
    plot_file = 'encryption_cost_chart.png'
    plt.savefig(plot_file)
    print(f"Chart saved to {plot_file}")
    
    # 5. Save Summary Table
    table_file = 'encryption_cost_table.csv'
    # Filter columns to only those with data
    final_summary = summary_df.dropna(axis=1, how='all')
    final_summary.to_csv(table_file, index=False)
    print(f"Summary Table saved to {table_file}")
    
    # Print preview
    print("\n--- Cost Summary Preview ---")
    print(final_summary[['Dataset', 'Model', 'Baseline_Time_s', 'PHE_Slowdown', 'Concrete_Slowdown']].head().to_string())

if __name__ == "__main__":
    visualize_encryption_cost()
