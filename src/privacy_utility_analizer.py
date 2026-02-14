import pandas as pd
import os
import re

# Configurații căi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, 'results', 'metrics', 'results_wide.csv')
BASE_TRADE_DIR = os.path.join(BASE_DIR, 'results', 'analysis', 'privacy_tradeoff')

def load_utility_data():
    if not os.path.exists(METRICS_PATH):
        print(f"Error: {METRICS_PATH} not found.")
        return None

    df = pd.read_csv(METRICS_PATH)
    id_vars = ['Dataset', 'Model', 'Task_Type']
    
    # Regex pentru a prinde Metrica, Epsilon și Norm (Utility)
    patterns = {
        'Baseline': re.compile(r"^Baseline_(F1|R2)$"),
        'DP': re.compile(r"^DP_(F1|R2)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-PHE': re.compile(r"^PHE_(F1|R2)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-FHE': re.compile(r"^Concrete_(F1|R2)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-FHE-W': re.compile(r"^ConcreteW_(F1|R2)_Eps([0-9.]+)_Norm([0-9.]+)$")
    }

    rows = []
    for _, row in df.iterrows():
        base_info = {col: row[col] for col in id_vars}
        for col in df.columns:
            if col in id_vars: continue
            
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.isna(val): continue

            for method_name, pattern in patterns.items():
                match = pattern.match(col)
                if match:
                    metric_name = match.group(1)
                    eps = match.group(2) if len(match.groups()) >= 2 else "None"
                    norm = match.group(3) if len(match.groups()) >= 3 else "None"
                    
                    rows.append({
                        **base_info, 
                        'Method': method_name, 
                        'Metric_Type': metric_name, 
                        'Score': val,
                        'Epsilon': eps,
                        'Data_Norm': norm
                    })
                    break
    return pd.DataFrame(rows)

def process_tradeoff(df):
    if df is None or df.empty: return

    # Pentru fiecare Dataset, Model și Metodă, găsim rândul cu cel mai bun scor
    # Clasificare -> F1 maxim, Regresie -> R2 maxim
    idx = df.groupby(['Dataset', 'Model', 'Method'])['Score'].idxmax()
    best_results = df.loc[idx]

    for (ds, task), ds_df in best_results.groupby(['Dataset', 'Task_Type']):
        # Creăm folderul: analysis/privacy_tradeoff/classification/adult/
        folder_path = os.path.join(BASE_TRADE_DIR, task, ds)
        os.makedirs(folder_path, exist_ok=True)

        # Sortăm după Score descrescător ca să vedem cea mai bună metodă sus
        ds_df = ds_df.sort_values(by='Score', ascending=False)

        # Redenumim coloana Score în funcție de task pentru claritate
        metric_label = 'Best_F1_Score' if task == 'classification' else 'Best_R2_Score'
        ds_df = ds_df.rename(columns={'Score': metric_label})

        output_file = os.path.join(folder_path, 'tabel_utility.csv')
        
        # Selectăm coloanele finale
        final_cols = ['Method', 'Model', metric_label, 'Epsilon', 'Data_Norm']
        ds_df[final_cols].to_csv(output_file, index=False)
        print(f"Saved: {task}/{ds}/tabel_utility.csv")

if __name__ == "__main__":
    print("Analiză Privacy-Utility Tradeoff în curs...")
    utility_df = load_utility_data()
    process_tradeoff(utility_df)
    print("\nFinalizat! Rezultatele sunt în results/analysis/privacy_tradeoff/")