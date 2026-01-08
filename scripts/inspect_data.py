import pandas as pd
import sys
import os

files = ['adult.csv', 'data.csv', 'kag_risk_factors_cervical_cancer.csv', 'compas-scores-two-years.csv', 'creditcard.csv', 'heart.csv']
with open('schema_info.txt', 'w') as log:
    for f in files:
        try:
            path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', f)
            df = pd.read_csv(path, nrows=2)
            log.write(f"FILE: {f}\n")
            log.write(f"COLUMNS: {list(df.columns)}\n")
            log.write("-" * 20 + "\n")
        except Exception as e:
            log.write(f"FILE: {f} ERROR: {e}\n")
