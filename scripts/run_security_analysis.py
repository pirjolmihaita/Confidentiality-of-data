import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.models import ModelManager
from src.attacks import perform_dp_security_analysis
from src.utils import get_logger

logger = get_logger("security_analysis_script")

def run_security_analysis():
    dl = DataLoader()
    mm = ModelManager()
    
    # Run Analysis
    df = perform_dp_security_analysis(dl, mm, ds_name='adult', model_type='dt')
    
    # Save Results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'security_results.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"Security Analysis saved to {output_file}")
    
if __name__ == "__main__":
    run_security_analysis()
