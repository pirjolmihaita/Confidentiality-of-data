import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO)

def test_data():
    dl = DataLoader()
    
    for name in ['adult', 'heart', 'insurance', 'communities']:
        try:
            print(f"--- Testing {name} ---")
            X_train, X_test, y_train, y_test, preprocessor, task_type = dl.load_and_preprocess(name)
            print(f"Success! Task Type: {task_type}")
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            
            # Test preprocessing fit
            X_train_proc = preprocessor.fit_transform(X_train)
            print(f"Processed shape: {X_train_proc.shape}")
            
        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_data()
