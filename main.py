import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.runner import run_experiments

    if __name__ == "__main__":
        print("Starting Thesis Project Experiments...")
        run_experiments()
        print("Experiments finished.")
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print("Ensure you are running 'python main.py' from the project root.")
