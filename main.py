import sys
import os

# Add the project root (current directory) to sys.path so we can import 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.experiments import run_experiments
    
    if __name__ == "__main__":
        print("Starting Thesis Project Experiments...")
        run_experiments()
        print("Experiments finished.")
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print("Ensure you are running 'python main.py' from the project root.")
