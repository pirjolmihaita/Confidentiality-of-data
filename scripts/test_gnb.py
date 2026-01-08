
from diffprivlib.models import GaussianNB
import numpy as np

# Mock data
X = np.random.rand(10, 5)
y = np.random.randint(0, 2, 10)
bounds_list = [(-1, 1)] * 5
bounds_tuple = (-1, 1)

print("Testing list bounds...")
try:
    clf = GaussianNB(epsilon=1, bounds=bounds_list)
    clf.fit(X, y)
    print("List bounds working.")
except Exception as e:
    print(f"List bounds failed: {e}")

print("\nTesting tuple bounds...")
try:
    clf = GaussianNB(epsilon=1, bounds=bounds_tuple)
    clf.fit(X, y)
    print("Tuple bounds working.")
except Exception as e:
    print(f"Tuple bounds failed: {e}")
