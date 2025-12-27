# =============================================================
# File: train3_1.py
# Purpose: Estimate Gaussian parameters (mean, variance) for each class
# =============================================================
import numpy as np

# Load training data
data = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\P3_data\data_1\train.npz")
x = data["x"].flatten()
y = data["y"].flatten()

# Separate classes
x_pos = x[y == 1]
x_neg = x[y == -1]

# Compute statistics
mu_pos = np.mean(x_pos)
mu_neg = np.mean(x_neg)
var_pos = np.var(x_pos, ddof=1)
var_neg = np.var(x_neg, ddof=1)

print("Class +: mean =", mu_pos, " variance =", var_pos)
print("Class -: mean =", mu_neg, " variance =", var_neg)

# Save results for reuse
np.savez(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans3\model_3_1.npz",
         mu_pos=mu_pos, mu_neg=mu_neg, var_pos=var_pos, var_neg=var_neg)
print("Saved parameters to model_3_1.npz")
