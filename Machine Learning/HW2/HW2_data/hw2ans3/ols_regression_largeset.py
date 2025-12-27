# ------------------------------------------------------------
# File: ridge_regression.py
# Purpose: Ridge regression (MMSE with λ regularization)
# Author: Raj Shah - Rutgers University
# Course: CS 461 - Machine Learning Principles
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
save_dir = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\hw2ans3"
os.makedirs(save_dir, exist_ok=True)

data_path = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\P3_data\train_100.npz"

# --------------------------------------------------------
# Load Data
# --------------------------------------------------------
data = np.load(data_path)
x_train, y_train = data["x"], data["y"]

print(f"Loaded dataset: {data_path}")
print(f"Samples: {len(x_train)}")

# --------------------------------------------------------
# Design Matrix Φ(x)
# --------------------------------------------------------
def design_matrix(x, degree=9):
    """Create polynomial basis up to x^degree."""
    return np.vstack([x**i for i in range(degree + 1)]).T

Phi = design_matrix(x_train)

# --------------------------------------------------------
# Compute OLS Solution (no regularization)
# --------------------------------------------------------
w = np.linalg.pinv(Phi) @ y_train
print("OLS Weights (w):")
print(w)

# --------------------------------------------------------
# Predict and Plot
# --------------------------------------------------------
x_grid = np.linspace(0, 1, 200)
Phi_grid = design_matrix(x_grid)
y_pred = Phi_grid @ w

plt.figure(figsize=(7, 4))
plt.scatter(x_train, y_train, s=15, color="gray", alpha=0.6, label="Training Data")
plt.plot(x_grid, y_pred, color="green", linewidth=2, label="OLS Fit (train_100.npz)")
plt.title("OLS Regression on Larger Dataset")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
save_path = os.path.join(save_dir, "ols_large.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot saved as: {save_path}")