# ------------------------------------------------------------
# File: plot_models.py
# Purpose: Ordinary Least Squares (MMSE) regression
# Author: Raj Shah - Rutgers University
# Course: CS 461 - Machine Learning Principles
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
save_dir = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\hw2ans3"
train_path = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\P3_data\train.npz"

# Load training data
train = np.load(train_path)
x, y = train["x"], train["y"]

# ------------------------------------------------------------
# Load precomputed weights
# ------------------------------------------------------------
try:
    w_ols = np.load(os.path.join(save_dir, "ols_weights.npy"))       # OLS (λ=0)
    w_ridge = np.load(os.path.join(save_dir, "ridge_weights.npy"))   # Ridge (λ*)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing weight file: {e.filename}\n"
                            "Make sure ols_regression.py and ridge_regression.py have been run first.")

# ------------------------------------------------------------
# Create design matrix (10 polynomial basis functions)
# ------------------------------------------------------------
x_plot = np.linspace(0, 1, 200)
Phi_plot = np.vstack([x_plot**i for i in range(10)]).T

# Predictions
y_ols = Phi_plot @ w_ols
y_ridge = Phi_plot @ w_ridge

# ------------------------------------------------------------
# Plot Comparison
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="gray", s=35, label="Training data", alpha=0.8)
plt.plot(x_plot, y_ols, "r--", linewidth=2, label="MMSE (λ = 0)")
plt.plot(x_plot, y_ridge, "b-", linewidth=2, label="Ridge (λ*)")

plt.title("Comparison of MMSE and Ridge Regression Models", fontsize=12, weight="bold")
plt.xlabel("x", fontsize=11)
plt.ylabel("f(x)", fontsize=11)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------
output_path = os.path.join(save_dir, "model_comparison.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Figure saved successfully at: {output_path}")
plt.show()