# ------------------------------------------------------------
# File: evaluate_models.py
# Purpose: Evaluate test MSE for MMSE and Ridge models
# Author: Raj Shah - Rutgers University
# Course: CS 461 - Machine Learning Principles
# ------------------------------------------------------------
import numpy as np
import os

save_dir = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\hw2ans3"

# === Load test data ===
test = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\P3_data\test.npz")
x_test, y_test = test['x'], test['y']

def design_matrix(x, degree=9):
    return np.vstack([x**i for i in range(degree + 1)]).T

Phi_test = design_matrix(x_test)

# === Load weights ===
w_ols   = np.load(os.path.join(save_dir, "ols_weights.npy"))
w_ridge = np.load(os.path.join(save_dir, "ridge_weights.npy"))

mse_ols = np.mean((y_test - Phi_test @ w_ols) ** 2)
mse_ridge = np.mean((y_test - Phi_test @ w_ridge) ** 2)

print("Test MSE (λ=0):", mse_ols)
print("Test MSE (λ*):", mse_ridge)
