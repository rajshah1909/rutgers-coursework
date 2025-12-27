# ------------------------------------------------------------
# File: ols_regression.py
# Purpose: Ordinary Least Squares (MMSE) regression
# Author: Raj Shah - Rutgers University
# Course: CS 461 - Machine Learning Principles
# ------------------------------------------------------------
import numpy as np
from sklearn.model_selection import KFold
import os

# === Save directory ===
save_dir = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\hw2ans3"
os.makedirs(save_dir, exist_ok=True)

# === Load training data ===
data = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\P3_data\train.npz")
x_train, y_train = data['x'], data['y']

# === Design matrix with polynomial basis up to degree 9 ===
def design_matrix(x, degree=9):
    return np.vstack([x**i for i in range(degree + 1)]).T

Phi = design_matrix(x_train)

# === 5-fold cross-validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_errors = []

for train_idx, val_idx in kf.split(Phi):
    Phi_tr, Phi_val = Phi[train_idx], Phi[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # OLS closed-form: w = (ΦᵀΦ)⁻¹Φᵀy
    w = np.linalg.pinv(Phi_tr) @ y_tr
    y_pred = Phi_val @ w
    mse = np.mean((y_val - y_pred) ** 2)
    val_errors.append(mse)

val_error_avg = np.mean(val_errors)
np.save(os.path.join(save_dir, "ols_val_error.npy"), val_error_avg)
print("Average Validation MSE:", val_error_avg)
# ------------------------------------------------------------
# Save OLS weights for plotting later
# ------------------------------------------------------------
Phi_full = design_matrix(x_train)
w_ols = np.linalg.pinv(Phi_full) @ y_train
np.save(os.path.join(save_dir, "ols_weights.npy"), w_ols)
print("✅ OLS weights saved successfully at:", os.path.join(save_dir, "ols_weights.npy"))
