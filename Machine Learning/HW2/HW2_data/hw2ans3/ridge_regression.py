# ------------------------------------------------------------
# File: ridge_regression.py
# Purpose: Ridge regression (MMSE with λ regularization)
# Author: Raj Shah - Rutgers University
# Course: CS 461 - Machine Learning Principles
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

# === Output Directory ===
save_dir = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\hw2ans3"
os.makedirs(save_dir, exist_ok=True)

# === Load Data ===
data = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\P3_data\train.npz")
x_train, y_train = data["x"], data["y"]

# === Design Matrix ===
def design_matrix(x, degree=9):
    """Create Φ = [1, x, x², ..., x⁹]."""
    return np.vstack([x**i for i in range(degree + 1)]).T

Phi = design_matrix(x_train)

# === Lambda Range ===
lambdas = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_mse, weights = [], []

print("=== Ridge Regression with 5-Fold Cross-Validation ===")

for lam in lambdas:
    mse_folds = []
    for train_idx, val_idx in kf.split(Phi):
        Phi_train, Phi_val = Phi[train_idx], Phi[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        I = np.eye(Phi_train.shape[1])

        # Ridge Regression solution
        w = np.linalg.inv(Phi_train.T @ Phi_train + lam * I) @ Phi_train.T @ y_train_fold
        y_pred = Phi_val @ w
        mse_folds.append(np.mean((y_val_fold - y_pred) ** 2))

    avg_mse = np.mean(mse_folds)
    val_mse.append(avg_mse)
    weights.append(w)
    print(f"λ={lam:.1e} | Avg Val MSE={avg_mse:.6f}")

# === Find Best Lambda ===
val_mse = np.array(val_mse)
best_idx = np.argmin(val_mse)
lambda_star = lambdas[best_idx]
w_star = weights[best_idx]

# === Save Results ===
np.save(os.path.join(save_dir, "ridge_val_mse.npy"), val_mse)
np.save(os.path.join(save_dir, "ridge_lambda_star.npy"), lambda_star)
np.save(os.path.join(save_dir, "ridge_weights.npy"), w_star)

# === Plot Validation Curve ===
plt.figure(figsize=(8, 5))
plt.semilogx(lambdas, val_mse, marker="o", linewidth=1.5)
plt.xlabel("Lambda (λ)")
plt.ylabel("Average Validation MSE")
plt.title("Ridge Regression (5-Fold Cross-Validation)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ridge_val_plot.png"), dpi=300)
plt.close()

# === Print Summary ===
print("\n✅ Best λ* = %.3e" % lambda_star)
print("Validation MSE = %.6f" % val_mse[best_idx])
print("\nFiles saved to:")
print(save_dir)
print(" - ridge_val_mse.npy")
print(" - ridge_lambda_star.npy")
print(" - ridge_weights.npy")
print(" - ridge_val_plot.png")