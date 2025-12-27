# =============================================================
# File: train3_4.py
# Purpose: Estimate mean vector and covariance matrix for 2D GDA
# =============================================================
import numpy as np

data = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\P3_data\data_2\train.npz")
X = data["x"]
y = data["y"].flatten()

X_pos = X[y == 1]
X_neg = X[y == -1]

mu_pos = np.mean(X_pos, axis=0)
mu_neg = np.mean(X_neg, axis=0)
cov_pos = np.cov(X_pos.T)
cov_neg = np.cov(X_neg.T)

print("Class + mean:", mu_pos)
print("Class - mean:", mu_neg)
print("Class + covariance:\n", cov_pos)
print("Class - covariance:\n", cov_neg)

np.savez(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans3\model_3_4.npz",
         mu_pos=mu_pos, mu_neg=mu_neg, cov_pos=cov_pos, cov_neg=cov_neg)
