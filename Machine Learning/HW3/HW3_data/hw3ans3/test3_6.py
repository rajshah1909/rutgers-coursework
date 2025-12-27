# =============================================================
# File: test3_6.py
# Purpose: Evaluate accuracy using the true data-generating model
# =============================================================
import numpy as np

data = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\P3_data\data_2\test.npz")
X_test = data["x"]
y_test = data["y"].flatten()

def mvn_pdf(X, mu, cov):
    n = X.shape[1]
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** n) * det)
    diff = X - mu
    return norm_const * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))

I = np.eye(2)
mu_pos = np.array([0, 0])
mu_n1 = np.array([0, 2])
mu_n2 = np.array([0, -2])

p_pos = mvn_pdf(X_test, mu_pos, I)
p_neg = 0.5 * mvn_pdf(X_test, mu_n1, I) + 0.5 * mvn_pdf(X_test, mu_n2, I)
preds = np.where(p_pos >= p_neg, 1, -1)

acc = np.mean(preds == y_test) * 100
print(f"Accuracy using true densities (3.6): {acc:.2f}%")
