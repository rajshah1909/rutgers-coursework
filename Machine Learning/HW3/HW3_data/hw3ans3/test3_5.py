# =============================================================
# File: test3_5.py
# Purpose: Test 2D GDA classifier using equal priors
# =============================================================
import numpy as np

params = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans3\model_3_4.npz")
mu_pos, mu_neg = params["mu_pos"], params["mu_neg"]
cov_pos, cov_neg = params["cov_pos"], params["cov_neg"]

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

p_pos = mvn_pdf(X_test, mu_pos, cov_pos)
p_neg = mvn_pdf(X_test, mu_neg, cov_neg)
preds = np.where(p_pos >= p_neg, 1, -1)

acc = np.mean(preds == y_test) * 100
print(f"2D GDA Test Accuracy: {acc:.2f}%")
