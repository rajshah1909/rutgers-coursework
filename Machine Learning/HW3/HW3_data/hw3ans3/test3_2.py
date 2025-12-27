# =============================================================
# File: test3_2.py
# Purpose: Predict test labels using GDA decision rule (likelihood comparison)
# =============================================================
import numpy as np

# Load parameters and test data
params = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans3\model_3_1.npz")
mu_pos, mu_neg = params["mu_pos"], params["mu_neg"]
var_pos, var_neg = params["var_pos"], params["var_neg"]

data = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\P3_data\data_1\test.npz")
x_test = data["x"].flatten()
y_test = data["y"].flatten()

# Gaussian density function
def normal_pdf(x, mu, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * ((x - mu) ** 2) / var)

# Predict
preds = np.where(normal_pdf(x_test, mu_pos, var_pos) >=
                 normal_pdf(x_test, mu_neg, var_neg), 1, -1)

accuracy = np.mean(preds == y_test) * 100
print(f"Test Accuracy (3.2 decision rule): {accuracy:.2f}%")
