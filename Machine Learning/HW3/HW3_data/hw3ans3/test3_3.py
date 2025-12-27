# =============================================================
# File: test3_3.py
# Purpose: Apply MAP rule (includes class priors)
# =============================================================
import numpy as np

params = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans3\model_3_1.npz")
mu_pos, mu_neg = params["mu_pos"], params["mu_neg"]
var_pos, var_neg = params["var_pos"], params["var_neg"]

data = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\P3_data\data_1\train.npz")
x_train, y_train = data["x"].flatten(), data["y"].flatten()
p_pos = np.mean(y_train == 1)
p_neg = np.mean(y_train == -1)

data_test = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\P3_data\data_1\test.npz")
x_test = data_test["x"].flatten()
y_test = data_test["y"].flatten()

def normal_pdf(x, mu, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * ((x - mu) ** 2) / var)

# MAP decision rule: pick class with higher posterior probability
p1 = normal_pdf(x_test, mu_pos, var_pos) * p_pos
p2 = normal_pdf(x_test, mu_neg, var_neg) * p_neg
preds = np.where(p1 >= p2, 1, -1)

acc = np.mean(preds == y_test) * 100
print(f"MAP-based Test Accuracy: {acc:.2f}%")
