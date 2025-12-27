# ================================================================
# File: train2_2.py
# Purpose: Implement SMO for Linear SVM on Iris-setosa vs Iris-versicolor
# Author: Raj Shah
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ================================================================
# Load Data
# ================================================================
data = np.load(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\P2_data\data.npz")
X = data['x']
y = data['y']

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ================================================================
# Define linear kernel
# ================================================================
def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

# ================================================================
# SVM Class (Simplified SMO)
# ================================================================
class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_passes=5):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X, y):
        m, n = X.shape
        alphas = np.zeros(m)
        b = 0
        passes = 0

        while passes < self.max_passes:
            num_changed = 0
            for i in range(m):
                xi, yi = X[i], y[i]
                Ei = (np.sum((alphas * y) * (X @ xi)) + b) - yi

                if (yi * Ei < -self.tol and alphas[i] < self.C) or (yi * Ei > self.tol and alphas[i] > 0):
                    j = np.random.randint(0, m - 1)
                    if j == i: j = (j + 1) % m
                    xj, yj = X[j], y[j]
                    Ej = (np.sum((alphas * y) * (X @ xj)) + b) - yj

                    ai_old, aj_old = alphas[i], alphas[j]

                    # Compute L, H
                    if yi != yj:
                        L = max(0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0, aj_old + ai_old - self.C)
                        H = min(self.C, aj_old + ai_old)
                    if L == H:
                        continue

                    eta = 2 * np.dot(xi, xj) - np.dot(xi, xi) - np.dot(xj, xj)
                    if eta >= 0:
                        continue

                    alphas[j] = aj_old - (yj * (Ei - Ej)) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - aj_old) < 1e-5:
                        continue

                    alphas[i] = ai_old + yi * yj * (aj_old - alphas[j])

                    # Compute b1 and b2
                    b1 = b - Ei - yi * (alphas[i] - ai_old) * np.dot(xi, xi) - yj * (alphas[j] - aj_old) * np.dot(xi, xj)
                    b2 = b - Ej - yi * (alphas[i] - ai_old) * np.dot(xi, xj) - yj * (alphas[j] - aj_old) * np.dot(xj, xj)

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        self.b = b
        self.alphas = alphas
        self.w = np.sum((alphas * y)[:, None] * X, axis=0)

    def project(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.project(X))

# ================================================================
# Run SVM for C = 1, 10, 100
# ================================================================
C_values = [1, 10, 100]
output_path = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans2\svm_output.txt"
f = open(output_path, "w")

for C in C_values:
    model = SVM(C=C)
    model.fit(X, y)
    preds = model.predict(X)
    support_vectors = np.sum(model.alphas > 1e-5)
    margin = 2 / np.linalg.norm(model.w)

    acc = np.mean(preds == y) * 100

    print(f"C={C} | Support Vectors={support_vectors} | Margin={margin:.4f} | Accuracy={acc:.2f}%")
    f.write(f"C={C} | Support Vectors={support_vectors} | Margin={margin:.4f} | Accuracy={acc:.2f}%\n")

    # Plot
    plt.figure()
    plt.scatter(X[y==1,0], X[y==1,1], color='red', label='Iris-setosa (+1)')
    plt.scatter(X[y==-1,0], X[y==-1,1], color='blue', label='Iris-versicolor (-1)')

    x_plot = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    y_plot = -(model.w[0] * x_plot + model.b) / model.w[1]
    y_margin_up = -(model.w[0] * x_plot + model.b - 1) / model.w[1]
    y_margin_down = -(model.w[0] * x_plot + model.b + 1) / model.w[1]

    plt.plot(x_plot, y_plot, 'k-', label='Decision Boundary')
    plt.plot(x_plot, y_margin_up, 'r--', label='+1 Margin')
    plt.plot(x_plot, y_margin_down, 'b--', label='-1 Margin')
    plt.legend()
    plt.title(f"Soft-Margin SVM (C={C})")
    plt.savefig(fr"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans2\svm_plot_C{C}.png")
    plt.close()

f.close()
print("All results saved to hw3ans2 folder.")
