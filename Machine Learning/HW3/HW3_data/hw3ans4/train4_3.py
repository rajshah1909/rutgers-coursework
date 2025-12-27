# =============================================================
# File: train4_3.py
# Purpose: Train logistic regression with gradient descent (NumPy)
# =============================================================
import numpy as np
import os

base = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans4"

train = np.load(os.path.join(base, "train4_2.npz"))
X_train, y_train = train["x"], train["y"]

# Add bias column
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Initialize
w = np.zeros(X_train.shape[1])
lr = 1e-4
tol = 1e-4
max_iter = 15000

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def nll(X, y, w):
    p = sigmoid(X @ w)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.sum(y*np.log(p) + (1-y)*np.log(1-p))

prev = 1e12
for it in range(max_iter):
    p = sigmoid(X_train @ w)
    grad = X_train.T @ (p - y_train)

    w -= lr * grad
    cost = nll(X_train, y_train, w)

    if it % 500 == 0:
        print(f"Iter {it}, NLL = {cost:.4f}")

    if abs(prev - cost) < tol:
        print(f"Converged at iteration {it}")
        break

    prev = cost

np.savez(os.path.join(base, "weights4_3.npz"), w=w)
print("Saved weights to weights4_3.npz")
