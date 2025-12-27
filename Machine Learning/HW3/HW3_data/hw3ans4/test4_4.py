# =============================================================
# File: test4_4.py
# Purpose: Report train and test accuracy
# =============================================================

import numpy as np
import os

base = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans4"

# Load PCA reduced data
train = np.load(os.path.join(base, "train4_2.npz"))
test  = np.load(os.path.join(base, "test4_2.npz"))
X_train, y_train = train["x"], train["y"]
X_test,  y_test  = test["x"],  test["y"]

# Load weights
w = np.load(os.path.join(base, "weights4_3.npz"))["w"]

# Add bias
X_train = np.hstack([np.ones((X_train.shape[0],1)), X_train])
X_test  = np.hstack([np.ones((X_test.shape[0],1)),  X_test])

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z, -500, 500)))

train_pred = (sigmoid(X_train @ w) >= 0.5).astype(int)
test_pred  = (sigmoid(X_test  @ w) >= 0.5).astype(int)

train_acc = np.mean(train_pred == y_train) * 100
test_acc  = np.mean(test_pred  == y_test)  * 100

print(f"TRAIN ACCURACY: {train_acc:.2f}%")
print(f"TEST ACCURACY : {test_acc:.2f}%")
