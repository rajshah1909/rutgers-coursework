# =============================================================
# File: data4_2.py
# Purpose: Reduce TF-IDF 2000-D data to 50-D using PCA
# =============================================================
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os

base = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans4"

# Load TF-IDF matrix
df = pd.read_csv(os.path.join(base, "spam_ham.csv"), index_col=0)

# X = 2000 TF-IDF features, y = class
X = df.drop(columns=["cls"]).to_numpy()
y = df["cls"].to_numpy()

# Apply PCA → 50-D
pca = PCA(n_components=50, random_state=42)
X_reduced = pca.fit_transform(X)

# Save PCA model
with open(os.path.join(base, "pca_model.pkl"), "wb") as f:
    pickle.dump(pca, f)

# 3500 / 500 split (correct)
X_train = X_reduced[:3500]
y_train = y[:3500]
X_test  = X_reduced[3500:4000]
y_test  = y[3500:4000]

# Save as NPZ
np.savez(os.path.join(base, "train4_2.npz"), x=X_train, y=y_train)
np.savez(os.path.join(base, "test4_2.npz"),  x=X_test,  y=y_test)

print("Saved train4_2.npz, test4_2.npz, and pca_model.pkl")
