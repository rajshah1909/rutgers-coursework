# =============================================================
# File: test4_5.py
# Purpose: Correct classification of mail.txt using SAME TF-IDF + SAME PCA
# =============================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

base = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW3\HW3_data\hw3ans4"

# Load training TF-IDF data
df = pd.read_csv(os.path.join(base, "spam_ham.csv"), index_col=0)
X = df.drop(columns=["cls"])
vocab = list(X.columns)

# Rebuild EXACT vectorizer from vocabulary
vectorizer = TfidfVectorizer(vocabulary=vocab)

# Load email
with open(os.path.join(base, "mail.txt"), "r", encoding="utf8") as f:
    msg = f.read()

# Convert to 2000-D vector
X_mail = vectorizer.fit_transform([msg]).toarray()

# Load PCA model
with open(os.path.join(base, "pca_model.pkl"), "rb") as f:
    pca = pickle.load(f)

# Apply same PCA
X_mail_pca = pca.transform(X_mail)

# Add bias
X_mail_pca = np.hstack([np.ones((1,1)), X_mail_pca])

# Load trained LR weights
w = np.load(os.path.join(base, "weights4_3.npz"))["w"]

# Predict
p = 1/(1+np.exp(-np.clip(X_mail_pca @ w, -500, 500)))
label = "SPAM" if p >= 0.5 else "HAM"

print(f"Prediction: {label} (Probability = {p[0]:.4f})")
