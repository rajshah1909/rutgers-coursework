import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# === Paths ===
data_root = Path(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\hw2ans5")
out_dir   = data_root
out_dir.mkdir(exist_ok=True)

# === Load PCA-whitened training data ===
data = np.load(data_root / "train_pca_whitened.npz")
X, y = data["x"], data["y"]

# === Train/validation split (80/20) ===
n = len(X)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
split = int(0.8 * n)
X_train, X_val = X[idx[:split]], X[idx[split:]]
y_train, y_val = y[idx[:split]], y[idx[split:]]

# === Try polynomial degrees 1–6 ===
best_mse, best_deg, best_model = np.inf, None, None
for deg in range(1, 7):
    poly = PolynomialFeatures(degree=deg, include_bias=True)
    Xtr = poly.fit_transform(X_train)
    Xva = poly.transform(X_val)
    model = LinearRegression().fit(Xtr, y_train)
    val_mse = mean_squared_error(y_val, model.predict(Xva))
    print(f"Degree {deg}: Validation MSE = {val_mse:.3f}")
    if val_mse < best_mse:
        best_mse, best_deg, best_model, best_poly = val_mse, deg, model, poly

# === Save model, polynomial, and summary ===
import os
joblib.dump(best_model, out_dir / "year_model.pkl")
joblib.dump(best_poly,  out_dir / "year_poly.pkl")

with open(out_dir / "train_summary.txt", "w") as f:
    f.write(f"Best polynomial degree: {best_deg}\n")
    f.write(f"Validation MSE: {best_mse:.4f}\n")

print(f"✅ Training complete — Best degree = {best_deg}, Validation MSE = {best_mse:.4f}")
