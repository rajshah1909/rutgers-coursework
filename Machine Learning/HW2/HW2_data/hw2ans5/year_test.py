import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# === Paths ===
data_root = Path(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\P5_data")
out_dir   = Path(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\hw2ans5")

# === Load train and test data ===
train = np.load(data_root / "vgg16_train.npz")
test  = np.load(data_root / "vgg16_test.npz", allow_pickle=True)   # <-- FIXED HERE

X_train, y_train = train["logit"], train["year"]
X_test_full, y_test, filenames = test["logit"], test["year"], test["filename"]

# === PCA (same parameters as training) ===
pca = PCA(n_components=2, whiten=True, random_state=42)
pca.fit(X_train)
X_test = pca.transform(X_test_full)

# === Load model + polynomial ===
model = joblib.load(out_dir / "year_model.pkl")
poly  = joblib.load(out_dir / "year_poly.pkl")

# === Predict ===
X_poly = poly.transform(X_test)
y_pred = model.predict(X_poly)
mse = mean_squared_error(y_test, y_pred)
print("✅ Test MSE:", round(mse, 4))

# === Find best / worst predictions ===
errors = np.abs(y_pred - y_test)
best_idx, worst_idx = np.argmin(errors), np.argmax(errors)

best_file  = filenames[best_idx]
worst_file = filenames[worst_idx]

with open(out_dir / "test_results.txt", "w") as f:
    f.write(f"Test MSE: {mse:.4f}\n")
    f.write(f"Most accurate prediction: {best_file} (True = {y_test[best_idx]}, Pred = {y_pred[best_idx]:.2f})\n")
    f.write(f"Least accurate prediction: {worst_file} (True = {y_test[worst_idx]}, Pred = {y_pred[worst_idx]:.2f})\n")

print("Results saved to:", out_dir / "test_results.txt")
