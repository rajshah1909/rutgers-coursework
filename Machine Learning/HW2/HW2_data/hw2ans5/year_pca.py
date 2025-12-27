import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# === Paths ===
data_root = Path(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\P5_data")
out_dir   = Path(r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW2\HW2_data\hw2ans5")
out_dir.mkdir(parents=True, exist_ok=True)

# === Load training data ===
train = np.load(data_root / "vgg16_train.npz")
X, y = train["logit"], train["year"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# === PCA (2 components) with whitening ===
pca = PCA(n_components=2, whiten=True, random_state=42)
X_pca = pca.fit_transform(X)
np.savez(out_dir / "train_pca_whitened.npz", x=X_pca, y=y)

# === Scatter plots ===
fig = plt.figure(figsize=(10, 4))

# 1-D plot (PC1 vs. Year)
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X_pca[:, 0], y, c='blue', s=10)
ax1.set_xlabel("1st PCA Component")
ax1.set_ylabel("Year of Made")
ax1.set_title("Year over M=1 PCA")

# 2-D plot (PC1, PC2 vs. Year)
ax2 = fig.add_subplot(1, 2, 2)
sc = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10)
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_title("Year over M=2 PCA")
plt.colorbar(sc, ax=ax2, label="Year")

plt.tight_layout()
plt.savefig(out_dir / "year_over_pca.png", dpi=300)
plt.close()

print("✅ PCA and plots saved to:", out_dir)
