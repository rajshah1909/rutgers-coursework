import numpy as np
import os
import sys

def summarize_array(name, arr):
    """Prints summary statistics for each numpy array."""
    print(f"  → {name}: shape={arr.shape}, dtype={arr.dtype}")
    if np.issubdtype(arr.dtype, np.number):
        print(f"     min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}")
    print()

# === MAIN ===
# Allow folder argument or default to current directory
base_folder = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
print(f"\n🔍 Scanning recursively under: {base_folder}\n")

npz_files = []
for root, _, files in os.walk(base_folder):
    for f in files:
        if f.endswith('.npz'):
            npz_files.append(os.path.join(root, f))

if not npz_files:
    print("⚠️  No .npz files found anywhere under this folder.")
else:
    print(f"Found {len(npz_files)} .npz files in total.\n")
    for path in npz_files:
        print(f"📦 {path}")
        try:
            data = np.load(path)
            print(" Contains arrays:", data.files)
            for name in data.files:
                summarize_array(name, data[name])
        except Exception as e:
            print(f" ⚠️  Error reading {os.path.basename(path)}: {e}")
        print("="*70)
