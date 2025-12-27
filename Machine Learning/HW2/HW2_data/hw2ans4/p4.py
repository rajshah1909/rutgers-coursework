import argparse, os, sys, csv, math
from pathlib import Path
import numpy as np
from PIL import Image

# ---------- helpers ----------
def imread_gray(path: Path) -> np.ndarray:
    # Robust open: let PIL detect format (works even when files have no extension)
    with Image.open(path) as im:
        im = im.convert("L")  # grayscale
        return np.asarray(im, dtype=np.float64)

def imsave_gray(path: Path, arr: np.ndarray):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def minmax_to_255(vec: np.ndarray) -> np.ndarray:
    vmin, vmax = vec.min(), vec.max()
    if math.isclose(vmax, vmin):
        return np.zeros_like(vec)
    return ((vec - vmin) / (vmax - vmin) * 255.0)

def list_images(folder: Path):
    files = []
    for f in sorted(folder.iterdir()):
        if f.name.startswith('.') or f.name.lower().endswith('.ds_store'):
            continue
        if f.is_file():
            files.append(f)
    return files

# ---------- core ----------
def main():
    ap = argparse.ArgumentParser(description="CS461 P4 Eigenfaces")
    ap.add_argument("--data_root", required=True, help="Path to P4_data (has train/ and test/)")
    ap.add_argument("--out_dir", required=True, help="Where to save outputs")
    ap.add_argument("--test_name", default="subject15.normal",
                    help="File name inside test/ to approximate (default: subject15.normal)")
    ap.add_argument("--max_components", type=int, default=4000, help="Largest M to try (default 4000)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir  = data_root / "test"
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load training images into matrix X: (D pixels) x (N images) ----
    train_files = list_images(train_dir)
    if len(train_files) == 0:
        print(f"[ERROR] No training files found in {train_dir}")
        sys.exit(1)

    # Read first to define image size
    first_im = imread_gray(train_files[0])
    H, W = first_im.shape
    D = H * W

    X = np.zeros((D, len(train_files)), dtype=np.float64)
    for j, fp in enumerate(train_files):
        img = imread_gray(fp)
        if img.shape != (H, W):
            print(f"[WARN] Skipping {fp.name}, unexpected size {img.shape}, expected {(H,W)}")
            continue
        X[:, j] = img.flatten()

    # Remove any all-zero columns (in case some files were skipped)
    valid_cols = ~np.all(X == 0, axis=0)
    X = X[:, valid_cols]
    N = X.shape[1]
    if N == 0:
        print("[ERROR] No valid training images after size filtering.")
        sys.exit(1)

    # ---- 4.1 E[X], COV(X,X), spectral decomposition via SVD ----
    mean_face = np.mean(X, axis=1, keepdims=True)               # (D,1)
    Xc = X - mean_face                                          # center
    # Economical SVD: Xc = U * S * Vt, where
    # columns of U (D x r) are eigenvectors of COV in pixel space, S**2/(N-1)=eigenvalues
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)           # U:(D x r), S:(r,), Vt:(r x N)
    eigvals = (S**2) / (N - 1)                                  # λ_i
    E = U                                                       # eigenvectors in pixel space (columns)

    # Save mean face and a quick summary
    imsave_gray(out_dir / "mean_face.png", mean_face.reshape(H, W))
    with open(out_dir / "pca_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Image size: {H}x{W} (D={D}), #train images used: {N}\n")
        f.write("Top 20 eigenvalues:\n")
        top = np.minimum(20, eigvals.shape[0])
        for i in range(top):
            f.write(f"  λ[{i+1}] = {eigvals[i]:.6f}\n")
        f.write("\nE (eigenvectors) are columns of U from SVD(X_centered).\n")
        f.write("COV(X,X) = E Λ E^T (implicitly via SVD).\n")
    with open(out_dir / "eigenvalues_top200.csv", "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["index", "eigenvalue_lambda"])
        for i in range(min(200, eigvals.shape[0])):
            w.writerow([i+1, float(eigvals[i])])

    # ---- 4.3 visualize top-10 eigenvectors ("eigenfaces") ----
    top_k = min(10, E.shape[1])
    for i in range(top_k):
        face_vec = E[:, i]
        viz = minmax_to_255(face_vec).reshape(H, W)
        imsave_gray(out_dir / f"eigenface_{i+1:02d}.png", viz)

    # ---- 4.2 reconstruct chosen test image at M = {2,10,100,1000,4000} ----
    M_list = [2, 10, 100, 1000, 4000]
    # ensure M doesn't exceed available rank/components
    r_max = E.shape[1]
    M_list = [min(m, r_max, args.max_components) for m in M_list]

    test_path = test_dir / args.test_name
    if not test_path.exists():
        # Fallback: pick the first test file available
        test_files = list_images(test_dir)
        if not test_files:
            print(f"[ERROR] No test images found in {test_dir}")
            sys.exit(1)
        test_path = test_files[0]
        print(f"[WARN] Requested test '{args.test_name}' not found. Using '{test_path.name}'.")

    x_test = imread_gray(test_path)
    if x_test.shape != (H, W):
        print(f"[ERROR] Test image {test_path.name} has size {x_test.shape}, expected {(H,W)}")
        sys.exit(1)

    x_vec = x_test.flatten().astype(np.float64).reshape(-1, 1)
    x_centered = x_vec - mean_face

    # Project once using all components (efficient reuse)
    # Coeffs for principal components: a = E^T (x - mean)
    a_all = E.T @ x_centered                                   # (r,1)

    for M in M_list:
        if M <= 0: 
            continue
        EM = E[:, :M]                                          # (D x M)
        aM = a_all[:M, :]                                      # (M x 1)
        x_rec = mean_face + (EM @ aM)                          # (D x 1)
        imsave_gray(out_dir / f"reconstruction_M{M}.png", x_rec.reshape(H, W))

    # Also save the original test image for side-by-side comparison
    imsave_gray(out_dir / "test_original.png", x_test)

    # Done
    print("\n=== Outputs saved ===")
    print(out_dir.as_posix())
    print("Saved files:")
    print("  - mean_face.png")
    print("  - eigenface_01..10.png")
    print("  - reconstruction_M2/10/100/1000/4000.png (bounded by available components)")
    print("  - test_original.png")
    print("  - pca_summary.txt, eigenvalues_top200.csv")
    print("\nTip: Insert these images into your report for 4.2 and 4.3, and cite pca_summary.txt for 4.1.")
    
if __name__ == "__main__":
    main()
