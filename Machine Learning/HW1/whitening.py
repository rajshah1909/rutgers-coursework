# whitening.py
import numpy as np, os, sys

PREFERRED = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW1\x.npz"

def pca_whiten(Xc, k):
    # economy SVD on centered data (n x d)
    # NOTE: For very large d, randomized SVD would be better; here we cap k.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if k is not None:
        k = min(k, len(S))
        U, S, Vt = U[:, :k], S[:k], Vt[:k, :]
    # Whitening transform in PCA space: W = Xc * V * diag(1/sqrt(S^2/(n-1)))
    n = Xc.shape[0]
    eps = 1e-8
    inv_s = 1.0 / np.sqrt((S**2)/(max(n-1,1)) + eps)
    # Return projection matrix P = V^T * diag(inv_s), to right-multiply Xc
    P = Vt.T @ np.diag(inv_s)
    return P, Vt, S

def main(path, k=256, full_threshold=4096):
    data = np.load(path)
    x = data["x"] if "x" in data else data[list(data.keys())[0]]

    n, d = x.shape if x.ndim == 2 else (x.shape[0], 1)
    print(f"[whitening] X shape: {x.shape} (n={n}, d={d})")

    mu = x.mean(axis=0)
    Xc = x - mu

    if d <= full_threshold:
        # Full ZCA whitening via SVD of covariance
        print("[whitening] Using full ZCA whitening.")
        cov = (Xc.T @ Xc) / max(n-1,1)
        U, Svals, _ = np.linalg.svd(cov)
        eps = 1e-8
        A = (U @ np.diag(1.0 / np.sqrt(Svals + eps)) @ U.T)
        b = -A @ mu
        W = (x @ A.T) + b
        muW = W.mean(axis=0)
        covW = np.cov(W, rowvar=False)
        np.savez("x_white.npz", W=W, mu=mu, A=A, b=b, muW=muW, covW=covW, method="ZCA-full")
        print("[whitening] Saved x_white.npz (full ZCA).")
    else:
        # PCA whitening to top-k components
        print(f"[whitening] d={d} is large; using PCA whitening with k={k}.")
        P, Vt, S = pca_whiten(Xc, k=k)
        Wk = Xc @ P    # (n x k)
        muW = Wk.mean(axis=0)
        covW = np.cov(Wk, rowvar=False)
        explained = (S**2) / np.sum(S**2)
        np.savez("x_white.npz", Wk=Wk, mu=mu, P=P, muW=muW, covW=covW,
                 explained=explained, k=int(Wk.shape[1]), method="PCA-k")
        print("[whitening] Saved x_white.npz (PCA-k). First 10 explained ratios:", explained[:10])

if __name__ == "__main__":
    path = PREFERRED if os.path.exists(PREFERRED) else "x.npz"
    if not os.path.exists(path):
        print("[whitening] ERROR: x.npz not found.")
        sys.exit(1)
    # You can tweak k here if you want more/less components
    main(path, k=256, full_threshold=4096)
