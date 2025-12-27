# nb_cls.py
import pandas as pd, numpy as np, json, os, sys

PREFERRED = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW1\test.csv"

def gaussian(x, mu, var):
    return (1.0/np.sqrt(2*np.pi*var)) * np.exp(-((x-mu)**2)/(2*var))

def load_params():
    if os.path.exists("nb_params.json"):
        with open("nb_params.json","r") as f:
            return json.load(f)
    elif os.path.exists("nb_params.npz"):
        d = np.load("nb_params.npz", allow_pickle=True)
        return {k: (float(d[k]) if k.startswith(("mu_","var_","prior_")) else d[k].item() if d[k].shape==() else d[k]) for k in d.files}
    else:
        print("[nb_cls] ERROR: nb_params.json/npz not found. Run nb_train.py first.")
        sys.exit(1)

def main(path):
    p = load_params()
    df = pd.read_csv(path)

    # use the exact original column names detected during training
    gcol = p["col_glucose"]
    bcol = p["col_bp"]
    ycol = p["col_outcome"] if p.get("col_outcome") in df.columns else None

    preds = []
    for _, row in df.iterrows():
        g = row[gcol]
        b = row[bcol]
        p_pos = gaussian(g, p["mu_g_pos"], p["var_g_pos"]) * gaussian(b, p["mu_b_pos"], p["var_b_pos"]) * p["prior_pos"]
        p_neg = gaussian(g, p["mu_g_neg"], p["var_g_neg"]) * gaussian(b, p["mu_b_neg"], p["var_b_neg"]) * p["prior_neg"]
        preds.append(1 if p_pos > p_neg else 0)

    if ycol is not None:
        acc = float(np.mean(np.array(preds) == df[ycol].values))
        print("[nb_cls] Test Accuracy:", acc)
    else:
        print("[nb_cls] (No ground-truth column found in test.csv) Preds length:", len(preds))
    # Save predictions
    pd.Series(preds, name="pred").to_csv("nb_predictions.csv", index=False)
    print("[nb_cls] Saved nb_predictions.csv")

if __name__ == "__main__":
    path = PREFERRED if os.path.exists(PREFERRED) else "test.csv"
    if not os.path.exists(path):
        print("[nb_cls] ERROR: test.csv not found at preferred or local path.")
        sys.exit(1)
    main(path)
