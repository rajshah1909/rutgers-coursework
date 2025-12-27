# nb_train.py
import pandas as pd, numpy as np, json, os, sys

PREFERRED = r"C:\Users\RAJ RUTGERS\Desktop\Machine Learning\HW1\train.csv"

def canon_cols(cols):
    """normalize headers: lowercase, strip, remove spaces/underscores"""
    return [c.strip().lower().replace(" ", "").replace("_", "") for c in cols]

def find_col(canon_to_orig, wanted_list):
    for w in wanted_list:
        if w in canon_to_orig:
            return canon_to_orig[w]
    return None

def estimate_params(series):
    mu = float(series.mean())
    var = float(series.var(ddof=0))
    if not np.isfinite(var) or var <= 0:
        var = 1e-6
    return mu, var

def main(path):
    train = pd.read_csv(path)
    canon = canon_cols(train.columns)
    canon_to_orig = dict(zip(canon, train.columns))

    # Try common name variants
    outcome_name = find_col(canon_to_orig, ["outcome","diabetes","class","label","target","y"])
    glucose_name = find_col(canon_to_orig, ["glucose","plas","plasma","glucoselvl","glucoselevel"])
    bp_name      = find_col(canon_to_orig, ["bloodpressure","bp","bpressure","diastolic","bloodpress"])

    # If any not found, print columns and fail clearly
    missing = []
    if outcome_name is None: missing.append("Outcome column (try: Outcome/Diabetes/Class/Label/Target)")
    if glucose_name is None: missing.append("Glucose column (try: Glucose)")
    if bp_name is None:      missing.append("BloodPressure column (try: BloodPressure/BP)")
    if missing:
        print("\n[nb_train] ERROR: could not find required columns.")
        print("Available columns:", list(train.columns))
        print("Missing:", "; ".join(missing))
        sys.exit(1)

    # Ensure binary labels in {0,1}
    y = train[outcome_name]
    if set(pd.unique(y)) - {0,1}:
        # Try to coerce: assume positive class is the larger or string 'positive/yes/true'
        y2 = y.copy()
        if y2.dtype == object:
            y2 = y2.str.lower().map({"pos":1,"positive":1,"yes":1,"true":1,"neg":0,"negative":0,"no":0,"false":0})
        try:
            y2 = y2.astype(int)
        except Exception:
            print("\n[nb_train] ERROR: outcome labels must be 0/1. Unique values:", pd.unique(y))
            sys.exit(1)
        train[outcome_name] = y2

    pos = train[train[outcome_name]==1]
    neg = train[train[outcome_name]==0]

    mu_g_pos, var_g_pos = estimate_params(pos[glucose_name])
    mu_g_neg, var_g_neg = estimate_params(neg[glucose_name])
    mu_b_pos, var_b_pos = estimate_params(pos[bp_name])
    mu_b_neg, var_b_neg = estimate_params(neg[bp_name])

    params = {
        "prior_pos": float(len(pos)/len(train)),
        "prior_neg": float(len(neg)/len(train)),
        "mu_g_pos": mu_g_pos, "var_g_pos": var_g_pos,
        "mu_g_neg": mu_g_neg, "var_g_neg": var_g_neg,
        "mu_b_pos": mu_b_pos, "var_b_pos": var_b_pos,
        "mu_b_neg": mu_b_neg, "var_b_neg": var_b_neg,
        # keep original column names so nb_cls can read test.csv reliably
        "col_glucose": glucose_name,
        "col_bp": bp_name,
        "col_outcome": outcome_name
    }

    with open("nb_params.json","w") as f:
        json.dump(params, f, indent=2)
    np.savez("nb_params.npz", **params)
    print("\n[nb_train] Saved nb_params.json and nb_params.npz")
    print(json.dumps(params, indent=2))

if __name__ == "__main__":
    path = PREFERRED if os.path.exists(PREFERRED) else "train.csv"
    if not os.path.exists(path):
        print("[nb_train] ERROR: train.csv not found at preferred or local path.")
        sys.exit(1)
    main(path)
