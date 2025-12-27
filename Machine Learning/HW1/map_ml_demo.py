import numpy as np
def ml_est(x): return np.mean(x)
def map_est(x, mu0, sigma2_0, sigma2):
    N = len(x); mu_ml = ml_est(x)
    w0 = sigma2/(N*sigma2_0+sigma2)
    w_ml = (N*sigma2_0)/(N*sigma2_0+sigma2)
    return w0*mu0 + w_ml*mu_ml

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = rng.normal(5,2,20)
    print("ML:", ml_est(data))
    print("MAP:", map_est(data,0,4,4))
