import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from dgp_utils import (
    generate_covariates,   
    generate_instrument,  
    generate_errors,       
    generate_d,            
    assign_treatment       
)

# ------------------------------------------------------------------------------
# Algorithm 2: Calibrating instrument strength (pi) by targeting first-stage F-statistic
# This code finds the value of pi such that the average first-stage F-statistic from
# regressing treatment T on Z and X in simulated data matches a pre-specified target.
# Follows the procedure in Appendix D.1, Step 2 of the thesis.
# Used to define "weak" and "strong" IV regimes in all simulation designs.
# ------------------------------------------------------------------------------


def first_stage_F(pi, N, rho_uv, u_size, n_size, seed=42):
    """
    Simulate a single dataset and return the first-stage F-statistic for Z in regression of T ~ Z + X.
    Step-by-step:
      1. Generate covariates X (structure matches simulation design).
      2. Generate binary instrument Z.
      3. Draw correlated errors v, eps0, eps1 for endogeneity.
      4. Compute latent index for treatment assignment.
      5. Assign treatment T using thresholding.
      6. Run OLS regression T ~ Z + X; return F-stat for Z.
    """
    p = u_size + n_size
    beta = np.array([1.0 - j/u_size for j in range(u_size)] + [0.0]*(p-u_size)) ##### note here that k = u_size, this could maybe need to be changed)

    # 1.
    X = generate_covariates(N, u_size, n_size)
    # 2.
    Z = generate_instrument(N)
    # 3.
    v, eps0, eps1 = generate_errors(N, rho_uv)
    # 4.
    d = generate_d(X, Z, pi, v, beta)
    # 5.
    T = assign_treatment(d, treatment_share=0.5)
    # 6. regress T ~ Z + X
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
    df["Z"] = Z
    df["T"] = T
    y = df["T"]
    exog = add_constant(df[["Z"] + [f"X{i+1}" for i in range(X.shape[1])]])
    mdl = OLS(y, exog).fit()
    return float(mdl.f_test("Z = 0").fvalue)

def avg_F(pi, N, rho_uv, u_size, n_size, seed=None, nreps=500):
    """
    Monte Carlo average of first-stage F-statistics over nreps independent replications.
    Used to reduce noise in F-stat estimates.
    """
    return np.mean([
        first_stage_F(pi, N, rho_uv, u_size, n_size, seed=s)
        for s in range(nreps)
    ])

def find_pi_for_F(target_F, N, rho_uv, u_size, n_size,
                  pi_lower=0.01, pi_upper=10.0,
                  tol=0.001, max_iter=500, nreps=500):
    """
    Find value of pi so that the simulated average first-stage F-statistic equals target_F.
    Uses bisection search as in thesis Algorithm 2, step 3.
    Steps:
      - Evaluate F(pi_lower), F(pi_upper) to bracket the target
      - Iteratively bisect interval and compute avg_F at midpoint
      - Stop when within tolerance tol or after max_iter steps
    """
    low, high = pi_lower, pi_upper
    F_low  = avg_F(low,  N, rho_uv, u_size, n_size, nreps)
    F_high = avg_F(high, N, rho_uv, u_size, n_size, nreps)
    if not (F_low < target_F < F_high):
        raise ValueError(f"Target F not bracketed: F({low})={F_low:.2f}, F({high})={F_high:.2f}")

    for _ in range(max_iter):
        mid   = 0.5*(low + high)
        F_mid = avg_F(mid, N, rho_uv,n_size, u_size, nreps)
        if abs(F_mid - target_F) < tol:
            return mid
        if F_mid < target_F:
            low = mid
        else:
            high = mid
    return 0.5*(low + high)


if __name__ == "__main__":
    N      = 10000
    rho_uv = 0.5
    nreps  = 50
    u_size = 5
    n_size = 5
    # Find the π that on average gives F≈3 or F≈18 (i.e. +15 over weak)
    #pi_weak   = find_pi_for_F(3,  N, rho_uv ,n_size, u_size, nreps=nreps)
    pi_strong = find_pi_for_F(15, N, rho_uv, n_size, u_size, nreps=nreps)

    #print(f"π (F≈3):  {pi_weak:.4f}")
    print(f"π (F≈15): {pi_strong:.4f}")
