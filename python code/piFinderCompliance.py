import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

from dgp_utils import (
    generate_covariates,
    generate_instrument,
    generate_errors,
    generate_d,
    assign_treatment
)


def first_stage_F(pi, N, rho_uv, u_size, n_size, seed=None):
    """
    Computes the global first-stage F-statistic for T ~ Z + X using simulated data.
    Not used in the calibration loop of Algorithm 3, but included for diagnostic output
    (e.g., comparing compliance-based calibration to standard instrument strength).
    """
    if seed is not None:
        np.random.seed(seed)
    X = generate_covariates(N, u_size, n_size)
    Z = generate_instrument(N)
    v, eps0, eps1 = generate_errors(N, rho_uv)
    beta = np.array([1.0 - j/u_size for j in range(u_size)] + [0.0]*(n_size))
    d = generate_d(X, Z, pi, v, beta)
    T = assign_treatment(d, treatment_share=0.5)

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
    df['Z'] = Z
    df['T'] = T
    y = df['T']
    exog = add_constant(df[['Z'] + [f'X{i+1}' for i in range(X.shape[1])]])
    model = OLS(y, exog).fit()
    return float(model.f_test("Z=0").fvalue)


def compliance_score(pi, N, rho_uv, u_size, n_size, seed=None):
    """
    Simulates a single dataset and computes the average compliance score:
        μ(π) = E[T | Z=1] - E[T | Z=0]
    This measures the local identifying variation of Z on T, and directly reflects
    the strength of the instrument in the CLATE-relevant subpopulation.
    """
    if seed is not None:
        np.random.seed(seed)
    X = generate_covariates(N, u_size, n_size)
    Z = generate_instrument(N)
    v, eps0, eps1 = generate_errors(N, rho_uv)
    beta = np.array([1.0 - j/u_size for j in range(u_size)] + [0.0]*(n_size))
    d = generate_d(X, Z, pi, v, beta)
    T = assign_treatment(d, treatment_share=0.5)

    # Estimate mean outcomes by instrument
    t_z1 = T[Z == 1].mean()
    t_z0 = T[Z == 0].mean()
    return t_z1 - t_z0


def avg_F(pi, N, rho_uv, u_size, n_size, seeds):
    """
    Computes the average F-statistic across multiple seeds for smoother estimation.
    This is used only for logging and benchmarking (see main()).
    """
    return np.mean([first_stage_F(pi, N, rho_uv, u_size, n_size, seed=s) for s in seeds])


def avg_compliance(pi, N, rho_uv, u_size, n_size, seeds):
    """
    Monte Carlo average of the compliance score across different random seeds.
    This function smooths out randomness in μ̂(π) and is used in the bisection search.
    """
    return np.mean([compliance_score(pi, N, rho_uv, u_size, n_size, seed=s) for s in seeds])


def find_pi_for_F(target_F, N, rho_uv, u_size, n_size,
                  pi_lower=1e-3, pi_upper=5.0,
                  tol=1e-3, max_iter=30, nreps=30):
    """
    Binary search to find π such that avg_F(π) ≈ target_F.
    This mirrors Algorithm 2, included here for comparison to Algorithm 3.
    """
    seeds = list(range(nreps))
    low, high = pi_lower, pi_upper
    F_low  = avg_F(low,  N, rho_uv, u_size, n_size, seeds)
    F_high = avg_F(high, N, rho_uv, u_size, n_size, seeds)
    if not (F_low < target_F < F_high):
        raise ValueError(f"Target F not bracketed: F({low})={F_low:.2f}, F({high})={F_high:.2f}")

    for _ in range(max_iter):
        mid   = 0.5*(low + high)
        F_mid = avg_F(mid, N, rho_uv, u_size, n_size, seeds)
        if abs(F_mid - target_F) < tol:
            return mid
        if F_mid < target_F:
            low = mid
        else:
            high = mid
    return 0.5*(low + high)


def find_pi_for_mu(target_mu, N, rho_uv, u_size, n_size,
                   pi_lower=1e-3, pi_upper=5.0,
                   tol=1e-3, max_iter=30, nreps=30):
    """
    Main function for Algorithm 3:
    Finds π such that the average compliance score μ̂(π) ≈ target_mu using binary search.

    Steps:
      1. Simulate avg_compliance at lower and upper bounds to ensure target is bracketed.
      2. Iteratively bisect the interval and compute μ̂(π_mid).
      3. Stop if |μ̂(π_mid) - target_mu| < tol or after max_iter iterations.

    Returns:
        (π_star, μ̂(π_star)) → the π achieving the target compliance score and the achieved μ̂.
    """
    seeds = list(range(nreps))
    low, high = pi_lower, pi_upper
    mu_low  = avg_compliance(low,  N, rho_uv, u_size, n_size, seeds)
    mu_high = avg_compliance(high, N, rho_uv, u_size, n_size, seeds)
    if not (mu_low < target_mu < mu_high):
        raise ValueError(f"Target mu not bracketed: mu({low})={mu_low:.3f}, mu({high})={mu_high:.3f}")

    for _ in range(max_iter):
        mid    = 0.5*(low + high)
        mu_mid = avg_compliance(mid, N, rho_uv, u_size, n_size, seeds)
        if abs(mu_mid - target_mu) < tol:
            return mid, mu_mid
        if mu_mid < target_mu:
            low = mid
        else:
            high = mid
    final_mid = 0.5*(low + high)
    return final_mid, avg_compliance(final_mid, N, rho_uv, u_size, n_size, seeds)


def main():
    """
    Entry point for executing Algorithm 3:
      - Finds π that achieves a target compliance score (e.g., μ_target = 0.5).
      - Then computes the corresponding global first-stage F-statistic at π_star.
    Output is used in thesis simulations to validate identification under local vs. global criteria.
    """
    # Simulation settings
    N       = 10000
    rho_uv  = 0.5
    u_size, n_size = 5, 5
    nreps   = 50

    # Desired average compliance
    target_mu = 0.5

    # 1) Find pi for target compliance
    pi_star, mu_achieved = find_pi_for_mu(target_mu, N, rho_uv, u_size, n_size,
                                          pi_lower=0.01, pi_upper=10.0,
                                          tol=1e-3, max_iter=50, nreps=nreps)
    print(f"Found π ≈ {pi_star:.4f} for target average compliance μ={target_mu}")
    print(f"Achieved average compliance μ̂ = {mu_achieved:.4f}")

    # 2) Compute global F at this π
    seeds = list(range(nreps))
    F_star = avg_F(pi_star, N, rho_uv, u_size, n_size, seeds)
    print(f"Corresponding global first-stage F ≈ {F_star:.2f}")

if __name__ == "__main__":
    main()
