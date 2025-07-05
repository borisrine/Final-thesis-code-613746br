import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

# Utility functions for Monte Carlo simulations of heterogeneous treatment effects under endogeneity.
# Implements data-generating processes closely following Lechner & Mareckova (2025) and extensions
# described in Section 5 of the thesis.

# 1. GLOBAL PARAMETERS
# ------------------------------------------------------------------------------


# Generates covariate matrix X with structure matching Lechner & Mareckova (2025) and Section 5.2.1 of the thesis.
def generate_covariates(n, u_size, n_size):
    """
    Generate an (p x 10) covariate matrix X, where p = u+n:
      - First u_size columns ~ Uniform(-sqrt(12)/2, sqrt(12)/2)
      - Next n_size columns ~ Normal(0, 1)
    """
    X_uniform = np.random.uniform(-np.sqrt(12)/2, np.sqrt(12)/2, size=(n, u_size))
    X_normal  = np.random.normal(0, 1, size=(n, n_size))
    X = np.hstack([X_uniform, X_normal])
    return X

# Generates binary instrument Z ~ Bernoulli(0.5), as used for IV identification (see Section 5.2.3 of the thesis).
def generate_instrument(n):
    """
    Generate a binary instrument Z of length n:
      Z ~ Bernoulli(0.5)
    """
    Z = np.random.binomial(1, 0.5, size=n)
    return Z

# Computes the latent index for treatment assignment, incorporating X, Z, and unobserved v, reflecting endogeneity and instrument relevance as per thesis Section 5.2.4.
def generate_d(X, Z, pi, v, beta):
    """
    Latent index for treatment:
      d_i = (X_i' β) / normalizer + π·Z_i + v_i
    """
    normalizer = np.sqrt((beta**2).sum() / 1.25)
    d = X.dot(beta) / normalizer + pi * Z + v
    return d
# Assigns binary treatment based on upper quantile of latent index, implementing the thresholding rule described in Section 5.2.4.
def assign_treatment(d, treatment_share):
    q = np.quantile(d, treatment_share)  # 75th percentile (upper quantile)
    T = (d > q).astype(int)   # 1 if in upper quantile, 0 otherwise
    return T


# ---------------------------------------------------------------------------------------------
# EFFECT AND OUTCOME FUNCTIONS (from Lechner and Mareckova (2025) Appendix B) and my own thesis
# ---------------------------------------------------------------------------------------------

# Returns linear individual treatment effects as in Lechner & Mareckova (2025) and thesis Section 5.2.5, used for settings with linear heterogeneity.
def tau_linear(X, beta):
    """
    Linear IATE from L&M
    """
    normalizer = np.sqrt((beta**2).sum() / 1.25)
    zeta_iate = 0.42
    XB = X.dot(beta)
    return (zeta_iate * XB) / normalizer + 1.0


# Returns step-function IATE, introducing abrupt heterogeneity as described in thesis Section 5.2.5 (stepwise scenario).
def tau_step(X, beta):
    """
    Step‐function IATE from L&M:
      tau_i = f^{WA}(X[i,0]) * f^{WA}(X[i,1]) - 2.8
    
    Here X[:,0] is x_{1,i}, and X[:,1] is x_{2,i}.
    Returns a length‐N array of IATEs.
    """
    # Extract the first two covariates
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    # Compute f^{WA}(x1) and f^{WA}(x2)
    f1 = 1.0 + 1.0 / (1.0 + np.exp(-20.0 * (x1 - 1.0/3.0)))
    f2 = 1.0 + 1.0 / (1.0 + np.exp(-20.0 * (x2 - 1.0/3.0)))
    
    # Form the product and subtract 2.8
    return f1 * f2 - 2.8

# Returns step-function IATE, introducing abrupt heterogeneity as described in thesis Section 5.2.5 (stepwise scenario).
def generate_errors(n, rho_uv):
    """
    Draw (v, eps0, eps1) jointly ~ N(0, Σ)
    with Var=1 on the diagonal, Corr(v, eps0)=Corr(v, eps1)=rho_uv,
    and eps0 ⟂ eps1.
    Returns three length-n arrays.
    """
    # build a 3×3 covariance with zeros off-diagonal between eps0 & eps1
    cov = np.array([
        [1.0,      rho_uv,    rho_uv],
        [rho_uv,   1.0,       0.0    ],
        [rho_uv,   0.0,       1.0    ]
    ])
    mean = np.zeros(3)
    draws = np.random.multivariate_normal(mean, cov, size=n)
    v, eps0, eps1 = draws.T
    return v, eps0, eps1

# Generates potential outcomes Y(0), Y(1) given covariates, errors, and IATE; allows for different baseline outcome functions as in Section 5.2.5.
def generate_potential_outcomes(X, IATE, eps0, eps1, function_type, beta, delta=1):
    """
    Given:
      - X:            an (N x p) array of covariates
      - IATE:         a length-N array of τ(X_i)
      - eps0, eps1:   N(0,1) errors
      - function_type: one of "trivial", "sine", or other
      - beta:         coefficient vector for the baseline
      - delta:        strength of the nonlinear baseline

    Returns y0, y1 as in your draft.
    """
    # For both "sine" and any other type, compute the normalized index first
    normalizer = np.sqrt((beta**2).sum() / 1.25)
    tilde_y = X.dot(beta) / normalizer
    if function_type == "trivial":
        y0 = eps0
        y1 = IATE + eps1
        return y0, y1

    elif function_type == "sine":
        y0 = delta * np.sin(tilde_y) + eps0
        y1 = delta * np.sin(tilde_y) + IATE + eps1
        return y0, y1
    else:
        y0 = delta * np.sin(tilde_y) + eps0
        y1 = delta * np.sin(tilde_y) + IATE + eps1
        return y0, y1
         


#### PI finding functions #######


# Calculates first-stage F-statistic for T ~ Z + X in simulated data; used to calibrate instrument strength (Section 5.2.3, Appendix D).
def first_stage_F(pi, N, rho_uv, u_size, n_size, seed=2002):
    """
    Mirror the DGP in simulate_once:
      1) X ← generate_covariates(N,5,5)
      2) Z ← generate_instrument(N)
      3) (v, _, _) ← generate_errors(N, rho_uv)
      4) d ← generate_d(X, Z, pi, v)
      5) T ← assign_treatment(d, 0.5)
      6) F‐stat from regressing T on Z + X
    """
    np.random.seed(seed)
    p = u_size + n_size
    k = p/2
    beta = np.array([1.0 - j/k for j in range(k)] + [0.0]*(p-k)) ##### note here that k = u_size, this could maybe need to be changed)

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
# Returns the average first-stage F-statistic over multiple replications for robust instrument calibration.
def avg_F(pi, N, rho_uv, u_size, n_size, seed=None, nreps=50):
    return np.mean([
        first_stage_F(pi, N, rho_uv, u_size, n_size, seed=s)
        for s in range(nreps)
    ])


# Finds the value of the instrument coefficient (pi) such that the simulated first-stage F-statistic
# for T ~ Z + X matches a target value, using bisection search.
# This ensures that instrument strength in the simulation matches empirical or theoretical standards,
# as discussed in Section 5.2.3 and Appendix D of the thesis. This method is used for the Strong F but
# weak compliance setting, where we set F = 15

def find_pi_for_F(target_F, N, rho_uv, u_size, n_size,
                  pi_lower=0.01, pi_upper=10.0,
                  tol=0.001, max_iter=50, nreps=50):
    # Set initial lower and upper bounds for pi
    low, high = pi_lower, pi_upper
    
    # Evaluate average F-statistic at lower and upper bounds
    F_low  = avg_F(low,  N, rho_uv, u_size, n_size, seed=None, nreps=nreps)
    F_high = avg_F(high, N, rho_uv, u_size, n_size, seed=None, nreps=nreps)
    
    # Check that the target F-statistic is bracketed by F_low and F_high
    # If not, raise an error, as the target cannot be reached within the search interval
    if not (F_low < target_F < F_high):
        raise ValueError(f"Target F not bracketed: F({low})={F_low:.2f}, F({high})={F_high:.2f}")

    # Perform bisection search to find pi with desired F-statistic
    for _ in range(max_iter):
        # Calculate midpoint of current interval
        mid = 0.5 * (low + high)
        # Compute average F-statistic for current pi
        F_mid = avg_F(mid, N, rho_uv, u_size, n_size, nreps)
        # If F_mid is within tolerance of target, return current pi
        if abs(F_mid - target_F) < tol:
            return mid
        # Otherwise, update the search bounds depending on whether F_mid is too low or too high
        if F_mid < target_F:
            low = mid
        else:
            high = mid
    # If max_iter reached, return the midpoint as approximate solution
    return 0.5 * (low + high)



# ----------------------------------------------------------------------------
# 1. Methods for the actual simulation
# ----------------------------------------------------------------------------

# Simulates a single dataset draw, returning covariates, instrument, treatment, outcome, and true IATE as DataFrame (see Algorithm 1, Section 5.3.3).

def simulate_once(n, u_size, n_size, k, treatment_function, pi_val, rho_uv_val, seed=None):
    """
    Generate one dataset and return a DataFrame containing:
      - X1...X10: covariates
      - Z: instrument
      - T: treatment indicator
      - Y: observed outcome
      - tau_true: true individual treatment effect
    """
    p = u_size + n_size
    beta = np.array([1.0 - j/k for j in range(k)] + [0.0]*(p-k)) ### here still 

    if seed is not None:
        np.random.seed(seed)

    # Step 1: draw covariates
    X = generate_covariates(n, u_size, n_size)

    # Step 2: compute the true IATE function
    IATE = {'linear': tau_linear, 'step': tau_step}.get(treatment_function, lambda X, beta: np.zeros(X.shape[0]))(X, beta)

    # Step 3: draw instrument and errors
    Z = generate_instrument(n)
    v, eps0, eps1 = generate_errors(n, rho_uv_val)

    # Step 4: latent index and treatment assignment
    d = generate_d(X, Z, pi_val, v, beta)
    T = assign_treatment(d, 0.5)

    # Step 5: potential outcomes under sine baseline
    y0, y1 = generate_potential_outcomes(X, IATE, eps0, eps1, 'sine', beta, delta=1)
    Y = (1 - T) * y0 + T * y1

    # Assemble into DataFrame
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
    df['Z'] = Z
    df['T'] = T
    df['Y'] = Y
    df['tau_true'] = IATE
    return df

# Computes the population Local Average Treatment Effect (LATE) among compliers, following the procedure described in Section 5.3.1 and Appendix D.
def get_true_late(
    n_large: int,
    u_size: int,
    n_size: int,
    k: int,
    treatment_function: str,
    pi_val: float,
    rho_uv: float,
    seed: int = 0
) -> float:
    """
    Compute the complier‐ATE (LATE) for a binary instrument, following your 8-step recipe:
      1. draw X, (v,·,·)
      2. form mu = Xβ/normalizer
      3. construct d0 = mu + v,     d1 = mu + π + v
      4. draw Z at random, let c = median(mu + π·Z + v)
      5. define T0 = 1{d0>c}, T1 = 1{d1>c}
      6. IATE = tau_fn(X)
      7. compliers = {T0=0, T1=1}
      8. LATE = mean_{i∈compliers} IATE_i
    """
    # 1) rebuild β and normalizer
    beta = np.array([1.0 - j/k for j in range(k)] + [0.0] * (u_size + n_size - k))
    normalizer = np.sqrt((beta**2).sum() / 1.25)

    # 2) simulate X and errors
    np.random.seed(seed)
    
    X = generate_covariates(n_large, u_size, n_size)
    v, _, _ = generate_errors(n_large, rho_uv)

    # 3) latent indices
    mu = X.dot(beta) / normalizer
    d0 = mu + v
    d1 = mu + pi_val + v

    # 4) threshold c
    Z = generate_instrument(n_large)
    d_mix = mu + pi_val * Z + v
    c = np.quantile(d_mix, 0.5)

    # 5) potential treatments
    T0 = (d0 > c).astype(int)
    T1 = (d1 > c).astype(int)

    # 6) true IATE
    tau_fn = {'linear': tau_linear, 'step': tau_step}[treatment_function]
    IATE = tau_fn(X, beta)

    # 7–8) compliers & their avg effect
    mask = (T0 == 0) & (T1 == 1)
    return float(IATE[mask].mean())

