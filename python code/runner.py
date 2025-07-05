#!/usr/bin/env python3
# Main script for running the full Monte Carlo simulation grid for DRIV estimator.
# Runs all configurations (sample sizes, treatment functions, instrument strengths), as defined in Section 5 of the thesis.
# Results are saved with filenames encoding the design for easy retrieval and replication.

import os
import time
import numpy as np
import pandas as pd
from multiprocessing import freeze_support

from monteCarloSimulationDRIV import MonteCarloSimulator

# --------------------------------------------------------------------
# 1. Simulation Grid Settings
# --------------------------------------------------------------------
# Specify the treatment effect functions to compare (stepwise and linear heterogeneity).
treatment_functions = ["step", "linear"]

# Map: n → { weak‐compliance π, strong‐compliance π }
# Each (n, compliance) pair has a calibrated value of pi to achieve weak or strong compliance. 
# These are computed using the optimalPiFinderF file and optimalPiFinderCompliance file. 
# See Appendix D and Section 5.2.3 for how these values are computed.
pi_map = {
    1000:  {'weak': 0.3539,   'strong': 2.0587},
    10000: {'weak': 0.1134, 'strong': 2.0490},
}
ns = [1000, 10000]

# --------------------------------------------------------------------
# 2. Simulation Baseline Arguments (parameters fixed across grid)
# --------------------------------------------------------------------
base_args = dict(
    seed0      = 0,    # Random seed for reproducibility
    u_size     = 5,    # Number of uniform covariates
    n_size     = 5,    # Number of normal covariates
    k          = 6,    # Number of relevant covariates (affecting treatment/outcome)
    rho_uv_val = 0.5,  # Endogeneity parameter (correlation between error terms)
)

default_reps = 1000  # Number of Monte Carlo repetitions for n=1000 (Section 5.3.3)

# --------------------------------------------------------------------
# 3. Output Folder
# --------------------------------------------------------------------
# Define absolute path for results; ensures that all outputs are organized and reproducible. Change this if you want to run the code on your own device.
out_dir = (
    "/Users/borisrine/Library/CloudStorage/"
    "OneDrive-ErasmusUniversityRotterdam/Documents/Uni/"
    "bsc2/Year 4/Thesis/Extension Code/results"
)
os.makedirs(out_dir, exist_ok=True)


def prepend_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute column means for all numeric results and insert as first row.
    Used to facilitate quick inspection of aggregated simulation metrics.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[num_cols].mean()
    summary['rep'] = np.nan
    summary = summary.reindex(df.columns)
    return pd.concat([pd.DataFrame([summary]), df], ignore_index=True)

# --------------------------------------------------------------------
# 5. Main Loop: Full Monte Carlo Grid
# --------------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()

    # Loop over all treatment functions, sample sizes, and pi values
    for func in treatment_functions:
        for n in ns:
            # pull both weak & strong compliance settings
            for compliance in ["weak", "strong"]:
                pi = pi_map[n][compliance]

                # Use fewer repetitions for n=10000 to manage compute cost (Section 5.3.3)
                reps = 250 if n == 10000 else default_reps

                # Output file is named with function, compliance, pi, and n for clarity.
                fname = f"dmliv_results_{func}_{compliance}_pi_{pi:.4f}_n{n}.csv"
                outpath = os.path.join(out_dir, fname)
                # Skip this config if results already exist (for resuming long runs).
                if os.path.exists(outpath):
                    print(f"[{time.strftime('%H:%M:%S')}] Skipping {func}, "
                          f"{compliance}, pi={pi}, n={n} (exists)")
                    continue

                print(f"[{time.strftime('%H:%M:%S')}] Running {func}, "
                      f"{compliance}, pi={pi}, n={n}, reps={reps}")
                # Build simulator arguments for this configuration
                sim_kwargs = dict(
                    **base_args,
                    reps=reps,
                    treatment_function=func,
                    pi_val=pi,
                    n=n,
                )
                sim = MonteCarloSimulator(**sim_kwargs)

                start = time.time()
                df = sim.run()
                elapsed = time.time() - start
                # Insert summary row with average results at top for quick overview
                df_out = prepend_summary(df)
                df_out.to_csv(outpath, index=False)
                print(f"    → Done in {elapsed:.1f}s; saved to {outpath}")
