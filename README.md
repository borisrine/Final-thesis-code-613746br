
# Thesis Simulation Codebase (DRIV vs GRF under Endogeneity)

This repository contains the full codebase used for the simulations in the thesis:

> **"Machine Learning for Heterogeneous Treatment Effects under Endogeneity: A Comparison of Double Machine Learning and Generalized Random Forests"**

The simulation framework implements a controlled Monte Carlo environment to compare the performance of:
- **DRIV (Double Robust Instrumental Variables)** — estimated using the `econml` package in Python.
- **GRF (Generalized Random Forests)** — estimated using the `grf` package in R.

## Contents

```
├── Python code/
│   ├── dgp_utils.py               # Data-generating process (Python)
│   ├── monteCarloSimulationDRIV.py # Monte Carlo logic for DRIV
│   ├── runner.py                  # DRIV grid runner
│   ├── piFinderF.py               # Calibrates π to target F-statistic
│   ├── piFinderCompliance.py      # Calibrates π to target compliance
│
├── R code/
│   ├── dgp_utils.R                # Identical to `dgp_utils.py`, ported to R
│   └── runner.R                   # GRF grid runner — includes simulation logic
│
```

## How to Use

### 1. **(Optional)** Calibrate π-values

To reproduce the calibrated instrument strength (`π`) values used in both DRIV and GRF simulations:

```bash
python3 piFinderF.py           # For F-statistic calibration (e.g., F ≈ 15)
python3 piFinderCompliance.py  # For compliance score calibration (e.g., μ ≈ 0.5)
```

>  **Note:** These values are already hard-coded into `runner.py` and `runner.R`. Running this step is *not required* unless you want to replicate the π-finding procedure from scratch or if you want to run the simulation with different sample size, covariate size or correlation structure.

---

### 2. **Run DRIV Simulations (Python)**

To benchmark the DMLIV estimator using `econml`, simply run:

```bash
python3 runner.py
```

This executes a grid of simulations across:
- Treatment function types (`step`, `linear`)
- Sample sizes (`n = 1000`, `n = 10000`)
- Compliance regimes (`weak`, `strong`)

Output files are saved under `results/` and include summary rows for easy analysis.

---

### 3. **Run GRF Simulations (R)**

To benchmark the Generalized Random Forest estimator:

```r
source("runner.R")
```

This performs the same grid as the Python code, but using `instrumental_forest()` from the `grf` R package.

- The R version includes both the runner and the simulation loop (unlike Python where they're separate).
- The DGP functions are identical between `dgp_utils.py` and `dgp_utils.R`.

---

## Output

- Each simulation creates a CSV file like:

```
grf_R_results_step_weak_pi_0.3539_n1000.csv
dmliv_results_linear_strong_pi_2.0490_n10000.csv
```

- These contain both replication-level results and a prepended row with summary statistics (mean, median bias).

---

## Requirements

### Python:
Install packages via pip:

```bash
pip install numpy pandas scikit-learn statsmodels econml
```

### R:
Install required R packages:

```r
install.packages(c("grf", "parallel"))
```

---

## Notes

- All code assumes a structure consistent with the thesis (see Section 5 and Appendix D).
- Modify the `out_dir` paths inside `runner.py` and `runner.R` to point to your local output folder.
- Parallelization is implemented in both R (`mclapply`) and Python (`multiprocessing.Pool`).

---

## Contact

For questions or reproduction assistance, contact:

**Boris Rine**  
Erasmus School of Economics  
613746br@eur.nl
boris.rine033@gmail.com
