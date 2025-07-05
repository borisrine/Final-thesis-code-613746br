import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from econml.iv.dml import DMLIV
from econml.inference import BootstrapInference
import multiprocessing as mp


# Your DGP utilities
from dgp_utils import (
    simulate_once,
    get_true_late)

# Monte Carlo simulation class for benchmarking DRIV (Double Robust IV) estimator in Python.
# Simulates heterogeneous treatment effect estimation under endogeneity for one configuration.
# GRF (Generalized Random Forest) simulation is handled separately (in R).
# This class manages the generation of DGP samples, model estimation, and performance evaluation,
# as described in Section 5 and Appendix D of the thesis.


class MonteCarloSimulator:
    def __init__(self, reps, seed0=0, **sim_args):
        """
        Initialize the simulator with:
          reps:   Number of Monte Carlo repetitions for this configuration.
          seed0:  Base random seed for reproducibility.
          sim_args: Other simulation settings (sample size, instrument strength, etc.).
        """
        self.reps = reps
        self.seed0 = seed0
        self.sim_args = sim_args

    def _one_replication(self, args):
        """
        Run a single Monte Carlo replication:
          - Simulate train/test data from the DGP (simulate_once)
          - Fit DRIV estimator using econml
          - Compute LATE and CLATE metrics (bias, MSE, MAE, coverage)
        Returns dictionary of results for one run.
        This corresponds to large parts of algorithm 1 in the thesis. 
        """
        n, u_size, n_size, k, treatment_function, pi_val, rho_uv_val, rep, seed = args

        # True LATE and group‐LATE (for benchmarking), step 3 in algorithm 1 of thesis. 
        true_late = get_true_late(
            n_large=1_000_000, u_size=u_size, n_size=n_size,
            k=k, treatment_function=treatment_function,
            pi_val=pi_val, rho_uv=rho_uv_val, seed=self.seed0)
        
       # --- 1. Simulate  train dataset(Algorithm 1, steps 6–7)
        df_train = simulate_once(n, u_size, n_size, k,
                                 treatment_function, pi_val,
                                 rho_uv_val, seed + rep*3)
        df_test  = simulate_once(n, u_size, n_size, k,
                                 treatment_function, pi_val,
                                 rho_uv_val, seed + rep*3 + 1)
        X_train = df_train.filter(regex='^X').values
        T_train = df_train['T'].values
        Z_train = df_train['Z'].values
        Y_train = df_train['Y'].values
       # Simulate  train dataset(Algorithm 1, steps 10-11)
        X_test  = df_test.filter(regex='^X').values
        T_test  = df_test['T'].values
        Z_test  = df_test['Z'].values
        Y_test  = df_test['Y'].values
        tau_true = df_test['tau_true']

        

        # ----------------------------------------------------------------------------
        # 3. Estimate DRIV using econml's DMLIV, step 8 of algorithm 1. 
        # ----------------------------------------------------------------------------
        DRIV = DMLIV(
            model_y_xw=RandomForestRegressor(n_estimators=500, max_depth=6, n_jobs=1),
            model_t_xw=RandomForestClassifier(n_estimators=500, max_depth=6,n_jobs=1),
            model_t_xwz=RandomForestClassifier(n_estimators=500, max_depth=6,n_jobs=1),
            discrete_treatment=True,
            discrete_instrument=True
        )
        DRIV.fit(Y=Y_train, T=T_train, Z=Z_train,
                  X=X_train, inference=BootstrapInference())
        tau_DRIV_iate = DRIV.effect(X_test)
        late_DRIV      = DRIV.ate(X_test)
        lb_DRIV, ub_DRIV = DRIV.ate_interval(X_test)

        
        df_test['tau_DRIV'] = tau_DRIV_iate


        # ----------------------------------------------------------------------------
        # Compute metrics (steps 14 and 15 in algorithm 1)
        # ----------------------------------------------------------------------------
        # Step 14: LATE performance: bias, MSE, MAE, coverage (confidence interval includes ground truth)

        def metrics(est, lb, ub):
            bias   = est - true_late
            return {
                'late':           est,
                'bias_late':      bias,
                'mse_late':       bias**2,
                'mae_late':       abs(bias),
                'coverage_late':  int(lb <= true_late <= ub),
                'lb_late':        lb,
                'ub_late':        ub
            }

        m_DRIV = metrics(late_DRIV, lb_DRIV, ub_DRIV)

        # Individual level CLATE metrics (average across test set)
        def iate_metrics(tau_hat):
            mse  = mean_squared_error(tau_true, tau_hat)
            bias = np.mean(tau_hat - tau_true)
            mae  = np.mean(np.abs(tau_hat - tau_true))
            return {'mse_iate': mse, 'bias_iate': bias, 'mae_iate': mae}

        im_DRIV = iate_metrics(tau_DRIV_iate)

        # Return all relevant results for aggregation over replications
        return {
            'rep': rep,
            'true_late': true_late,
            # DRIV
            **{f'DRIV_{k}': v for k, v in m_DRIV.items()},
            **{f'DRIV_{k}': v for k, v in im_DRIV.items()},
        }

    def run(self):
        """
        Run the full Monte Carlo simulation (all repetitions) for this configuration.
        Uses multiprocessing for speed: each replication is run in parallel.
        Returns a pandas DataFrame of results (one row per replication).
        """

        # Prepare arguments for each replication
        arg_list = [
            (
                self.sim_args['n'],
                self.sim_args['u_size'],
                self.sim_args['n_size'],
                self.sim_args['k'],
                self.sim_args['treatment_function'],
                self.sim_args['pi_val'],
                self.sim_args['rho_uv_val'],
                r,
                self.seed0 + r
            )
            for r in range(self.reps)
        ]
        # Use all available CPU cores for parallel execution
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(self._one_replication, arg_list)
        return pd.DataFrame(results)

### example usage, not relevant for actually running the code for my thesis. That is done using runner.py 
def main():
    pi_strong = 2
    rho_uv    = 0.5

    sim = MonteCarloSimulator(
        reps=1,        # e.g. 100 Monte Carlo draws
        seed0=1234,
        n=1000,
        u_size=5,
        n_size=5,
        k=6,
        treatment_function='linear',
        pi_val=pi_strong,
        rho_uv_val=rho_uv
    )

    start = time.time()
    df = sim.run()
    # compute and prepend the summary row
    num_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[num_cols].mean()
    summary['rep'] = np.nan
    summary = summary.reindex(df.columns)
    df = pd.concat([pd.DataFrame([summary]), df], ignore_index=True)
    print(df.head())
    df.to_csv('/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Thesis/Extension Code/results/sim_results_DRIV_test.csv', index=False)
    print(f"Total time: {time.time() - start:.1f}s")

if __name__ == '__main__':
    main()
