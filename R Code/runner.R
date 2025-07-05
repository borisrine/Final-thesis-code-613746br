# Monte Carlo simulation for Generalized Random Forests (GRF) under endogeneity.
# Mirrors the design of `runner.py` and `monteCarloSimulationDRIV.py` used for DRIV in Python.
# For detailed explanations of the DGP, estimator logic, and evaluation metrics, refer to those files.
# This script performs the GRF benchmarking in R, using the same grid of configurations, 
# with parallelization for computational efficiency.


library(grf)
library(parallel)
library(ggplot2)

# Source your data‑generating utilities
dgp_utils_path <- "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Thesis/Extension Code/R tests/dgp_utils.R"
source(dgp_utils_path)

# Main function: runs one configuration of the Monte Carlo simulation
run_monte_carlo_fast <- function(reps,
                                 seed0 = 0,
                                 sim_args = list(n, u_size, n_size, k,
                                                 treatment_function,
                                                 pi_val, rho_uv_val)) {
  # number of cores to use
  n_cores <- detectCores() - 1
  
  # Precompute population truths once
  true_late <- get_true_late(
    n_large            = 1e6,
    u_size             = sim_args$u_size,
    n_size             = sim_args$n_size,
    k                  = sim_args$k,
    treatment_function = sim_args$treatment_function,
    pi_val             = sim_args$pi_val,
    rho_uv            = sim_args$rho_uv_val,
    seed               = seed0
  )
  
  
  
  # Define one Monte Carlo replication (data draw → fit GRF → evaluate metrics)
  one_rep <- function(rep) {
    # seeds
    seed_train <- seed0 + rep * 2
    seed_test  <- seed_train + 1
    
    # i) Generate training & test data
    df_train <- simulate_once(n                  = sim_args$n,
                              u_size             = sim_args$u_size,
                              n_size             = sim_args$n_size,
                              k                  = sim_args$k,
                              treatment_function = sim_args$treatment_function,
                              pi_val             = sim_args$pi_val,
                              rho_uv_val         = sim_args$rho_uv_val,
                              seed               = seed_train)
    df_test  <- simulate_once(n                  = sim_args$n,
                              u_size             = sim_args$u_size,
                              n_size             = sim_args$n_size,
                              k                  = sim_args$k,
                              treatment_function = sim_args$treatment_function,
                              pi_val             = sim_args$pi_val,
                              rho_uv_val         = sim_args$rho_uv_val,
                              seed               = seed_test)
    
    # Extract relevant variables from training and test sets
    X_train <- as.matrix(df_train[, grep("^X", names(df_train))])
    Tr_train <- df_train$Tr
    Z_train  <- df_train$Z
    Y_train  <- df_train$Y
    X_test   <- as.matrix(df_test[, grep("^X", names(df_test))])
    Z_test   <- df_test$Z
    Y_test   <- df_test$Y
    tau_true <- df_test$tau_true
    
    # Fit IV-GRF (GRF for heterogeneous treatment effects under endogeneity)
    iv_forest <- instrumental_forest(
      X         = X_train,
      Y         = Y_train,
      W         = Tr_train,
      Z         = Z_train,
      num.trees = 2000,
      seed      = seed0 + rep
    )
    
    iv_pred   <- predict(iv_forest, newdata = X_test, estimate.variance = TRUE)
    tau_hat   <- iv_pred$predictions
    se_late   <- average_treatment_effect(iv_forest)[2]
    
    # Compute LATE and IATE metrics
    late_est  <- average_treatment_effect(iv_forest)[1]
    bias_late <- late_est - true_late
    
    data.frame(
      rep,
      true_late,
      grf_late            = late_est,
      bias_late_grf       = bias_late,
      mse_late_grf        = bias_late^2,
      mae_late_grf        = abs(bias_late),
      coverage_late_grf   = as.integer(
        (late_est - qnorm(0.975)*se_late <= true_late) &
          (true_late <= late_est + qnorm(0.975)*se_late)
      ),
      mse_grf_iate        = mean((tau_hat - tau_true)^2),
      mean_bias_grf_iate  = mean(tau_hat - tau_true),
      mae_grf_iate        = mean(abs(tau_hat - tau_true))
    )
  }
  
  # Parallel execution of replications
  results_list <- mclapply(seq_len(reps), one_rep, mc.cores = n_cores)
  results_df   <- do.call(rbind, results_list)
  
  # Add summary row (mean of all numeric columns)
  num_cols    <- sapply(results_df, is.numeric)
  summary_row <- as.data.frame(as.list(colMeans(results_df[, num_cols])))
  summary_row$rep <- NA_real_
  summary_row <- summary_row[names(results_df)]
  
  # Add median bias values (not row-averaged)
  med_bias_late <- median(results_df$bias_late_grf,      na.rm = TRUE)
  med_bias_iate <- median(results_df$mean_bias_grf_iate, na.rm = TRUE)
  
  # stick them on the summary_row only
  summary_row$median_bias_late_grf <- med_bias_late
  summary_row$median_bias_grf_iate  <- med_bias_iate
  
  results_df$median_bias_late_grf <- NA_real_
  results_df$median_bias_grf_iate  <- NA_real_
  
  # 5) bind mean‐summary, then all replicates
  rbind(summary_row, results_df)
}


# -------------------------------------------------
# Grid runner script (similar to runner.py)
# -------------------------------------------------

# 1. Define treatment functions, sample sizes, and pi‐map ----
treatment_functions <- c("step", "linear")
ns                  <- c(1000, 10000)

# For each n, a named vector of (weak, strong) pi‐values
pi_map <- list(
  `1000`  = c(weak = 0.3539, strong = 2.0587),
  `10000` = c(weak = 0.1134, strong = 2.0490)
)

# 2. Base args and default reps for n=1000 ----
base_sim_args <- list(
  u_size    = 5,
  n_size    = 5,
  k         = 6,
  rho_uv_val = 0.5
)
default_reps <- 1000

# 3. Output directory ----
out_dir <- "/Users/borisrine/.../results"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# 4. Run grid ----
for (fn in treatment_functions) {
  for (n in ns) {
    # pull the two pi‐values for this n
    pis <- pi_map[[as.character(n)]]
    
    # assign reps: 1k reps for n=1k, 250 for n=10k
    reps <- if (n == 10000) 250 else default_reps
    
    for (compliance in names(pis)) {
      pi_val <- pis[[compliance]]
      
      # build output filename
      out_fname <- sprintf(
        "grf_R_results_%s_%s_pi_%.4f_n%d.csv",
        fn, compliance, pi_val, n
      )
      out_path <- file.path(out_dir, out_fname)
      
      # skip if already done
      if (file.exists(out_path)) {
        message(sprintf(
          "[%s] Skipping %s | %s | pi=%.4f | n=%d (file exists)",
          Sys.time(), fn, compliance, pi_val, n
        ))
        next
      }
      
      message(sprintf(
        "[%s] Running %s | %s | pi=%.4f | n=%d | reps=%d",
        Sys.time(), fn, compliance, pi_val, n, reps
      ))
      
      # assemble sim args
      sim_args <- modifyList(base_sim_args, list(
        treatment_function = fn,
        pi_val             = pi_val,
        n                  = n,
        reps               = reps
      ))
      
      # run and save
      res <- run_monte_carlo_fast(reps = sim_args$reps,
                                  seed0 = 0,
                                  sim_args = sim_args)
      write.csv(res, file = out_path, row.names = FALSE)
      message("  → Saved to ", out_path)
    }
  }
}
