# Required packages
library(data.table)
library(haven)
library(ranger)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(parallel)

# ------------------------------------------------------------------------------
# Global variable definitions (as in Python's y_col, d_col, x_cols)
# ------------------------------------------------------------------------------
y_col  <- "net_tfa"
d_col  <- "e401"
x_cols <- c("age", "inc", "tw", "fsize", "marr", "twoearn", "db", "pira", "hown")

# ------------------------------------------------------------------------------
# 0. Load data
# ------------------------------------------------------------------------------
data <- as.data.table(
  read_dta("http://dmlguide.github.io/assets/dta/PVW_data.dta")
)
Y <- data[[y_col]]
D <- data[[d_col]]
# We'll use X both as a matrix (for numeric operations) and as a data.frame (for ranger)
X_mat <- as.matrix(data[, ..x_cols])
X_df  <- as.data.frame(data[, ..x_cols])
n      <- nrow(X_mat)

# ------------------------------------------------------------------------------
# Estimator functions
# ------------------------------------------------------------------------------

estimate_IPW_no_cf <- function(Y, D, X_df, true_ATE) {
  # 1. Fit propensity model
  
  ps_mod <- ranger(
    formula       = D ~ .,
    data          = cbind(D = D, X_df),
    num.trees     = 1000,
    max.depth     = NULL,
    min.node.size = NULL,
    num.threads   = detectCores(),
    probability   = TRUE
  )
  p_hat <- predict(ps_mod, X_df)$predictions[, 2]
  
  # 2. IPW estimate
  ipw_scores <- D * Y / p_hat - (1 - D) * Y / (1 - p_hat)
  theta_hat  <- mean(ipw_scores)
  
  # 3. SE
  se_hat <- sqrt(var(ipw_scores) / length(Y))
  
  # 4. Coverage
  ci_lower <- theta_hat - 1.96 * se_hat
  ci_upper <- theta_hat + 1.96 * se_hat
  covered  <- (ci_lower <= true_ATE && true_ATE <= ci_upper)
  
  list(theta_hat = theta_hat, se_hat = se_hat, covered = covered)
}

estimate_RA_no_cf <- function(Y, D, X_df, true_ATE) {
  # 1. Outcome regressions
  
  rf1 <- ranger(
    formula       = Y ~ .,
    data          = cbind(Y = Y[D == 1], X_df[D == 1, ]),
    num.trees     = 1000,
    max.depth     = 8,
    num.threads   = detectCores()
  )
  mu1_hat <- predict(rf1, X_df)$predictions
  
  
  rf0 <- ranger(
    formula       = Y ~ .,
    data          = cbind(Y = Y[D == 0], X_df[D == 0, ]),
    num.trees     = 1000,
    max.depth     = 8,
    num.threads   = detectCores()
  )
  mu0_hat <- predict(rf0, X_df)$predictions
  
  # 2. RA estimate
  ra_scores <- mu1_hat - mu0_hat
  theta_hat <- mean(ra_scores)
  
  # 3. SE
  se_hat <- sqrt(var(ra_scores) / length(Y))
  
  # 4. Coverage
  ci_lower <- theta_hat - 1.96 * se_hat
  ci_upper <- theta_hat + 1.96 * se_hat
  covered  <- (ci_lower <= true_ATE && true_ATE <= ci_upper)
  
  list(theta_hat = theta_hat, se_hat = se_hat, covered = covered)
}

estimate_DR_no_cf <- function(Y, D, X_df, true_ATE) {
  # 1. Propensity
  
  prop_mod <- ranger(
    formula     = D ~ .,
    data        = cbind(D = D, X_df),
    num.trees   = 1000,
    max.depth   = 4,
    num.threads = detectCores(),
    probability = TRUE
  )
  r_full <- predict(prop_mod, X_df)$predictions[, 2]
  
  # 2. Outcomes
  
  out1_mod <- ranger(
    formula       = Y ~ .,
    data          = cbind(Y = Y[D == 1], X_df[D == 1, ]),
    num.trees     = 1000,
    max.depth     = 8,
    num.threads   = detectCores()
  )
  ell1 <- predict(out1_mod, X_df)$predictions
  
  
  out0_mod <- ranger(
    formula       = Y ~ .,
    data          = cbind(Y = Y[D == 0], X_df[D == 0, ]),
    num.trees     = 1000,
    max.depth     = 8,
    num.threads   = detectCores()
  )
  ell0 <- predict(out0_mod, X_df)$predictions
  
  # 3. DR scores
  alpha <- D / r_full - (1 - D) / (1 - r_full)
  psi   <- alpha * (Y - (D * ell1 + (1 - D) * ell0)) + (ell1 - ell0)
  
  # 4. Estimate & SE
  theta_hat <- mean(psi)
  se_hat    <- sd(psi) / sqrt(length(psi))
  
  # 5. Coverage
  ci_lower <- theta_hat - 1.96 * se_hat
  ci_upper <- theta_hat + 1.96 * se_hat
  covered  <- (ci_lower <= true_ATE && true_ATE <= ci_upper)
  
  list(theta_hat = theta_hat, se_hat = se_hat, covered = covered)
}

estimate_IPW_cf <- function(Y, D, X_df, true_ATE) {
  K <- 10
  n <- length(Y)
  
  folds <- sample(rep(1:K, length.out = n))
  m_vals <- numeric(n)
  
  for (k in seq_len(K)) {
    train <- which(folds != k)
    test  <- which(folds == k)
    
    # Propensity on train
    ps_mod <- ranger(
      formula       = D ~ .,
      data          = cbind(D = D[train], X_df[train, ]),
      num.trees     = 1000,
      max.depth     = 4,
      num.threads   = detectCores(),
      probability   = TRUE
    )
    p_hat <- predict(ps_mod, X_df[test, ])$predictions[, 2]
    
    # IPW scores
    m_vals[test] <- D[test] * Y[test] / p_hat -
      (1 - D[test]) * Y[test] / (1 - p_hat)
  }
  
  theta_hat <- mean(m_vals)
  se_hat    <- sqrt(var(m_vals) / n)
  ci_lower  <- theta_hat - 1.96 * se_hat
  ci_upper  <- theta_hat + 1.96 * se_hat
  covered   <- (ci_lower <= true_ATE && true_ATE <= ci_upper)
  
  list(theta_hat = theta_hat, se_hat = se_hat, covered = covered)
}

estimate_RA_cf <- function(Y, D, X_df, true_ATE) {
  K <- 10
  n <- length(Y)
  
  folds <- sample(rep(1:K, length.out = n))
  m_vals <- numeric(n)
  
  for (k in seq_len(K)) {
    train <- which(folds != k)
    test  <- which(folds == k)
    
    # Outcome models on train
    rf1 <- ranger(
      formula       = Y ~ .,
      data          = cbind(Y = Y[train][D[train] == 1], X_df[train,][D[train] == 1, ]),
      num.trees     = 1000,
      max.depth     = 8,
      num.threads   = detectCores()
    )
    rf0 <- ranger(
      formula       = Y ~ .,
      data          = cbind(Y = Y[train][D[train] == 0], X_df[train,][D[train] == 0, ]),
      num.trees     = 1000,
      max.depth     = 8,
      num.threads   = detectCores()
    )
    
    mu1_hat <- predict(rf1, X_df[test, ])$predictions
    mu0_hat <- predict(rf0, X_df[test, ])$predictions
    
    m_vals[test] <- mu1_hat - mu0_hat
  }
  
  theta_hat <- mean(m_vals)
  se_hat    <- sqrt(var(m_vals) / n)
  ci_lower  <- theta_hat - 1.96 * se_hat
  ci_upper  <- theta_hat + 1.96 * se_hat
  covered   <- (ci_lower <= true_ATE && true_ATE <= ci_upper)
  
  list(theta_hat = theta_hat, se_hat = se_hat, covered = covered)
}


estimate_DML <- function(Y, D, X_df, true_ATE) {
  # Y: numeric vector (n)
  # D: binary {0,1} vector (n)
  # X_df: data.frame or data.table of covariates (n x p)
  # true_ATE: numeric scalar
  # K: number of folds (default 10)
  # seed: RNG seed
  # n_trees, max_depth_ps, max_depth_mu: ranger tuning
  K = 10
  n_trees = 1000
  max_depth_ps = 4
  max_depth_mu = 8
  n <- length(Y)
  X <- as.matrix(X_df)  # numeric covariate matrix
  
  # create fold assignments
  folds <- sample(rep(1:K, length.out = n))
  
  # placeholder for influence-function values
  psi_vals <- numeric(n)
  
  for (k in seq_len(K)) {
    # indices
    train_idx <- which(folds != k)
    test_idx  <- which(folds == k)
    
    # split data
    X_tr <- X[train_idx, , drop = FALSE]
    D_tr <- D[train_idx]
    Y_tr <- Y[train_idx]
    X_te <- X[test_idx, , drop = FALSE]
    D_te <- D[test_idx]
    Y_te <- Y[test_idx]
    
    # 1) propensity score r_hat(x) = P(D=1|x)
    ps_fit <- ranger(
      dependent.variable.name = "D",
      data       = data.frame(D = factor(D_tr), X_tr),
      num.trees  = n_trees,
      max.depth  = max_depth_ps,
      probability= TRUE,
      num.threads = detectCores()
    )
    ps_pred <- predict(ps_fit, data = X_te)$predictions
    r_hat   <- ps_pred[, "1"]
    
    # 2) outcome model μ1(x) for the treated
    rf1 <- ranger(
      dependent.variable.name = "Y",
      data        = data.frame(Y = Y_tr[D_tr == 1],
                               X_tr[D_tr == 1, , drop = FALSE]),
      num.trees   = n_trees,
      max.depth   = max_depth_mu,
      num.threads = detectCores()
    )
    ell1_hat <- predict(rf1, data = X_te)$predictions
    
    # 3) outcome model μ0(x) for the controls
    rf0 <- ranger(
      dependent.variable.name = "Y",
      data        = data.frame(Y = Y_tr[D_tr == 0],
                               X_tr[D_tr == 0, , drop = FALSE]),
      num.trees   = n_trees,
      max.depth   = max_depth_mu,
    )
    ell0_hat <- predict(rf0, data = X_te)$predictions
    
    # 4) doubly‐robust scores
    ell_hat <- D_te * ell1_hat + (1 - D_te) * ell0_hat
    psi_vals[test_idx] <- (D_te / r_hat - (1 - D_te) / (1 - r_hat)) * (Y_te - ell_hat) +
      (ell1_hat - ell0_hat)
  }
  
  # 5) aggregate
  theta_hat <- mean(psi_vals)
  se_hat    <- sd(psi_vals) / sqrt(n)
  
  # confidence interval
  ci_lower <- theta_hat - qnorm(0.975) * se_hat
  ci_upper <- theta_hat + qnorm(0.975) * se_hat
  covered  <- (ci_lower <= true_ATE && true_ATE <= ci_upper)
  
  return(list(
    theta_hat = theta_hat,
    se_hat    = se_hat,
    ci        = c(ci_lower, ci_upper),
    covered   = covered
  ))
}



# ------------------------------------------------------------------------------
##### Main simulation driver #####
# ------------------------------------------------------------------------------
main <- function() {
  # 1. Reduced‐form models (full‐sample)
  
  prop_model <- ranger(
    formula     = D ~ .,
    data        = cbind(D = D, X_df),
    num.trees   = 1000,
    min.node.size = 10,
    num.threads = detectCores(),
    probability = TRUE
  )
  r_hat <- predict(prop_model, X_df)$predictions[, 2]
  
  
  out1_model <- ranger(
    formula       = Y ~ .,
    data          = cbind(Y = Y[D == 1], X_df[D == 1, ]),
    num.trees     = 1000,
    min.node.size = 10,
    num.threads   = detectCores()
  )
  ell1_hat <- predict(out1_model, X_df)$predictions
  
  
  out0_model <- ranger(
    formula       = Y ~ .,
    data          = cbind(Y = Y[D == 0], X_df[D == 0, ]),
    num.trees     = 1000,
    min.node.size = 10,
    num.threads   = detectCores()
  )
  ell0_hat <- predict(out0_model, X_df)$predictions
  
  # 2. Residual variances
  sigma2_1 <- mean((Y[D == 1] - ell1_hat[D == 1])^2)
  sigma2_0 <- mean((Y[D == 0] - ell0_hat[D == 0])^2)
  
  # 3. Simulations
  n_sims   <- 1150
  methods  <- c("IPW_no_cf","RA_no_cf","DR_no_cf","IPW_cf","RA_cf","DML")
  results  <- lapply(methods, function(m) list(est = numeric(n_sims),
                                               se  = numeric(n_sims),
                                               cov = logical(n_sims)))
  names(results) <- methods
  
  true_ATE <- mean(ell1_hat - ell0_hat)
  est_funcs <- list(IPW_no_cf = estimate_IPW_no_cf,
                    RA_no_cf  = estimate_RA_no_cf,
                    DR_no_cf  = estimate_DR_no_cf,
                    IPW_cf    = estimate_IPW_cf,
                    RA_cf     = estimate_RA_cf,
                    DML       = estimate_DML)
  set.seed(42)
  for (s in seq_len(n_sims)) {
    D_sim <- rbinom(n, 1, r_hat)
    eps   <- numeric(n)
    eps[D_sim == 1] <- rnorm(sum(D_sim == 1), 0, sqrt(sigma2_1))
    eps[D_sim == 0] <- rnorm(sum(D_sim == 0), 0, sqrt(sigma2_0))
    Y_sim <- ifelse(D_sim == 1, ell1_hat, ell0_hat) + eps
    
    for (m in methods) {
      res <- est_funcs[[m]](Y_sim, D_sim, X_df, true_ATE)
      results[[m]]$est[s] <- res$theta_hat
      results[[m]]$se[s]  <- res$se_hat
      results[[m]]$cov[s] <- res$covered
    }
  }
  
  # 4. Summary (Table 2, Panel B)
  summary_dt <- data.table(
    Method                 = methods,
    Mean_Bias              = sapply(methods, function(m) mean(results[[m]]$est - true_ATE)),
    Median_Bias            = sapply(methods, function(m) median(results[[m]]$est - true_ATE)),
    `Median Abs. Dev.`     = sapply(methods, function(m) median(abs(results[[m]]$est - true_ATE))),
    `Std. Dev.`            = sapply(methods, function(m) sd(results[[m]]$est)),
    `Mean Standard Error`  = sapply(methods, function(m) mean(results[[m]]$se)),
    `Coverage Rate (0.95)` = sapply(methods, function(m) mean(results[[m]]$cov))
  )
  cat(sprintf("True ATE (calibrated): %.3f\n\n", true_ATE))
  print(summary_dt)
  
  invisible(summary_dt)
}

system.time({
  main()
  Sys.sleep(2)  # optional, just simulates pause
  cat("Script done\n")
})
