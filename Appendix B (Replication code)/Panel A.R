

# Load libraries and data
library(haven)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)
library(ggplot2)
library(ranger)
# suppress messages during fitting
lgr::get_logger("mlr3")$set_threshold("warn")

# load data as a data.table
data = as.data.table(read_dta("http://dmlguide.github.io/assets/dta/PVW_data.dta"))

# Define variables
y_col <- "net_tfa"
d_col <- "e401"
x_cols <- c("age", "inc", "tw", "fsize", "marr", "twoearn", "db", "pira", "hown")

Y <- data[[y_col]]
D <- data[[d_col]]
X <- data[, ..x_cols]

####### IPW Estimator #######



# Set up task and learner for propensity score estimation
task <- TaskClassif$new(id = "pscore", backend = data.table(D = as.factor(D), X), target = "D")
learner_ps <- lrn("classif.ranger",
                  num.trees = 1000,
                  max.depth = 4,
                  predict_type = "prob")
learner_ps$train(task)
r_hat <- learner_ps$predict(task)$prob[, 2]

# Compute plain IPW estimate
ipw_scores <- D * Y / r_hat - (1 - D) * Y / (1 - r_hat)
theta_ipw <- mean(ipw_scores)
n <- length(ipw_scores)
se_hat <- sqrt(var(ipw_scores) / n)

cat("Plain IPW ATE estimate: ", theta_ipw, "\n")
cat("The Standard Error of IPW theta is: ", se_hat, "\n")

####### IPW Estimator with crossfitting #######
# Required libraries
library(ranger)    # for random forest with probability=TRUE

# 1. Set up cross‐fitting
K <- 10
n <- nrow(X)
folds <- sample(rep(1:K, length.out = n))   # random assignment to folds

# Prepare storage for fold‐specific IPW scores
m_vals <- numeric(n)

# 2. Loop over folds
for (k in seq_len(K)) {
  train_idx <- which(folds != k)
  test_idx  <- which(folds == k)
  
  # Split training data
  X_tr <- X[train_idx, , drop = FALSE]
  D_tr <- D[train_idx]
  Y_tr <- Y[train_idx]
  
  # Fit propensity score model (random forest classifier)
  ps_fit <- ranger(
    dependent.variable.name = "D",
    data        = data.frame(D = as.factor(D_tr), X_tr),
    num.trees   = 1000,
    max.depth   = 4,
    probability = TRUE,
    seed        = 42,
    num.threads = parallel::detectCores()
  )
  
  # Predict propensity scores on test set
  X_te <- X[test_idx, , drop = FALSE]
  ps_pred <- predict(ps_fit, data = X_te)$predictions
  # Second column is P(D=1 | X)
  r_hat <- ps_pred[, "1"]
  
  # Compute IPW score: D*Y/r_hat − (1−D)*Y/(1−r_hat)
  D_te <- D[test_idx]
  Y_te <- Y[test_idx]
  m_vals[test_idx] <- D_te * Y_te / r_hat - (1 - D_te) * Y_te / (1 - r_hat)
}

# 3. Estimate θ̂ and its standard error
theta_hat <- mean(m_vals)
se_hat    <- sd(m_vals) / sqrt(n)

# 4. Print results
cat("The IPW theta with cross‐fitting is:", theta_hat, "\n")
cat("The standard error of IPW theta with cross‐fitting is:", se_hat, "\n")


####### RA estimator ##########
# 3a) Fit μ̂₁(x) = E[Y | X = x, D=1] using only X (no D column)
p <- ncol(X)
if (inherits(X, "data.table")) {
  data_treated <- cbind(data.table(Y = Y[D == 1]), X[D == 1])
} else {
  data_treated <- data.frame(Y = Y[D == 1], X[D == 1, , drop = FALSE])
}
task_mu1 <- TaskRegr$new(id = "mu1", backend = data_treated, target = "Y")

learner_mu1 <- lrn("regr.ranger",
                   num.trees = 1000,
                   max.depth = 8,
                   seed = 42,
                   predict_type = "response"
                   )
learner_mu1$train(task_mu1)

mu1_hat <- learner_mu1$predict_newdata(newdata = X)$response

# 3b) Fit μ̂₀(x) = E[Y | X = x, D=0]
if (inherits(X, "data.table")) {
  data_control <- cbind(data.table(Y = Y[D == 0]), X[D == 0])
} else {
  data_control <- data.frame(Y = Y[D == 0], X[D == 0, , drop = FALSE])
}
task_mu0 <- TaskRegr$new(id = "mu0", backend = data_control, target = "Y")

learner_mu0 <- lrn("regr.ranger",
                   num.trees = 1000,
                   max.depth = 8,
                   seed = 42,
                   predict_type = "response"
                  )
learner_mu0$train(task_mu0)

mu0_hat <- learner_mu0$predict_newdata(newdata = X)$response

# 5) Compute the plain RA estimator
ra_scores <- mu1_hat - mu0_hat
theta_ra <- mean(ra_scores)
n <- length(ra_scores)
se_hat <- sqrt(var(ra_scores) / n)

cat("Plain RA ATE estimate: ", theta_ra, "\n")
cat("The Standard error of the plain RA is: ", se_hat, "\n")


######## RA Estimator with crossfitting ####


# Assuming X is a data.frame or matrix of predictors,
# Y is a numeric vector of outcomes,
# D is a binary treatment indicator vector (0 or 1),
# n is the total sample size (n <- nrow(X))

set.seed(42)

# 1. Set up cross‐fitting
K <- 10
n <- nrow(X)
folds <- sample(rep(1:K, length.out = n))   # random assignment to folds

# Prepare storage for fold‐specific scores
m_vals <- numeric(n)

# 2. Loop over folds
for (k in seq_len(K)) {
  train_idx <- which(folds != k)
  test_idx  <- which(folds == k)
  
  # Subset training data by treatment group
  X_tr   <- X[train_idx, , drop = FALSE]
  Y_tr   <- Y[train_idx]
  D_tr   <- D[train_idx]
  
  # Outcomes for treated (D=1) and control (D=0)
  tr1_idx <- which(D_tr == 1)
  tr0_idx <- which(D_tr == 0)
  
  # Fit random forests
  rf1 <- ranger(
    dependent.variable.name = "Y",
    data       = data.frame(Y = Y_tr[tr1_idx], X_tr[tr1_idx, , drop = FALSE]),
    num.trees  = 1000,
    max.depth  = 8,
    seed       = 42,
    num.threads = parallel::detectCores()
  )
  rf0 <- ranger(
    dependent.variable.name = "Y",
    data       = data.frame(Y = Y_tr[tr0_idx], X_tr[tr0_idx, , drop = FALSE]),
    num.trees  = 1000,
    max.depth  = 8,
    seed       = 42,
    num.threads = parallel::detectCores()
  )
  
  # Predict on the test set
  X_te <- X[test_idx, , drop = FALSE]
  mu1_hat <- predict(rf1, data = X_te)$predictions
  mu0_hat <- predict(rf0, data = X_te)$predictions
  
  # Compute the score (difference in predicted outcomes)
  m_vals[test_idx] <- mu1_hat - mu0_hat
}

# 3. Estimate θ̂ and its standard error
theta_hat  <- mean(m_vals)
se_hat     <- sd(m_vals) / sqrt(n)

# 4. Print results
cat("The RA theta with cross‐fitting is:", theta_hat, "\n")
cat("The standard error of RA theta with cross‐fitting is:", se_hat, "\n")


##### Doubly Robust without Cross Ftting #####
# 1. Estimate propensity scores: r̂(X) = P(D=1 | X)
prop_full <- ranger(
  dependent.variable.name = "D",
  data        = data.frame(D = as.factor(D), X),
  num.trees   = 1000,
  max.depth   = 4,
  probability = TRUE,
  seed        = 42,
  num.threads = parallel::detectCores()
)
r_full <- predict(prop_full, data = X)$predictions[, "1"]

# 2. Train outcome models for treated and control
out1_full <- ranger(
  dependent.variable.name = "Y",
  data        = data.frame(Y = Y[D == 1], X[D == 1, , drop = FALSE]),
  num.trees   = 1000,
  max.depth   = 8,
  seed        = 42,
  num.threads = parallel::detectCores()
)
ell1_full <- predict(out1_full, data = X)$predictions

out0_full <- ranger(
  dependent.variable.name = "Y",
  data        = data.frame(Y = Y[D == 0], X[D == 0, , drop = FALSE]),
  num.trees   = 1000,
  max.depth   = 8,
  seed        = 42,
  num.threads = parallel::detectCores()
)
ell0_full <- predict(out0_full, data = X)$predictions

# 3. Compute the doubly robust score
alpha_full <- D / r_full - (1 - D) / (1 - r_full)
psi_nocf <- alpha_full * (Y - (D * ell1_full + (1 - D) * ell0_full)) + (ell1_full - ell0_full)

# 4. Estimate ATE and standard error
theta_nocf <- mean(psi_nocf)
se_nocf <- sd(psi_nocf) / sqrt(n)

# 5. Print result
cat("DR without cross‑fitting:\n")
cat(sprintf("  Estimate: %.2f, Std. Err: %.2f\n", theta_nocf, se_nocf))

###### Doubly Robust with Cross Fitting #####
set.seed(42)
K <- 10
n <- nrow(X)
folds <- sample(rep(1:K, length.out = n))

# storage for the DR scores
psi_vals <- numeric(n)

for (k in seq_len(K)) {
  # split
  train_idx <- which(folds != k)
  test_idx  <- which(folds == k)
  
  # training data
  X_tr <- X[train_idx, , drop = FALSE]
  D_tr <- D[train_idx]
  Y_tr <- Y[train_idx]
  
  # 1) propensity score on train, predict on test
  ps_fit <- ranger(
    dependent.variable.name = "D",
    data        = data.frame(D = as.factor(D_tr), X_tr),
    num.trees   = 1000,
    max.depth   = 4,
    probability = TRUE,
    seed        = 42
  )
  ps_pred <- predict(ps_fit, data = X[test_idx, , drop = FALSE])$predictions
  r_hat   <- ps_pred[, "1"]
  
  # 2) outcome model μ1 on treated train
  rf1 <- ranger(
    dependent.variable.name = "Y",
    data       = data.frame(Y = Y_tr[D_tr == 1],
                            X_tr[D_tr == 1, , drop = FALSE]),
    num.trees  = 1000,
    max.depth  = 8,
    seed       = 42
  )
  ell1_hat <- predict(rf1, data = X[test_idx, , drop = FALSE])$predictions
  
  # 3) outcome model μ0 on control train
  rf0 <- ranger(
    dependent.variable.name = "Y",
    data       = data.frame(Y = Y_tr[D_tr == 0],
                            X_tr[D_tr == 0, , drop = FALSE]),
    num.trees  = 1000,
    max.depth  = 8,
    seed       = 42
  )
  ell0_hat <- predict(rf0, data = X[test_idx, , drop = FALSE])$predictions
  
  # 4) compute DR‐score on test fold
  D_te <- D[test_idx]
  Y_te <- Y[test_idx]
  ell_hat <- D_te * ell1_hat + (1 - D_te) * ell0_hat
  
  psi_vals[test_idx] <- (D_te / r_hat - (1 - D_te) / (1 - r_hat)) * 
    (Y_te - ell_hat) +
    (ell1_hat - ell0_hat)
}

# 5) aggregate
theta_dr_cf <- mean(psi_vals)
se_dr_cf    <- sd(psi_vals) / sqrt(n)

cat("DR ATE (cross-fitted):", theta_dr_cf, "\n")
cat("SE (cross-fitted DR):",    se_dr_cf,    "\n")
