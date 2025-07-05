##### This code is identical to dgp_utils.py, for detailed comments see dgp_utils.py




generate_covariates <- function(n, u_size, n_size) {
  uni_min <- -sqrt(12) / 2
  uni_max <-  sqrt(12) / 2
  X_uniform <- matrix(
    runif(n * u_size, min = uni_min, max = uni_max),
    nrow = n, ncol = u_size
  )
  X_normal <- matrix(
    rnorm(n * n_size, mean = 0, sd = 1),
    nrow = n, ncol = n_size
  )
  X <- cbind(X_uniform, X_normal)
  return(X)
}

generate_instrument <- function(n) {
  Z <- rbinom(n, size = 1, prob = 0.5)
  return(Z)
}


generate_d <- function(X, Z, pi, v, beta) {
  normalizer <- sqrt(sum(beta^2) / 1.25)
  d <- as.vector(X %*% beta) / normalizer + pi * Z + v
  return(d)
}

assign_treatment <- function(d, treatment_share) {
  q <- quantile(d, probs = treatment_share, na.rm = TRUE)
  Tr <- as.integer(d > q)
  return(Tr)
}


tau_linear <- function(X, beta) {
  normalizer <- sqrt(sum(beta^2) / 1.25)
  zeta_iate  <- 0.42
  XB <- as.vector(X %*% beta)
  return((zeta_iate * XB) / normalizer + 1.0)
}

tau_step <- function(X, beta) {
  x1 <- X[, 1]
  x2 <- X[, 2]
  
  f1 <- 1.0 + 1.0 / (1.0 + exp(-20.0 * (x1 - 1.0/3.0)))
  f2 <- 1.0 + 1.0 / (1.0 + exp(-20.0 * (x2 - 1.0/3.0)))
  
  return(f1 * f2 - 2.8)
}

generate_errors <- function(n, rho_uv) {
  Sigma <- matrix(c(
    1.0,     rho_uv, rho_uv,
    rho_uv,  1.0,    0.0,
    rho_uv,  0.0,    1.0
  ), nrow = 3, byrow = TRUE)
  
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Package 'MASS' is required for generate_errors().")
  }
  draws <- MASS::mvrnorm(n = n, mu = rep(0, 3), Sigma = Sigma)
  
  v    <- draws[, 1]
  eps0 <- draws[, 2]
  eps1 <- draws[, 3]
  
  return(list(v = v, eps0 = eps0, eps1 = eps1))
}

generate_potential_outcomes <- function(X, IATE, eps0, eps1,
                                        function_type = c("trivial", "sine", "other"),
                                        beta, delta = 1) {
  normalizer <- sqrt(sum(beta^2) / 1.25)
  tilde_y    <- as.vector(X %*% beta) / normalizer
  
  function_type <- match.arg(function_type)
  
  if (function_type == "trivial") {
    y0 <- eps0
    y1 <- IATE + eps1
    
  } else if (function_type == "sine") {
    y0 <- delta * sin(tilde_y) + eps0
    y1 <- delta * sin(tilde_y) + IATE + eps1
    
  } else {
    y0 <- delta * sin(tilde_y) + eps0
    y1 <- delta * sin(tilde_y) + IATE + eps1
  }
  
  return(list(y0 = y0, y1 = y1))
}

simulate_once <- function(n, u_size, n_size, k,
                          treatment_function = c("linear", "step"),
                          pi_val, rho_uv_val, seed = NULL) {
  p <- u_size + n_size
  
  beta <- c(sapply(0:(k-1), function(j) 1 - j / k),
            rep(0, p - k))
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  X <- generate_covariates(n, u_size, n_size)
  
  treatment_function <- match.arg(treatment_function)
  if (treatment_function == "linear") {
    IATE <- tau_linear(X, beta)
  } else if (treatment_function == "step") {
    IATE <- tau_step(X, beta)
  } else {
    IATE <- rep(0, n)
  }
  
  Z   <- generate_instrument(n)
  errs <- generate_errors(n, rho_uv_val)
  v    <- errs$v
  eps0 <- errs$eps0
  eps1 <- errs$eps1
  
  d <- generate_d(X, Z, pi_val, v, beta)
  Tr <- assign_treatment(d, treatment_share = 0.5)
  
  out <- generate_potential_outcomes(
    X, IATE, eps0, eps1,
    function_type = "sine",
    beta = beta,
    delta = 1
  )
  y0 <- out$y0
  y1 <- out$y1
  Y  <- ifelse(Tr == 1, y1, y0)
  
  df <- as.data.frame(X)
  colnames(df) <- paste0("X", seq_len(ncol(X)))
  df$Z        <- Z
  df$Tr       <- Tr
  df$Y        <- Y
  df$tau_true <- IATE
  
  return(df)
}



get_true_late <- function(n_large, u_size, n_size, k,
                          treatment_function = c("linear", "step"),
                          pi_val, rho_uv, seed = 0) {
  treatment_function <- match.arg(treatment_function)
  
  p <- u_size + n_size
  beta <- c(sapply(0:(k-1), function(j) 1 - j/k),
            rep(0, p - k))
  normalizer <- sqrt(sum(beta^2) / 1.25)
  
  set.seed(seed)
  X    <- generate_covariates(n_large, u_size, n_size)
  errs <- generate_errors(n_large, rho_uv)
  v    <- errs$v
  
  mu  <- as.vector(X %*% beta) / normalizer
  d0  <- mu + v
  d1  <- mu + pi_val + v
  
  Z     <- generate_instrument(n_large)
  d_mix <- mu + pi_val * Z + v
  c_thr <- as.numeric(quantile(d_mix, probs = 0.5))
  
  Tr0 <- as.integer(d0 > c_thr)
  Tr1 <- as.integer(d1 > c_thr)
  
  tau_fn <- if (treatment_function == "linear") tau_linear else tau_step
  IATE   <- tau_fn(X, beta)
  
  complier_mask <- (Tr0 == 0 & Tr1 == 1)
  return(mean(IATE[complier_mask]))
}

