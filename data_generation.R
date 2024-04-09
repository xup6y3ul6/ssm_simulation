
generate_ssm_data <- function(N, nT, seed = NULL, init = NULL) {
  if (!is.null(seed))  set.seed(seed)
  
  # data structure
  y <- rep(list(matrix(0, nrow = 2, ncol = nT)), N)
  mu <- vector("list", N)
  theta <- rep(list(matrix(0, nrow = 2, ncol = nT)), N)
  theta0 <- vector("list", N)
  Phi <- vector("list", N)
  epsilon <- vector("list", N)
  omega <- vector("list", N)
  tau_epsilon <- vector("list", N)
  Cor_epsilon <- vector("list", N)
  Sigma_epsilon <- vector("list", N)
  tau_omega <- vector("list", N)
  Cor_omega <- vector("list", N)
  Sigma_omega <- vector("list", N)
  
  
  # hyper-parameter settings
  m0 <- c(0, 0)
  C0 <- diag(c(10, 10))
  
  gamma_mu <- c(50, 20)
  Psi_mu <- diag(c(0.3, 0.3))
  
  gamma_Phi <- c(0.5, 0.2, -0.3, 0.5)
  Psi_Phi <- diag(c(0.2, 0.1, 0.1, 0.2))
   
  gamma_tau_epsilon <- c(1.2, 1.1)
  diag_Psi_tau_epsilon <- c(1, 0.8)
  eta_Cor_epsilon <- 3
  
  gamma_tau_omega <- c(-0.8, 1)
  diag_Psi_tau_omega <- c(0.5, 0.8)
  eta_Cor_omega <- 5
  
  # generate parameters
  for (n in 1:N) {
    mu[[n]] <- MASS::mvrnorm(1, gamma_mu, Psi_mu)
    
    Phi[[n]] <- MASS::mvrnorm(1, gamma_Phi, Psi_Phi) |> matrix(2, 2)
    
    tau_epsilon[[n]] <- rlnorm(2, gamma_tau_epsilon, sqrt(diag_Psi_tau_epsilon))
    Cor_epsilon[[n]] <- rethinking::rlkjcorr(1, 2, eta_Cor_epsilon)
    Sigma_epsilon[[n]] <- diag(tau_epsilon[[n]]) %*% Cor_epsilon[[n]] %*% diag(tau_epsilon[[n]])
    
    tau_omega[[n]] <- rlnorm(2, gamma_tau_omega, sqrt(diag_Psi_tau_omega))
    Cor_omega[[n]] <- rethinking::rlkjcorr(1, 2, eta_Cor_omega)
    Sigma_omega[[n]] <- diag(tau_omega[[n]]) %*% Cor_omega[[n]] %*% diag(tau_omega[[n]])
  }
  
  # generate data
  for (n in 1:N) {
    theta0[[n]] <- MASS::mvrnorm(1, m0, C0)
    
    omega[[n]] <- MASS::mvrnorm(nT, c(0, 0), Sigma_omega[[n]]) |> t()
    theta[[n]][, 1] <- Phi[[n]] %*% theta0[[n]] + omega[[n]][, 1]
    for (t in 2:nT) {
      theta[[n]][, t] <- Phi[[n]] %*% theta[[n]][, t-1] + omega[[n]][, t]
    }
    
    epsilon[[n]] <- MASS::mvrnorm(nT, c(0, 0), Sigma_epsilon[[n]]) |> t()
    y[[n]] <- mu[[n]] + theta[[n]] + epsilon[[n]] 
  }
  
  # return
  return(list(
    N = N,
    nT = nT,
    y = y,
    theta = theta,
    theta0 = theta0,
    Phi = Phi,
    tau_epsilon = tau_epsilon,
    Cor_epsilon = Cor_epsilon,
    Sigma_epsilon = Sigma_epsilon,
    tau_omega = tau_omega,
    Cor_omega = Cor_omega,
    Sigma_omega = Sigma_omega,
    m0 = m0, 
    C0 = C0,
    gamma_mu = gamma_mu,
    Psi_mu = Psi_mu,
    gamma_Phi = gamma_Phi,
    Psi_Phi = Psi_Phi,
    gamma_tau_epsilon = gamma_tau_epsilon,
    diag_Psi_tau_epsilon = diag_Psi_tau_epsilon,
    eta_Cor_epsilon = eta_Cor_epsilon,
    gamma_tau_omega = gamma_tau_omega,
    diag_Psi_tau_omega = diag_Psi_tau_omega,
    eta_Cor_omega = eta_Cor_omega
  ))
}