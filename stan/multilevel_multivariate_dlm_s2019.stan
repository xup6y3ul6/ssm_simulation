#include ssm_function.stan

data {
  int<lower=1> N; // number of subjects
  int<lower=1> T; // number of observation for each subject
  array[N] matrix[2, T] y; // observations 
}

transformed data {
  vector[2] m_0; // prior mean of the intial state
  cov_matrix[2] C_0; // diagonal of the prior covariance of the intial state
  matrix[2, 2] F; // transition matrix
  
  m_0 = rep_vector(50.0, 2);
  C_0 = diag_matrix(rep_vector(1000.0, 2));
  F = diag_matrix(rep_vector(1.0, 2));
}

parameters {
  // parameters
  array[N] vector[2] mu; // ground mean/trane
  array[N] matrix[2, 2] Phi; // autoregressive parameters
  
  vector<lower=0>[N] sigma2_epsilon;
  vector<lower=0>[N] sigma2_omega;
  array[N] vector<lower=0>[4] ervar; 
  
  // hyperparameters
  vector[2] gamma_mu; // prior mean of the ground mean
  cov_matrix[2] Psi_mu; // prior covariance of the ground mean
  vector[4] gamma_Phi; // prior mean of the autoregressive parameters
  cov_matrix[4] Psi_Phi; // prior covariance of the autoregressive parameters
  
  real gamma_log_sigma2_epsilon;
  real<lower=0> psi_log_sigma2_epsilon;
  real gamama_log_sigma2_omega;
  real<lower=0> psi_log_sigma2_omega;
  vector[4] gamma_log_ervar; 
  vector<lower=0>[4] psi_log_ervar;
}

transformed parameters {
  array[N] cov_matrix[2] R;
  array[N] cov_matrix[2] Q;
  
  for (n in 1:N) {
    R[n, 1, 1] = sigma2_epsilon[n] + ervar[n, 1];
    R[n, 1, 2] = sigma2_epsilon[n];
    R[n, 2, 1] = sigma2_epsilon[n];
    R[n, 2, 2] = sigma2_epsilon[n] + ervar[n, 2];
    Q[n, 1, 1] = sigma2_omega[n] + ervar[n, 3];
    Q[n, 1, 2] = sigma2_omega[n];
    Q[n, 2, 1] = sigma2_omega[n];
    Q[n, 2, 2] = sigma2_omega[n] + ervar[n, 4];
  }
}

model {
  // level 1 (within subject)
  for (n in 1:N) {
    y[n] ~ gaussian_dlm_obs(F, Phi[n], R[n], Q[n], m_0, C_0);
  }
  
  // level 2 (between subject)
  for (n in 1:N) {
    mu[n] ~ multi_normal(gamma_mu, Psi_mu);
    to_vector(Phi[n]) ~ multi_normal(gamma_Phi, Psi_Phi);
    ervar[n] ~ lognormal(gamma_log_ervar, psi_log_ervar);
  }
  
  sigma2_epsilon ~ lognormal(gamma_log_sigma2_epsilon, psi_log_sigma2_epsilon);
  sigma2_omega ~ lognormal(gamama_log_sigma2_omega, psi_log_sigma2_omeg |> a);
  
  
  // the (hyper)priors of parameters are set as the Stan default values
}

generated quantities {
  array[N] matrix[2, T] y_hat;
  array[N] matrix[2, 2] Tau;
  array[N] vector[2] rel_W;
  vector[2] mu_R;
  vector[2] rel_B;

  for (n in 1:N) {
    // within-subject reliability
    Tau[n] = to_matrix((identity_matrix(2 * 2) - kronecker_prod(Phi[n], Phi[n])) \ to_vector(Q[n]), 
                       2, 2);

    for (p in 1:2) {
      rel_W[n, p] = Tau[n, p, p] / (Tau[n, p, p] + R[n, p, p]);
    }
  }

  // between-subject reliability
  for (p in 1:2) {
    mu_R[p] = exp(gamma_log_sigma2_epsilon + 0.5 * psi_log_sigma2_epsilon^2) + 
      exp(gamma_log_ervar[p] + 0.5 * psi_log_ervar[p]^2);
    
    rel_B[p] = Psi_mu[p, p] / (Psi_mu[p, p] + mean(Tau[, p, p]) + mu_R[p]);
  }
}
