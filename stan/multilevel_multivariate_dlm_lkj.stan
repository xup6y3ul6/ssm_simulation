#include ssm_function.stan

data {
  int<lower=1> N; // number of subjects
  int<lower=1> T; // number of observation for each subject
  array[N] matrix[2, T] y; // observations 
}

transformed data {
  vector[2] m_0; // prior mean of the intial state
  cov_matrix[2] C_0; // prior covariance of the intial state
  matrix[2, 2] F; // transition matrix
  
  m_0 = rep_vector(50.0, 2);
  C_0 = diag_matrix(rep_vector(1000.0, 2));
  F = diag_matrix(rep_vector(1.0, 2));
}

parameters {
  array[N] vector[2] mu; // ground mean/trane
  array[N] matrix[2, 2] Phi; // autoregressive parameters
  
  array[N] cholesky_factor_corr[2] L_Omega_R; 
  array[N] cholesky_factor_corr[2] L_Omega_Q;
  array[N] vector<lower=0>[2] tau_R;
  array[N] vector<lower=0>[2] tau_Q;
  
  vector[2] gamma_mu; // prior mean of the ground mean
  cov_matrix[2] Psi_mu; // prior covariance of the ground mean
  vector[2 * 2] gamma_Phi; // prior mean of the autoregressive parameters
  cov_matrix[2 * 2] Psi_Phi; // prior covariance of the autoregressive parameters
  
  vector<lower=0>[2] gamma_tau_R; 
  vector<lower=0>[2] diag_Psi_tau_R; 
  vector<lower=0>[2] gamma_tau_Q; 
  vector<lower=0>[2] diag_Psi_tau_Q; 
  real<lower=0> eta_R; // prior shape of the LKJ prior for the correlation matrix of the residuals
  real<lower=0> eta_Q; // prior shape of the LKJ prior for the correlation matrix of the latent states
  
}

transformed parameters {
  array[N] matrix[2, 2] L_Sigma_R;
  array[N] matrix[2, 2] L_Sigma_Q;
  array[N] matrix[2, 2] R;
  array[N] matrix[2, 2] Q;
  vector[2] mu_R;
   
  for (n in 1:N) {
    L_Sigma_R[n] = diag_pre_multiply(tau_R[n], L_Omega_R[n]);
    R[n] = L_Sigma_R[n] * L_Sigma_R[n]';

    L_Sigma_Q[n] = diag_pre_multiply(tau_Q[n], L_Omega_Q[n]);
    Q[n] = L_Sigma_Q[n] * L_Sigma_Q[n]';
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
    
    tau_R[n] ~ lognormal(gamma_tau_R, diag_Psi_tau_R);
    tau_Q[n] ~ lognormal(gamma_tau_Q, diag_Psi_tau_Q);
    L_Omega_R[n] ~ lkj_corr_cholesky(eta_R);
    L_Omega_Q[n] ~ lkj_corr_cholesky(eta_Q);
  }
  
  // the (hyper)priors of parameters are set as the Stan default values
}

generated quantities {
  array[N] matrix[2, T] y_hat;
  array[N] matrix[2, 2] Tau;
  array[N] vector[2] rel_W;
  vector[2] rel_B;
  vector[2] mu_R;

  for (n in 1:N) {
    // within-subject reliability
    Tau[n] = to_matrix((identity_matrix(2 * 2) - kronecker_prod(Phi[n], Phi[n])) \ to_vector(Q[n]), 2, 2);

    for (p in 1:2) {
      rel_W[n, p] = Tau[n, p, p] / (Tau[n, p, p] + R[n, p, p]);
    }
  }

  // between-subject reliability
  for (p in 1:2) {
    mu_R[p] = exp(2 * gamma_tau_R[p] + 2 * diag_Psi_tau_R[p]^2);
    rel_B[p] = Psi_mu[p, p] / (Psi_mu[p, p] + mean(Tau[, p, p]) + mu_R[p]);
  }
}
