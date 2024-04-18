#include ssm_function.stan

data {
  int<lower=1> N; // number of subjects
  int<lower=1> T; // number of observation for each subject
  int<lower=1> P; // number of affects
  array[N] matrix[P, T] y; // observations 
}

transformed data {
  vector[P] m_0; // prior mean of the intial state
  //cov_matrix[P] C_0; // prior covariance of the intial state
  vector[P] diag_C_0; // diagonal of the prior covariance of the intial state
  
  m_0 = rep_vector(50.0, P);
  //C_0 = diag_matrix(rep_vector(sqrt(1000), P));
  diag_C_0 = rep_vector(sqrt(1000), P);
}

parameters {
  array[N] vector[P] mu; // ground mean/trane
  array[N] vector[P] theta_0; // initial latent state
  array[N] matrix[P, T] theta; // latent states
  array[N] matrix[P, P] Phi; // autoregressive parameters
  
  array[N] cholesky_factor_corr[P] L_Omega_R; 
  array[N] cholesky_factor_corr[P] L_Omega_Q;
  array[N] vector<lower=0>[P] tau_R;
  array[N] vector<lower=0>[P] tau_Q;
  
  vector[P] gamma_mu; // prior mean of the ground mean
  cov_matrix[P] Psi_mu; // prior covariance of the ground mean
  vector[P * P] gamma_Phi; // prior mean of the autoregressive parameters
  cov_matrix[P * P] Psi_Phi; // prior covariance of the autoregressive parameters
  
  // vector<lower=0>[P] alpha_tau_R; // prior shape of the inverse gamma prior for the residual variance
  // vector<lower=0>[P] beta_tau_R; // prior scale of the inverse gamma prior for the residual variance
  // vector<lower=0>[P] alpha_tau_Q; // prior shape of the inverse gamma prior for the latent state variance
  // vector<lower=0>[P] beta_tau_Q; // prior scale of the inverse gamma prior for the latent state variance
  real<lower=0> eta_R; // prior shape of the LKJ prior for the correlation matrix of the residuals
  real<lower=0> eta_Q; // prior shape of the LKJ prior for the correlation matrix of the latent states
  
}

transformed parameters {
  array[N] matrix[P, P] L_Sigma_R;
  array[N] matrix[P, P] L_Sigma_Q;
  // array[N] cov_matrix[P] R;
  // array[N] cov_matrix[P] Q;
  // vector[P] mu_R;
   
  for (n in 1:N) {
    L_Sigma_R[n] = diag_pre_multiply(tau_R[n], L_Omega_R[n]);
    // R[n] = L_Sigma_R[n] * L_Sigma_R[n]';
    
    L_Sigma_Q[n] = diag_pre_multiply(tau_Q[n], L_Omega_Q[n]);
    // Q[n] = L_Sigma_Q[n] * L_Sigma_Q[n]';
  }
  
  // for (p in 1:P) {
  //   mu_R[p] = beta_tau_R[p] / (alpha_tau_R[p] - 1);
  // }
}

model {
  // level 1 (within subject)
  //array[N] matrix[P, T+1] theta_0T;
  array[N] matrix[P, T] mutheta;
  //array[N] matrix[P, T+1] Phitheta;
  
  for (n in 1:N) {
      theta_0[n] ~ normal(m_0, diag_C_0);
      theta[n, , 1] ~ multi_normal_cholesky(Phi[n] * theta_0[n], L_Sigma_Q[n]);
      
      for (t in 2:T) {
        theta[n, , t] ~ multi_normal_cholesky(Phi[n] * theta[n, , t-1], L_Sigma_Q[n]);
      }
      
      mutheta[n, , 1:T] = mu[n] * rep_row_vector(1.0, T) + theta[n, , 1:T];
      
      for (t in 1:T) {
        y[n, , t] ~ multi_normal_cholesky(mutheta[n, , t], L_Sigma_R[n]);
      }
    }
  
  // level 2 (between subject)
  for (n in 1:N) {
    mu[n] ~ multi_normal(gamma_mu, Psi_mu);
    to_vector(Phi[n]) ~ multi_normal(gamma_Phi, Psi_Phi);
    
    // tau_R[n] ~ inv_gamma(alpha_tau_R, beta_tau_R);
    // tau_Q[n] ~ inv_gamma(alpha_tau_Q, beta_tau_Q);
    
    tau_R[n] ~ cauchy(0, 2.5);
    tau_Q[n] ~ cauchy(0, 2.5);
    // L_Omega_R[n] ~ lkj_corr_cholesky(eta_R);
    // L_Omega_Q[n] ~ lkj_corr_cholesky(eta_Q);
    L_Omega_R[n] ~ lkj_corr_cholesky(eta_R);
    L_Omega_Q[n] ~ lkj_corr_cholesky(eta_Q);
  }
  
  // the (hyper)priors of parameters are set as the Stan default values
}

generated quantities {
  // array[N] matrix[P, T] y_hat;
  // array[N] matrix[P, P] Tau;
  // array[N] vector[P] rel_W;
  // vector[P] rel_B;
  // 
  // for (n in 1:N) {
  //   // prediction
  //   for (t in 1:T) {
  //     y_hat[n][, t] = mu[n] + theta[n][, t];
  //   }
  // 
  //   // within-subject reliability
  //   Tau[n] = to_matrix((identity_matrix(P * P) - kronecker_prod(Phi[n], Phi[n])) \ to_vector(Q[n]), P, P);
  // 
  //   for (p in 1:P) {
  //     rel_W[n, p] = Tau[n, p, p] / (Tau[n, p, p] + R[n, p, p]);
  //   }
  // }
  // 
  // // between-subject reliability
  // for (p in 1:P) {
  //   rel_B[p] = Psi_mu[p, p] / (Psi_mu[p, p] + mean(Tau[, p, p]) + mu_R[p]);
  // }
}
