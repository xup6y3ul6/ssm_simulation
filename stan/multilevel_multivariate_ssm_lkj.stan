#include ssm_function.stan

data {
  int<lower=1> N; // number of subjects
  int<lower=1> T; // number of observation for each subject
  array[N] matrix[2, T] y; // observations 
}

transformed data {
  vector[2] m_0; // prior mean of the intial state
  vector[2] diag_C_0; // diagonal of the prior covariance of the intial state
  
  m_0 = rep_vector(50.0, 2);
  diag_C_0 = rep_vector(sqrt(1000), 2);
}

parameters {
  array[N] vector[2] mu; // ground mean/trane
  array[N] vector[2] theta_0; // initial latent state
  array[N] matrix[2, T] theta; // latent states
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
  
  for (p in 1:2) {
    mu_R[p] = exp(gamma_tau_R[p]*2 + 0.5 * diag_Psi_tau_R[p]*4);
  }
}

model {
  // level 1 (within subject)
  array[N] matrix[2, T] mutheta;
  
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

  for (n in 1:N) {
    // prediction
    for (t in 1:T) {
      y_hat[n][, t] = mu[n] + theta[n][, t];
    }

    // within-subject reliability
    Tau[n] = to_matrix((identity_matrix(2 * 2) - kronecker_prod(Phi[n], Phi[n])) \ to_vector(Q[n]), 2, 2);

    for (p in 1:2) {
      rel_W[n, p] = Tau[n, p, p] / (Tau[n, p, p] + R[n, p, p]);
    }
  }

  // between-subject reliability
  for (p in 1:2) {
    rel_B[p] = Psi_mu[p, p] / (Psi_mu[p, p] + mean(Tau[, p, p]) + mu_R[p]);
  }
}
