#include ssm-function.stan

data {
  int<lower=1> N; // number of subjects
  array[N] int<lower=1> T; // number of observation for each subject
  int<lower=1> max_T; // maximum number of observation
  int<lower=1> P; // number of affects
  array[N] matrix[P, max_T] y; // observations 
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
  array[N] matrix[P, max_T] theta; // latent states
  array[N] matrix[P, P] Phi; // autoregressive parameters
  
  //array[N] matrix[P, max_T] epsilon;
  //array[N] matrix[P, max_T] omega;
  array[N] cholesky_factor_corr[P] L_Omega_R; 
  array[N] cholesky_factor_corr[P] L_Omega_Q;
  array[N] vector<lower=0>[P] tau_R;
  array[N] vector<lower=0>[P] tau_Q;
  
  vector[P] gamma_mu; // prior mean of the ground mean
  cov_matrix[P] Psi_mu; // prior covariance of the ground mean
  vector[P * P] gamma_Phi; // prior mean of the autoregressive parameters
  cov_matrix[P * P] Psi_Phi; // prior covariance of the autoregressive parameters
  
}

transformed parameters {
  // array[N] cov_matrix[P] R; // covariance of the measurment error
  // array[N] cov_matrix[P] Q; // covariance of the innovation noise
  // for (n in 1:N) {
  //   R[n] = diag_pre_multiply(tau_R[n], L_Omega_R[n]) * diag_pre_multiply(tau_R[n], L_Omega_R[n])';
  //   Q[n] = diag_pre_multiply(tau_Q[n], L_Omega_Q[n]) * diag_pre_multiply(tau_Q[n], L_Omega_Q[n])';
  // }
  
  array[N] matrix[P, P] L_Sigma_R;
  array[N] matrix[P, P] L_Sigma_Q;
   
  for (n in 1:N) {
    L_Sigma_R[n] = diag_pre_multiply(tau_R[n], L_Omega_R[n]);
    L_Sigma_Q[n] = diag_pre_multiply(tau_Q[n], L_Omega_Q[n]);
  }
}

model {
  // level 1 (within subject)
  array[N] matrix[P, max_T+1] theta_0T;
  array[N] matrix[P, max_T] mutheta;
  array[N] matrix[P, max_T+1] Phitheta;
  
  for (n in 1:N) {
    theta_0[n] ~ normal(m_0, diag_C_0);
    theta_0T[n] = append_col(theta_0[n], theta[n]);
    Phitheta[n, , 1:T[n]] = Phi[n] * theta_0T[n, , 1:T[n]];
    mutheta[n, , 1:T[n]] = mu[n] * rep_row_vector(1.0, T[n]) + theta[n, , 1:T[n]];
    
    for (t in 1:T[n]) {
      theta[n, , t] ~ multi_normal_cholesky(Phitheta[n, , t], L_Sigma_Q[n]);
      y[n, , t] ~ multi_normal_cholesky(mutheta[n, , t], L_Sigma_R[n]);
    }
  }
  
  // level 2 (between subject)
  for (n in 1:N) {
    mu[n] ~ multi_normal(gamma_mu, Psi_mu);
    to_vector(Phi[n]) ~ multi_normal(gamma_Phi, Psi_Phi);

    tau_R[n] ~ cauchy(0, 2.5);
    tau_Q[n] ~ cauchy(0, 2.5);
    L_Omega_R[n] ~ lkj_corr_cholesky(2);
    L_Omega_Q[n] ~ lkj_corr_cholesky(2);
  }
  
  // the (hyper)priors of parameters are set as the Stan default values
}

generated quantities {
  // array[N] matrix[P, max_T] y_hat;
  // array[N] matrix[P, P] Tau; 
  // array[N] vector[P] rel_W;
  // vector[P] rel_B;
  // 
  // for (n in 1:N) {
  //   // prediction 
  //   for (t in 1:T[n]) {
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
