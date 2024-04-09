#include ssm_function.stan

data {
  int<lower=1> N; // number of subjects
  array[N] int<lower=1> T; // number of observation for each subject
  int<lower=1> max_T; // maximum number of observation
  int<lower=1> P; // number of affects
  array[N] matrix[P, max_T] y; // observations 
  array[N] vector[P] m_0; // prior mean of the intial state
  array[N] cov_matrix[P] C_0; // prior covariance of the intial state
}

parameters {
  array[N] vector[P] mu; // ground mean/trane
  array[N] vector[P] theta_0; // initial latent state
  array[N] matrix[P, max_T] theta; // latent states
  array[N] matrix[P, P] Phi; // autoregressive parameters
  array[N] cov_matrix[P] R; // covariance of the measurment error
  array[N] cov_matrix[P] Q; // covariance of the innovation noise
  
  vector[P] gamma_mu; // prior mean of the ground mean
  cov_matrix[P] Psi_mu; // prior covariance of the ground mean
  vector[P * P] gamma_Phi; // prior mean of the autoregressive parameters
  cov_matrix[P * P] Psi_Phi; // prior covariance of the autoregressive parameters
  vector[P * (P + 1) / 2] gamma_R; // prior mean of the covariance of the measurement error
  vector[P * (P + 1) / 2] diag_Psi_R; // prior covariance of the covariance of the measurement error
  vector[P * (P + 1) / 2] gamma_Q; // prior mean of the covariance of the innovation noise
  vector[P * (P + 1) / 2] diag_Psi_Q; // prior covariance of the covariance of innovation noise
}

model {
  // level 1 (within subject)
  for (n in 1:N) {
    // when t = 0
    theta_0[n] ~ multi_normal(m_0[n], C_0[n]);
  
    // when t = 1
    theta[n][, 1] ~ multi_normal(Phi[n] * theta_0[n], Q[n]);
    y[n][, 1] ~ multi_normal(mu[n] + theta[n][, 1], R[n]);
    
    // when t = 2, ..., T 
    for (t in 2:T[n]) {
      theta[n][, t] ~ multi_normal(Phi[n] * theta[n][, t - 1], Q[n]);
      y[n][, t] ~ multi_normal(mu[n] + theta[n][, t], R[n]);
    }
  }
  
  // level 2 (between subject)
  for (n in 1:N) {
    mu[n] ~ multi_normal(gamma_mu, Psi_mu);
    to_vector(Phi[n]) ~ multi_normal(gamma_Phi, Psi_Phi);
    to_vector_lower_tri(R[n]) ~ normal(gamma_R, sqrt(diag_Psi_R));
    to_vector_lower_tri(Q[n]) ~ normal(gamma_Q, sqrt(diag_Psi_Q));
  }
  
  // the (hyper)priors of parameters are set as the Stan default values
}

generated quantities {
  array[N] matrix[P, max_T] y_hat;
  array[N] matrix[P, P] Tau; 
  array[N] vector[P] rel_W;
  vector[P] rel_B;
  
  for (n in 1:N) {
    // prediction 
    for (t in 1:T[n]) {
      y_hat[n][, t] = mu[n] + theta[n][, t];
    }
    
    // within-subject reliability
    Tau[n] = to_matrix((identity_matrix(P * P) - kronecker_prod(Phi[n], Phi[n])) \ to_vector(Q[n]), P, P);
  
    for (p in 1:P) {
      rel_W[n, p] = Tau[n, p, p] / (Tau[n, p, p] + R[n, p, p]);
    }
  }
  
  // between-subject reliability
  for (p in 1:P) {
    rel_B[p] = Psi_mu[p, p] / (Psi_mu[p, p] + mean(Tau[, p, p]) + gamma_R[index_of_diag_lower_tri(p, P)]);
  }
}
