#include ssm-function.stan

data {
  int<lower=1> N; // number of subjects
  array[N] int<lower=1> T; // number of observation for each subject
  int<lower=1> max_T; // maximum number of observation
  array[N] matrix[2, max_T] y; // observations 
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
  array[N] matrix[2, max_T] theta; // latent states
  array[N] matrix[2, 2] Phi; // autoregressive parameters
  
  array[N] vector<lower=0>[2] tau_R; 
  array[N] vector<lower=0>[2] tau_Q; 
  array[N] corr_matrix[2] Sigma_R;
  array[N] corr_matrix[2] Sigma_Q;
  
  
  vector[2] gamma_mu; // prior mean of the ground mean
  cov_matrix[2] Psi_mu; // prior covariance of the ground mean
  vector[4] gamma_Phi; // prior mean of the autoregressive parameters
  cov_matrix[4] Psi_Phi; // prior covariance of the autoregressive parameters
  
}

transformed parameters {
  array[N] cov_matrix[2] R;
  array[N] cov_matrix[2] Q;
  
  for (n in 1:N) {
    R[n] = quad_form_diag(Sigma_R[n], tau_R[n]);
    Q[n] = quad_form_diag(Sigma_Q[n], tau_Q[n]);
  }
  
}

model {
  // level 1 (within subject)
  array[N] matrix[2, max_T+1] theta_0T;
  array[N] matrix[2, max_T] mutheta;
  array[N] matrix[2, max_T+1] Phitheta;
  
  for (n in 1:N) {
    theta_0[n] ~ normal(m_0, diag_C_0);
    theta_0T[n] = append_col(theta_0[n], theta[n]);
    Phitheta[n, , 1:T[n]] = Phi[n] * theta_0T[n, , 1:T[n]];
    mutheta[n, , 1:T[n]] = mu[n] * rep_row_vector(1.0, T[n]) + theta[n, , 1:T[n]];
    
    for (t in 1:T[n]) {
      theta[n, , t] ~ multi_normal(Phitheta[n, , t], Q[n]);
      y[n, , t] ~ multi_normal(mutheta[n, , t], R[n]);
    }
  }
  
  // level 2 (between subject)
  for (n in 1:N) {
    mu[n] ~ multi_normal(gamma_mu, Psi_mu);
    to_vector(Phi[n]) ~ multi_normal(gamma_Phi, Psi_Phi);

    tau_R[n](2 elements) ~ uniform(0, 100);
    tau_Q[n] ~ uniform(0, 100);
    Sigma_R[n][1, 2] ~ uniform(-1, 1);
    Sigma_Q[n][1, 2] ~ uniform(-1, 1);
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
