# Loac packages and source files ====
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(cmdstanr)


source("data_generation.R")
source("custom_functions.R")

# Generate the data ====
N <- 25
nT <- 50
seed <- 1294
gen_data <- generate_ssm_data(N, nT, seed)
mkdir("sim_data")
saveRDS(gen_data, "sim_data/data_N25T50S1294.rds")

# Run the stan model ====
mssm <- cmdstan_model("stan/multilevel_multivariate_ssm_lkj.stan")
mssm_data <- list(N = N,
                  `T` = nT,
                  y = gen_data$y)
n_chain <- 6
n_warmup <- 4000
n_sampling <- 4000


output_dir <- str_glue("stan/results/ssm_lkj_N{N}T{nT}_W{n_warmup}S{n_sampling}_S{seed}")
dir.create(output_dir)

mssm_fit <- mssm$sample(data = mssm_data, 
                        chains = n_chain, 
                        parallel_chains = n_chain, 
                        iter_warmup = n_warmup, 
                        iter_sampling = n_sampling, 
                        seed = seed, 
                        refresh = 2000, 
                        show_messages = FALSE,
                        output_dir = output_dir)
