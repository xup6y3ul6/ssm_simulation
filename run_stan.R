#library(lubridate)
#library(tsibble)
library(cmdstanr)
register_knitr_engine(override = FALSE)
#library(posterior)
#library(bayesplot)
#color_scheme_set("brewer-Spectral")
# library(loo)

source("custom_functions.R")
#pos_neg_color <- scales::hue_pal()(2)

gen_data <- readRDS("sim_data/data_N10T25.RDS")

mssm <- cmdstan_model("stan/multilevel_multivariate_ssm_lkj.stan")

mssm_data <- list(N = 10,
                  `T` = 25,
                  P = 2,
                  y = gen_data$y)


mssm_fit <- mssm$sample(data = mssm_data, 
                        chains = 2, 
                        parallel_chains = 2, 
                        iter_warmup = 100, 
                        iter_sampling = 100, 
                        seed = 1294, 
                        refresh = 2000, 
                        show_messages = TRUE)
