library(ggplot2)

GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, 
                           draw_group = function(self, data, ..., draw_quantiles = NULL) {
  data <- transform(data, 
                    xminv = x - violinwidth * (x - xmin), 
                    xmaxv = x + violinwidth * (xmax - x))
  grp <- data[1, "group"]
  newdata <- plyr::arrange(transform(data, 
                                     x = if (grp %% 2 == 1) xminv else xmaxv), 
                           if (grp %% 2 == 1) y else -y)
  newdata <- rbind(newdata[1, ], newdata, 
                   newdata[nrow(newdata), ], newdata[1, ])
  newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
  
  if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
    stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <= 1))
    quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
    aesthetics <- data[rep(1, nrow(quantiles)), 
                       setdiff(names(data), c("x", "y")), 
                       drop = FALSE]
    aesthetics$alpha <- rep(1, nrow(quantiles))
    both <- cbind(quantiles, aesthetics)
    quantile_grob <- GeomPath$draw_panel(both, ...)
    ggplot2:::ggname("geom_split_violin", 
                     grid::grobTree(GeomPolygon$draw_panel(newdata, ...), 
                                    quantile_grob))
  }
  else {
    ggplot2:::ggname("geom_split_violin", 
                     GeomPolygon$draw_panel(newdata, ...))
  }
})

geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", 
                              position = "identity", ..., 
                              draw_quantiles = NULL, trim = TRUE, 
                              scale = "area", na.rm = FALSE, 
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes, 
        params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}

createStanInitsPreviousRun=
  # function to initialize one stan run (via cmdstanr) from the endpoint of another
  # returns a function that returns inits.
  # this function is discussed on stan forums here: https://discourse.mc-stan.org/t/saving-reusing-adaptation-in-cmdstanr/19166/51
  # known bugs:
  #   errors (complaining about missing argument) when used inside a dplyr pipeline (e.g. "fit %>% createStanInitsPreviousRun()") because non standard evaluation somehow breaks the creation of the promise because aparrently the arguments dont get evaulated at the right moment and so the data cant be captured in the environment.
  
  # # useage examples:
  # 
  # # initial pure warmup run
  # lastRunResults=mod$sample(
  #     data=data_for_stan
#     ,chains=parallel_chains
#     ,parallel_chains=parallel_chains
#     # initialisation and adaptation settings
#     ,iter_warmup=iter_warmup # run automatic initialisation and adaptation
#     # sampling settings
#     ,iter_sampling=0
#     # settings to allow chains to be initialized from this one
#     ,save_warmup=T # for getting inits from a pure warmup run
#     ,sig_figs=18 # in crease output bitdepth to match the one used internally by stan. prevents rounding errors when initializing a run from this one (e.g. erroneously initializing at zero causing infinite density)
#     ,output_dir="stanSamples/" # makes stan save its output to a non-temporary directory (since the r-object returned doen't contain all the results, but instead links to the file). Often needed to be able to continue sampling a model in a separate r-session or after its associated r-object has been removed (to be precise: the standard temporary output file is deleten when the r-object representing the sampling results (and thus linking to this file) has bee ngrbage collected, or the R-session holding it closed.)
# )
# 
# # followup pure sampling runs
# lastRunResults=mod$sample(
#     data=data_for_stan
#     ,chains=parallel_chains
#     ,parallel_chains=parallel_chains
#     # initialisation and adaptation settings
#     ,iter_warmup=0 # use prespecified adaptation and inits
#     ,adapt_engaged=FALSE # use prespecified adaptation and inits
#     ,inv_metric=lastRunResults$inv_metric(matrix=F)
#     ,step_size=lastRunResults$metadata()$step_size_adaptation
#     ,init=createInitsFunction(lastRunResults)
#     # sampling settings
#     ,iter_sampling=iter_sample # difference
#     # settings to allow chains to be initialized from this one
#     ,save_warmup=T # needs to be set to TRUE even if no warmup is executed, because otherwise the function that extracts draws from the run complains since it is set to include warmup draws.
#     ,sig_figs=18 # increase output bitdepth to match the one used internally by stan. prevents rounding errors when initializing a run from this one (e.g. erroneously initializing at zero causing infinite density)
#     ,output_dir="stanSamples/" # makes stan save its output to a non-temporary directory (since the r-object returned doen't contain all the results, but instead links to the file). Often needed to be able to continue sampling a model in a separate r-session or after its associated r-object has been removed (to be precise: the standard temporary output file is deleten when the r-object representing the sampling results (and thus linking to this file) has bee ngrbage collected, or the R-session holding it closed.)
# )
function(
    lastRunResults # cmdstanr fit which was fitted with include_warmup=T (this is required even if the number of warmup draws was zero, because of an API limitation)
)
{
  initsFunction = function(chain_id){ # modified from https://discourse.mc-stan.org/t/saving-reusing-adaptation-in-cmdstanr/19166/44
    #   note: this function is not pure on its own. It captures lastRunResults from the enclosing environment, forming a pure closure.
    lastRun_draws = lastRunResults$draws(inc_warmup=T) # note that this row errors when used on a model which wasnt run with save_warmup=TRUE.
    final_warmup_value = lastRun_draws[dim(lastRun_draws)[1],chain_id,2:(dim(lastRun_draws)[3])]
    final_warmup_value %>% 
      tibble::as_tibble(
        .name_repair = function(names){
          dimnames(final_warmup_value)$variable
        }
      ) %>% 
      tidyr::pivot_longer(cols=dplyr::everything()) %>% 
      tidyr::separate(
        name
        , into = c('variable','index')
        , sep = "\\["
        , fill = 'right'
      ) %>% 
      dplyr::mutate(
        index = 
          stringr::str_replace(index,']','') %>% 
          {ifelse(!is.na(.),.,"")}
      ) %>% 
      dplyr::group_split(variable) %>% 
      purrr::map(
        .f = function(x){
          out= 
            x %>%
            dplyr::mutate(
              index=strsplit(index,",",fixed=TRUE)
              ,maxIndexes=
                do.call(rbind,index) %>%
                {array(as.numeric(.),dim=dim(.))} %>%
                plyr::aaply(2,max) %>%
                list()
            ) %>%
            {
              array(
                .$value
                ,dim=
                  if(length(.$maxIndexes[[1]])>0){
                    .$maxIndexes[[1]]
                  } else {
                    1
                  }
              )
            } %>% 
            list()
          names(out) = x$variable[1]
          out
        }
      ) %>% 
      unlist(recursive=F)
  }
  force(lastRunResults) # force evaluation of lastRunResults (i could've just written "lastRunResults=lastRunResults") to change its promise to an actual value that can get captured by the closure (since R has lazy evaluation objects dont get passed by reference or by copy, but by passing a promise to the object, if the object changes before the promise is evaluated then the promise returns the changed object instead of the "promised" one. Thus in that case the closure captures the changed object instead of the one originally passed (by promise) to its environment.)
  initsFunction # return the closure
}
