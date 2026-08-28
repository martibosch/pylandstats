# Measure the execution time of computing the landscape metrics with landscapemetrics,
# i.e., the reference values reported in the performance notes annex of the user guide
# (docs/user-guide/a02-performance-notes.ipynb).
#
# Run it with `pixi run -e benchmark benchmark-landscapemetrics <raster> [<raster> ...]`.
#
# ACHTUNG: ported from the `metricsBenchmark.R` script of the pylandstats-notebooks
# repository, which used the `raster` package (retired in favour of `terra`).

library(landscapemetrics)
library(terra)

metrics_benchmark <- function(landscape_filepath) {
  landscape <- rast(landscape_filepath)

  # landscapemetrics only implements the mean, standard deviation and coefficient of
  # variation of each patch metric (whereas pylandstats also implements the
  # area-weighted mean, the median and the range), so the comparison is restricted to
  # these three
  suffixes <- c("mn", "sd", "cv")
  patch_metrics <- c(
    "lsm_p_area", "lsm_p_perim", "lsm_p_para", "lsm_p_shape", "lsm_p_frac", "lsm_p_enn"
  )

  aggregation_metrics <- function(level) {
    level_metrics <- gsub("_p_", paste0("_", level, "_"), patch_metrics)
    unlist(lapply(suffixes, function(suffix) paste(level_metrics, suffix, sep = "_")))
  }

  class_metrics <- c(
    "lsm_c_ca", "lsm_c_pland", "lsm_c_np", "lsm_c_pd", "lsm_c_lpi", "lsm_c_te",
    "lsm_c_ed", "lsm_c_lsi", aggregation_metrics("c")
  )
  landscape_metrics <- c(
    "lsm_l_ta", "lsm_l_np", "lsm_l_pd", "lsm_l_lpi", "lsm_l_te", "lsm_l_ed",
    "lsm_l_lsi", "lsm_l_contag", "lsm_l_shdi", aggregation_metrics("l")
  )
  metrics <- c(patch_metrics, class_metrics, landscape_metrics)

  elapsed <- system.time(result <- calculate_lsm(landscape, what = metrics))
  cat(sprintf(
    "%s: %d rows in %.2f s (elapsed)\n",
    basename(landscape_filepath), nrow(result), elapsed[["elapsed"]]
  ))

  invisible(result)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("provide at least one raster file path")
}
cat(sprintf(
  "landscapemetrics %s, terra %s, R %s.%s\n",
  packageVersion("landscapemetrics"), packageVersion("terra"),
  R.version$major, R.version$minor
))
for (landscape_filepath in args) {
  metrics_benchmark(landscape_filepath)
}
