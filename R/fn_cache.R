# R/fn_cache.R — Memoisation helpers
# Loaded from app.R via source("R/fn_cache.R")
# Safe fallback if memoise is not installed: returns the function as-is.

.cache_dir <- file.path(getwd(), ".acustera_cache")
if (!dir.exists(.cache_dir)) dir.create(.cache_dir, showWarnings = FALSE)

have_memoise <- requireNamespace("memoise", quietly = TRUE) &&
                requireNamespace("cachem",   quietly = TRUE)

if (have_memoise) {
  .mem_cache <- cachem::cache_disk(dir = .cache_dir, max_size = 500 * 1024^2,
                                   max_age = 60 * 60 * 24 * 30)
  memo <- function(f) memoise::memoise(f, cache = .mem_cache)
} else {
  memo <- function(f) f
}

# Quick hash of a data frame to invalidate the cache
df_sig <- function(df, cols = NULL) {
  if (is.null(df) || nrow(df) == 0) return("empty")
  if (!is.null(cols)) cols <- intersect(cols, names(df))
  if (length(cols) == 0) cols <- names(df)[1:min(5, ncol(df))]
  digest::digest(list(dim(df), df[cols]), algo = "xxhash64")
}

cache_clear <- function() {
  if (have_memoise) tryCatch(.mem_cache$reset(), error = function(e) NULL)
  invisible(TRUE)
}
