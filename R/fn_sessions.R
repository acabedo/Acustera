# R/fn_sessions.R — Save/load complete analyses to a single .rds file.
# An "analysis" = set of features + transcript + overlaps + sess_meta + spk_meta
# for one or more couples, with date/author metadata.

acustera_sessions_dir <- function() {
  d <- file.path(getwd(), "sessions")
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  d
}

acustera_session_save <- function(rv, label = NULL) {
  if (is.null(rv$features) || nrow(rv$features) == 0)
    stop("No features to save.")
  ts     <- format(Sys.time(), "%Y%m%d_%H%M%S")
  pairs  <- sort(unique(as.character(rv$features$couple)))
  n_seg  <- nrow(rv$features)
  n_sess <- length(unique(paste(rv$features$couple, rv$features$therapy_num)))
  lbl    <- if (!is.null(label) && nzchar(label))
              gsub("[^A-Za-z0-9_-]+","_", label)
            else paste(pairs, collapse = "-")
  fname  <- sprintf("%s__%s.rds", ts, substr(lbl, 1, 40))
  path   <- file.path(acustera_sessions_dir(), fname)
  payload <- list(
    meta = list(
      saved_at  = Sys.time(),
      label     = label %||% paste(pairs, collapse = ", "),
      pairs     = pairs,
      n_pairs   = length(pairs),
      n_sess    = n_sess,
      n_seg     = n_seg,
      has_trans = !is.null(rv$transcript) && nrow(rv$transcript) > 0,
      has_ov    = !is.null(rv$overlaps)   && nrow(rv$overlaps)   > 0,
      has_tok   = !is.null(rv$tokens)     && nrow(rv$tokens)     > 0,
      app_ver   = "AcusTera v2"
    ),
    features   = rv$features,
    transcript = rv$transcript,
    overlaps   = rv$overlaps,
    sess_meta  = rv$sess_meta,
    spk_meta   = rv$spk_meta,
    tokens     = rv$tokens
  )
  saveRDS(payload, path, compress = "xz")
  path
}

acustera_session_list <- function() {
  d <- acustera_sessions_dir()
  files <- list.files(d, pattern = "\\.rds$", full.names = TRUE)
  if (length(files) == 0) return(data.frame(
    file = character(), label = character(), saved_at = as.POSIXct(character()),
    pairs = character(), n_pairs = integer(), n_sess = integer(),
    n_seg = integer(), has_trans = logical(), has_tok = logical(),
    stringsAsFactors = FALSE))
  rows <- lapply(files, function(f) {
    m <- tryCatch(readRDS(f)$meta, error = function(e) NULL)
    if (is.null(m)) return(NULL)
    data.frame(
      file      = f,
      label     = m$label      %||% basename(f),
      saved_at  = m$saved_at   %||% file.info(f)$mtime,
      pairs     = paste(m$pairs %||% "", collapse = ", "),
      n_pairs   = as.integer(m$n_pairs %||% NA),
      n_sess    = as.integer(m$n_sess  %||% NA),
      n_seg     = as.integer(m$n_seg   %||% NA),
      has_trans = isTRUE(m$has_trans),
      has_tok   = isTRUE(m$has_tok),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows[!vapply(rows, is.null, logical(1))]) |>
    (\(x) x[order(x$saved_at, decreasing = TRUE), , drop = FALSE])()
}

acustera_session_load <- function(paths) {
  if (length(paths) == 0) stop("Select at least one analysis.")
  parts <- lapply(paths, readRDS)
  rbind_safe <- function(lst) {
    lst <- lst[!vapply(lst, is.null, logical(1))]
    if (length(lst) == 0) return(NULL)
    dplyr::bind_rows(lst)
  }
  out <- list(
    features   = rbind_safe(lapply(parts, `[[`, "features")),
    transcript = rbind_safe(lapply(parts, `[[`, "transcript")),
    overlaps   = rbind_safe(lapply(parts, `[[`, "overlaps")),
    sess_meta  = rbind_safe(lapply(parts, `[[`, "sess_meta")),
    spk_meta   = rbind_safe(lapply(parts, `[[`, "spk_meta")),
    tokens     = rbind_safe(lapply(parts, `[[`, "tokens"))
  )
  # Dedup by couple + session in metadata
  if (!is.null(out$sess_meta) && all(c("couple","therapy_num") %in% names(out$sess_meta)))
    out$sess_meta <- dplyr::distinct(out$sess_meta, couple, therapy_num, .keep_all = TRUE)
  if (!is.null(out$spk_meta) &&
      all(c("couple","therapy_num","speaker_id") %in% names(out$spk_meta)))
    out$spk_meta <- dplyr::distinct(out$spk_meta, couple, therapy_num, speaker_id, .keep_all = TRUE)
  out
}

acustera_session_delete <- function(path) {
  if (file.exists(path)) file.remove(path) else FALSE
}
