# R/fn_textgrid.R — Minimal Praat TextGrid reader and writer.
# IntervalTier only (no PointTier). Text format (not binary).

# Label written on speech intervals that have no transcription, so that
# they are not confused with the empty (silent) intervals Praat requires.
SPEECH_MARK <- "<speech>"

# ── Writer ─────────────────────────────────────────────────────────────
# tiers: list(list(name = "man", intervals = tibble(xmin, xmax, text)), ...)
# xmin/xmax in seconds. text can be "".
write_textgrid <- function(tiers, path, xmin = 0, xmax = NULL) {
  if (is.null(xmax)) {
    xmax <- max(vapply(tiers, function(t)
      if (nrow(t$intervals) > 0) max(t$intervals$xmax) else 0,
      numeric(1)), na.rm = TRUE)
  }
  esc <- function(s) gsub('"', '""', s %||% "")
  L <- c(
    'File type = "ooTextFile"',
    'Object class = "TextGrid"',
    "",
    sprintf("xmin = %.6f", xmin),
    sprintf("xmax = %.6f", xmax),
    "tiers? <exists>",
    sprintf("size = %d", length(tiers)),
    "item []:"
  )
  for (i in seq_along(tiers)) {
    t <- tiers[[i]]
    iv <- t$intervals
    # Fill with empty intervals between segments to meet
    # Praat's requirement (continuous coverage [xmin,xmax]).
    if (is.null(iv) || nrow(iv) == 0) {
      iv2 <- data.frame(xmin = xmin, xmax = xmax, text = "")
    } else {
      iv <- iv[order(iv$xmin), , drop = FALSE]
      rows <- list()
      cur <- xmin
      for (k in seq_len(nrow(iv))) {
        if (iv$xmin[k] > cur + 1e-6)
          rows[[length(rows)+1]] <- data.frame(xmin = cur,
            xmax = iv$xmin[k], text = "")
        rows[[length(rows)+1]] <- data.frame(xmin = iv$xmin[k],
          xmax = iv$xmax[k], text = iv$text[k] %||% "")
        cur <- iv$xmax[k]
      }
      if (cur < xmax - 1e-6)
        rows[[length(rows)+1]] <- data.frame(xmin = cur,
          xmax = xmax, text = "")
      iv2 <- do.call(rbind, rows)
    }
    L <- c(L,
      sprintf("    item [%d]:", i),
      '        class = "IntervalTier"',
      sprintf('        name = "%s"', esc(t$name)),
      sprintf("        xmin = %.6f", xmin),
      sprintf("        xmax = %.6f", xmax),
      sprintf("        intervals: size = %d", nrow(iv2))
    )
    for (k in seq_len(nrow(iv2))) {
      L <- c(L,
        sprintf("        intervals [%d]:", k),
        sprintf("            xmin = %.6f", iv2$xmin[k]),
        sprintf("            xmax = %.6f", iv2$xmax[k]),
        sprintf('            text = "%s"', esc(iv2$text[k]))
      )
    }
  }
  writeLines(L, con = path, useBytes = TRUE)
  invisible(path)
}

# Convenience: from a tibble of segments (start, end, speaker, text)
# produces a TextGrid with one IntervalTier per speaker.
segs_to_textgrid <- function(segs, path, text_col = "text",
                              duracion_total = NULL) {
  if (is.null(segs) || nrow(segs) == 0) {
    write_textgrid(list(list(name = "vacio",
                             intervals = data.frame(xmin=double(),
                                xmax=double(), text=character()))),
                    path, xmin = 0, xmax = duracion_total %||% 1)
    return(invisible(path))
  }
  speakers <- unique(segs$speaker)
  speakers <- speakers[!is.na(speakers)]
  tiers <- lapply(speakers, function(h) {
    sub <- segs[segs$speaker == h, , drop = FALSE]
    tx  <- if (text_col %in% names(sub)) sub[[text_col]] else ""
    list(name = h,
         intervals = data.frame(xmin = sub$start %||% sub$seg_start,
                                xmax = sub$end   %||% sub$seg_end,
                                text = ifelse(is.na(tx) | !nzchar(trimws(tx)),
                                              SPEECH_MARK, tx),
                                stringsAsFactors = FALSE))
  })
  xmax <- duracion_total %||%
    max(c(segs$end %||% segs$seg_end, 0), na.rm = TRUE)
  write_textgrid(tiers, path, xmin = 0, xmax = xmax)
}

# ── Reader ───────────────────────────────────────────────────────────────
# Returns a tibble with columns: start, end, duration, speaker, speaker, text.
# speaker = tier name as-is. speaker = normalised:
#   "man"/"woman"/"therapist" if the tier name contains those
#   substrings (case-insensitive); otherwise, speaker as-is.
read_textgrid <- function(path) {
  raw <- readLines(path, warn = FALSE, encoding = "UTF-8")
  # Detect format: long (header "File type") vs short
  is_short <- any(grepl('^"ooTextFile"', raw))
  rows <- list()
  if (is_short) {
    # Short format: bare values, line by line
    # header: "ooTextFile" / "TextGrid" / xmin / xmax / <exists> / size
    # per tier: "IntervalTier" / name / xmin / xmax / size / (xmin xmax text)*
    i <- 1
    while (i <= length(raw)) {
      if (grepl('^"IntervalTier"', raw[i])) {
        name <- gsub('^"|"$', "", trimws(raw[i+1]))
        n    <- as.integer(trimws(raw[i+4]))
        i <- i + 5
        for (k in seq_len(n)) {
          xmin <- as.numeric(trimws(raw[i]))
          xmax <- as.numeric(trimws(raw[i+1]))
          text <- gsub('^"|"$', "", trimws(raw[i+2]))
          rows[[length(rows)+1]] <- data.frame(
            start = xmin, end = xmax, speaker = name,
            text = text, stringsAsFactors = FALSE)
          i <- i + 3
        }
      } else i <- i + 1
    }
  } else {
    # Long format: "class = ..." / "name = ..." / etc.
    tier_name <- NA_character_
    i <- 1
    while (i <= length(raw)) {
      ln <- raw[i]
      if (grepl('class\\s*=\\s*"IntervalTier"', ln)) {
        # next relevant line: name = "..."
        j <- i + 1
        while (j <= length(raw) && !grepl('name\\s*=', raw[j])) j <- j + 1
        if (j <= length(raw))
          tier_name <- sub('.*name\\s*=\\s*"(.*)".*', "\\1", raw[j])
        i <- j
      } else if (grepl('intervals\\s*\\[\\d+\\]', ln)) {
        xmin <- NA_real_; xmax <- NA_real_; text <- ""
        for (k in 1:6) {
          if (i + k > length(raw)) break
          l2 <- raw[i + k]
          if (grepl('xmin\\s*=', l2)) xmin <- as.numeric(sub('.*=\\s*', "", l2))
          else if (grepl('xmax\\s*=', l2)) xmax <- as.numeric(sub('.*=\\s*', "", l2))
          else if (grepl('text\\s*=', l2)) {
            text <- sub('.*text\\s*=\\s*"(.*)".*', "\\1", l2)
            break     # stop here: the next lines belong to the next interval
          }
        }
        if (!is.na(xmin) && !is.na(xmax))
          rows[[length(rows)+1]] <- data.frame(
            start = xmin, end = xmax, speaker = tier_name,
            text = text, stringsAsFactors = FALSE)
        i <- i + 4
      } else i <- i + 1
    }
  }
  if (length(rows) == 0) return(NULL)
  df <- do.call(rbind, rows)
  df <- df[!is.na(df$start) & !is.na(df$end) & df$end > df$start, , drop = FALSE]
  # Discard empty intervals (silences): a speech interval must have a label.
  # AcusTera writes "<speech>" for speech intervals without transcription.
  df <- df[nzchar(trimws(df$text)), , drop = FALSE]
  df$text[trimws(df$text) == SPEECH_MARK] <- ""
  df$duration <- df$end - df$start
  # Map tier → canonical speaker
  lc <- tolower(df$speaker)
  df$speaker <- dplyr::case_when(
    grepl("woman|female|fem",        lc) ~ "woman",
    grepl("^man$|male|varon|varón",  lc) ~ "man",
    grepl("terap|therap|psico",      lc) ~ "therapist",
    TRUE                                 ~ df$speaker
  )
  tibble::as_tibble(df)
}
