# R/fn_review.R — Diarization review platform (runs entirely on the local machine)
#
# Workflow
#   1. review_prepare(): cuts excerpts of the session audios (one per session
#      quartile, at a random position) and writes, for each one, a WAV file and
#      a TextGrid with empty tiers man / woman / therapist (and, optionally, a
#      "words" tier from the transcription, which helps to place boundaries but
#      carries no speaker information). The automatic diarization is NOT shown
#      to the reviewer, so the reference is not biased towards any engine.
#   2. The reviewer annotates each excerpt in Praat: every stretch where a
#      person speaks is marked in that person's tier with any label (e.g. "x");
#      overlapping speech is marked in both tiers. The TextGrid is saved in place.
#   3. review_evaluate(): compares every available automatic output with the
#      reference: diarization error rate (DER) with optimal speaker mapping
#      (NIST definition: missed speech + false alarm + speaker confusion over
#      reference speech, with a forgiveness collar around reference boundaries)
#      and, for role-labelled outputs, the role attribution error (the same
#      measure without mapping, i.e. roles must match by name).

REVIEW_ROLES <- c("man", "woman", "therapist")
REVIEW_AUDIO_EXT <- "\\.(wav|mp3|flac|m4a|ogg)$"

.media_duration <- function(path) {
  out <- tryCatch(system2("ffprobe", c("-v", "error", "-show_entries", "format=duration",
                                       "-of", "csv=p=0", shQuote(path)), stdout = TRUE, stderr = FALSE),
                  error = function(e) NA_character_)
  suppressWarnings(as.numeric(trimws(out[1])))
}

# ── 1. Excerpts ─────────────────────────────────────────────────────────────
review_prepare <- function(audio_paths, out_dir, n_per_session = 2, len_min = 5,
                           seed = 2026, words_dir = NULL, log = message) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  man_f <- file.path(out_dir, "review_manifest.csv")
  old <- if (file.exists(man_f)) utils::read.csv(man_f, stringsAsFactors = FALSE) else NULL
  set.seed(seed)
  L <- len_min * 60
  rows <- list()
  for (a in audio_paths) {
    stem <- tools::file_path_sans_ext(basename(a))
    dur  <- .media_duration(a)
    if (is.na(dur) || dur < L) { log("  ", stem, ": shorter than the excerpt; skipped"); next }
    for (k in seq_len(n_per_session)) {
      q  <- (k - 1) %% 4
      lo <- dur * q / 4; hi <- max(lo, dur * (q + 1) / 4 - L)
      st <- round(stats::runif(1, lo, hi), 1)
      id <- sprintf("%s__x%02d", stem, k)
      wav <- file.path(out_dir, paste0(id, ".wav"))
      tg  <- file.path(out_dir, paste0(id, ".TextGrid"))
      if (!is.null(old) && id %in% old$id) {            # never overwrite an annotation
        log("  ", id, ": already exists; kept"); next
      }
      system2("ffmpeg", c("-y", "-loglevel", "error", "-ss", sprintf("%.3f", st), "-t", sprintf("%.3f", L),
                          "-i", shQuote(a), "-ac", "1", "-ar", "16000", shQuote(wav)),
              stdout = FALSE, stderr = FALSE)
      if (!file.exists(wav)) { log("  ", id, ": audio could not be cut"); next }
      tiers <- lapply(REVIEW_ROLES, \(r) list(name = r,
                      intervals = data.frame(xmin = double(), xmax = double(), text = character())))
      wf <- if (!is.null(words_dir) && nzchar(words_dir))
              find_precomputed(words_dir, stem, "asr") else NULL
      if (!is.null(wf)) {
        w <- tryCatch(read_words_file(wf)$words, error = function(e) NULL)
        if (!is.null(w) && nrow(w) > 0) {
          w <- w[w$w_start >= st & w$w_end <= st + L, , drop = FALSE]
          w <- w[order(w$w_start), , drop = FALSE]
          if (nrow(w) > 0) {
            xs <- w$w_start - st; xe <- w$w_end - st
            if (nrow(w) > 1) xe[-nrow(w)] <- pmin(xe[-nrow(w)], xs[-1])   # no overlaps in a tier
            keep <- xe > xs + 1e-3
            tiers[[4]] <- list(name = "words", intervals = data.frame(
              xmin = xs[keep], xmax = xe[keep], text = trimws(w$word[keep]), stringsAsFactors = FALSE))
          }
        }
      }
      write_textgrid(tiers, tg, xmin = 0, xmax = L)
      rows[[length(rows) + 1]] <- data.frame(id = id, stem = stem, audio = a, start = st, end = st + L,
                                             wav = basename(wav), textgrid = basename(tg),
                                             created = format(Sys.time(), "%Y-%m-%d %H:%M"),
                                             stringsAsFactors = FALSE)
      log("  ", id, sprintf(": %.0f–%.0f s", st, st + L))
    }
  }
  man <- rbind(old, if (length(rows)) do.call(rbind, rows))
  if (!is.null(man)) utils::write.csv(man, man_f, row.names = FALSE)
  invisible(man)
}

# Reference segments of an excerpt (relative times) — role tiers only
review_reference <- function(tg_path) {
  d <- tryCatch(read_textgrid(tg_path), error = function(e) NULL)
  if (is.null(d) || nrow(d) == 0) return(tibble::tibble(start = double(), end = double(), speaker = character()))
  d <- d[d$speaker %in% REVIEW_ROLES, c("start", "end", "speaker"), drop = FALSE]
  tibble::as_tibble(d)
}

review_status <- function(out_dir) {
  man_f <- file.path(out_dir, "review_manifest.csv")
  if (!file.exists(man_f)) return(NULL)
  man <- utils::read.csv(man_f, stringsAsFactors = FALSE)
  st <- lapply(seq_len(nrow(man)), function(i) {
    r <- review_reference(file.path(out_dir, man$textgrid[i]))
    data.frame(intervals = nrow(r), speech_s = round(sum(r$end - r$start), 1),
               speakers = length(unique(r$speaker)))
  })
  man <- cbind(man, do.call(rbind, st))
  man$status <- ifelse(man$intervals > 0, "annotated", "pending")
  man
}

# ── 2. Automatic outputs to evaluate ────────────────────────────────────────
# For a session stem, looks in each folder for:
#   <stem>_<engine>.rttm, <stem>_diarizen.json   → speaker labels (mapping)
#   <stem>.TextGrid                               → AcusTera roles (no mapping)
review_hypotheses <- function(stem, dirs) {
  out <- list()
  for (d in dirs) {
    d <- path.expand(trimws(d))
    if (!nzchar(d) || !dir.exists(d)) next
    fl <- list.files(d)
    low <- tolower(fl); s <- tolower(stem)
    for (f in fl[startsWith(low, paste0(s, "_")) & grepl("\\.rttm$", low)]) {
      eng <- sub("\\.rttm$", "", substring(f, nchar(stem) + 2), ignore.case = TRUE)
      nm <- if (tolower(eng) == "diarizen") "DiariZen" else if (tolower(eng) == "pyannote") "pyannote 3.1" else eng
      if (is.null(out[[nm]])) out[[nm]] <- list(segs = read_diar_file(file.path(d, f)), roles = FALSE)
    }
    j <- fl[low == paste0(s, "_diarizen.json")]
    if (length(j) && is.null(out[["DiariZen"]]))
      out[["DiariZen"]] <- list(segs = read_diar_file(file.path(d, j[1])), roles = FALSE)
    t <- fl[low == paste0(s, ".textgrid")]
    if (length(t)) {
      g <- tryCatch(read_textgrid(file.path(d, t[1])), error = function(e) NULL)
      if (!is.null(g) && any(g$speaker %in% REVIEW_ROLES))
        out[[paste0("Roles: ", basename(d))]] <-
          list(segs = tibble::as_tibble(g[g$speaker %in% REVIEW_ROLES, c("start", "end", "speaker")]),
               roles = TRUE)
    }
  }
  out
}

# ── 3. Diarization error rate (frame-based, 10 ms) ──────────────────────────
.frames <- function(segs, labels, n, step) {
  M <- matrix(FALSE, n, length(labels), dimnames = list(NULL, labels))
  for (i in seq_len(nrow(segs))) {
    a <- max(1L, floor(segs$start[i] / step) + 1L); b <- min(n, floor(segs$end[i] / step))
    if (b >= a) M[a:b, match(segs$speaker[i], labels)] <- TRUE
  }
  M
}

# Injective ref -> hyp mapping that maximises overlap (exhaustive: few speakers)
.best_map <- function(O) {
  nr <- nrow(O); nh <- ncol(O); best <- -1; bm <- integer(nr)
  rec <- function(r, used, cur, sc) {
    if (r > nr) { if (sc > best) { best <<- sc; bm <<- cur }; return(invisible()) }
    rec(r + 1, used, c(cur, 0L), sc)
    for (h in setdiff(seq_len(nh), used)) rec(r + 1, c(used, h), c(cur, h), sc + O[r, h])
  }
  if (nr > 0 && nh > 0) rec(1, integer(0), integer(0), 0)
  bm
}

der_score <- function(ref, hyp, dur, collar = 0.25, score_overlap = TRUE,
                      map = TRUE, step = 0.01) {
  n <- ceiling(dur / step)
  rl <- sort(unique(ref$speaker)); hl <- sort(unique(hyp$speaker))
  R <- .frames(ref, rl, n, step); H <- .frames(hyp, hl, n, step)
  mask <- rep(TRUE, n)
  if (collar > 0 && nrow(ref) > 0) {
    for (b in c(ref$start, ref$end)) {
      a <- max(1L, floor((b - collar) / step) + 1L); z <- min(n, floor((b + collar) / step))
      if (z >= a) mask[a:z] <- FALSE
    }
  }
  nr <- rowSums(R); nh <- rowSums(H)
  if (!score_overlap) mask <- mask & nr <= 1
  if (map) {
    O <- crossprod(R[mask, , drop = FALSE] * 1, H[mask, , drop = FALSE] * 1)
    mp <- .best_map(O)
  } else mp <- match(rl, hl, nomatch = 0L)
  correct <- numeric(n)
  for (r in seq_along(rl)) if (mp[r] > 0) correct <- correct + (R[, r] & H[, mp[r]])
  miss <- pmax(nr - nh, 0); fa <- pmax(nh - nr, 0); conf <- pmin(nr, nh) - correct
  tot <- sum(nr[mask]) * step
  data.frame(ref_s = tot, miss_s = sum(miss[mask]) * step, fa_s = sum(fa[mask]) * step,
             conf_s = sum(conf[mask]) * step)
}

review_evaluate <- function(out_dir, hyp_dirs, collar = 0.25, score_overlap = TRUE, log = message) {
  st <- review_status(out_dir)
  if (is.null(st)) stop("No review_manifest.csv in ", out_dir)
  st <- st[st$status == "annotated", , drop = FALSE]
  if (nrow(st) == 0) stop("No annotated excerpt yet.")
  res <- list(); segs <- list()
  for (i in seq_len(nrow(st))) {
    m <- st[i, ]; L <- m$end - m$start
    ref <- review_reference(file.path(out_dir, m$textgrid))
    segs[[m$id]] <- list(Reference = ref)
    hy <- review_hypotheses(m$stem, hyp_dirs)
    if (!length(hy)) log("  ", m$id, ": no automatic output found")
    for (nm in names(hy)) {
      h <- hy[[nm]]$segs
      h <- h[h$end > m$start & h$start < m$end, , drop = FALSE]
      h$start <- pmax(h$start, m$start) - m$start; h$end <- pmin(h$end, m$end) - m$start
      h <- h[h$end > h$start, , drop = FALSE]
      segs[[m$id]][[nm]] <- h
      d <- der_score(ref, h, L, collar, score_overlap, map = TRUE)
      res[[length(res) + 1]] <- cbind(data.frame(id = m$id, stem = m$stem, system = nm,
                                                 metric = "DER (optimal mapping)"), d)
      if (isTRUE(hy[[nm]]$roles)) {
        d2 <- der_score(ref, h, L, collar, score_overlap, map = FALSE)
        res[[length(res) + 1]] <- cbind(data.frame(id = m$id, stem = m$stem, system = nm,
                                                   metric = "Role attribution error"), d2)
      }
    }
  }
  det <- do.call(rbind, res)
  if (is.null(det)) stop("No automatic output to evaluate in the folders given.")
  det$error_pct <- round(100 * (det$miss_s + det$fa_s + det$conf_s) / det$ref_s, 1)
  sm <- det |> dplyr::group_by(system, metric) |>
    dplyr::summarise(excerpts = dplyr::n(), ref_min = round(sum(ref_s) / 60, 1),
                     missed_pct = round(100 * sum(miss_s) / sum(ref_s), 1),
                     false_alarm_pct = round(100 * sum(fa_s) / sum(ref_s), 1),
                     confusion_pct = round(100 * sum(conf_s) / sum(ref_s), 1),
                     error_pct = round(100 * sum(miss_s + fa_s + conf_s) / sum(ref_s), 1),
                     .groups = "drop") |> dplyr::arrange(metric, error_pct)
  list(summary = sm, detail = det, segs = segs,
       settings = list(collar = collar, score_overlap = score_overlap))
}

review_timeline <- function(seglist) {
  d <- do.call(rbind, lapply(names(seglist), function(nm) {
    s <- seglist[[nm]]
    if (is.null(s) || nrow(s) == 0) return(NULL)
    data.frame(system = nm, start = s$start, end = s$end, speaker = as.character(s$speaker))
  }))
  if (is.null(d)) return(plotly::plotly_empty())
  d$row <- factor(paste(d$system, d$speaker, sep = " · "),
                  levels = rev(unique(paste(d$system, d$speaker, sep = " · "))))
  p <- ggplot2::ggplot(d, ggplot2::aes(x = start, xend = end, y = row, yend = row, colour = system,
                                       text = sprintf("%s · %s<br>%.2f–%.2f s", system, speaker, start, end))) +
    ggplot2::geom_segment(linewidth = 4) +
    ggplot2::labs(x = "Time in excerpt (s)", y = NULL, colour = NULL) +
    ggplot2::theme_minimal(base_size = 11) + ggplot2::theme(legend.position = "top")
  plotly::ggplotly(p, tooltip = "text")
}

# ── UI and server ───────────────────────────────────────────────────────────
review_tab_ui <- function() {
  shinydashboard::tabItem("tab_review",
    shiny::fluidRow(shinydashboard::box(
      title = "Diarization review: how it works", width = 12, status = "info", collapsible = TRUE,
      shiny::HTML(paste0(
        "<ol>",
        "<li><b>Create excerpts.</b> AcusTera cuts excerpts of each session (one per quartile, at a random position) ",
        "and writes a WAV and a TextGrid with empty tiers <code>man</code>, <code>woman</code> and <code>therapist</code>. ",
        "The automatic diarization is not shown, so the reference is not biased towards any engine. ",
        "Optionally, a <code>words</code> tier from the transcription helps to place boundaries.</li>",
        "<li><b>Annotate in Praat.</b> Open the WAV and the TextGrid together (<i>View &amp; Edit</i>). ",
        "Mark every stretch where a person speaks in that person's tier, with any label (e.g. <code>x</code>). ",
        "Mark overlapping speech in both tiers. Close a stretch at pauses longer than about 0.3 s. ",
        "Leave silence, laughter without speech and noise unlabelled. Save the TextGrid in place (same name).</li>",
        "<li><b>Evaluate.</b> AcusTera compares every automatic output it finds (RTTM or DiariZen JSON per engine, ",
        "and the role-labelled TextGrids of AcusTera) with your reference: <b>DER</b> with optimal speaker mapping ",
        "(missed speech + false alarm + speaker confusion, over reference speech) and, for role-labelled outputs, ",
        "the <b>role attribution error</b> (the same measure without mapping).</li>",
        "</ol>",
        "<p class='text-muted'>Everything runs on this computer: audio, excerpts, annotations and scores never leave it. ",
        "A collar of 0.25 s around reference boundaries is the usual convention; set it to 0 for a strict score.</p>")))),
    shiny::fluidRow(
      shinydashboard::box(title = "1 · Create review excerpts", width = 6, status = "primary",
        shiny::textInput("rev_audio_dir", "Folder with the session audios", width = "100%",
                         placeholder = "e.g. ~/proyecto/audios"),
        shiny::textInput("rev_dir", "Review folder (excerpts, TextGrids, manifest)",
                         value = file.path(getwd(), "review"), width = "100%"),
        shiny::fluidRow(
          shiny::column(4, shiny::numericInput("rev_n", "Excerpts per session", 2, 1, 8)),
          shiny::column(4, shiny::numericInput("rev_len", "Minutes per excerpt", 5, 1, 20)),
          shiny::column(4, shiny::numericInput("rev_seed", "Seed", 2026, 1, 1e6))),
        shiny::textInput("rev_words_dir", "Optional: folder with transcriptions for a 'words' tier",
                         value = "", width = "100%"),
        shiny::actionButton("rev_prepare", "Create excerpts", icon = shiny::icon("scissors"),
                            class = "btn btn-primary"),
        shiny::verbatimTextOutput("rev_prepare_log")),
      shinydashboard::box(title = "2 · Evaluate the automatic diarization", width = 6, status = "primary",
        shiny::textAreaInput("rev_hyp_dirs", "Folders with automatic outputs (one per line)",
                             value = paste(file.path(getwd(), c("transcriptions", "textgrids")), collapse = "\n"),
                             rows = 3, width = "100%"),
        shiny::fluidRow(
          shiny::column(6, shiny::numericInput("rev_collar", "Collar (s)", 0.25, 0, 1, 0.05)),
          shiny::column(6, shiny::checkboxInput("rev_overlap", "Score overlapping speech", TRUE))),
        shiny::actionButton("rev_eval", "Evaluate", icon = shiny::icon("check-double"),
                            class = "btn btn-success"),
        shiny::downloadButton("rev_download", "Download results (CSV)"),
        shiny::verbatimTextOutput("rev_eval_log"))),
    shiny::fluidRow(shinydashboard::box(title = "Review progress", width = 12, status = "warning",
      shiny::actionButton("rev_refresh", "Refresh", icon = shiny::icon("rotate")),
      DT::DTOutput("rev_status"))),
    shiny::fluidRow(
      shinydashboard::box(title = "Error by system", width = 12, status = "success", DT::DTOutput("rev_summary")),
      shinydashboard::box(title = "Error by excerpt", width = 12, status = "success",
                          collapsible = TRUE, collapsed = TRUE, DT::DTOutput("rev_detail"))),
    shiny::fluidRow(shinydashboard::box(title = "Timeline: reference vs. automatic", width = 12,
      shiny::selectInput("rev_plot_id", "Excerpt", choices = NULL),
      plotly::plotlyOutput("rev_timeline", height = "420px"))))
}

review_server <- function(input, output, session) {
  rv_rev <- shiny::reactiveValues(status = NULL, eval = NULL, plog = "", elog = "")
  refresh <- function() rv_rev$status <- review_status(path.expand(input$rev_dir))
  shiny::observeEvent(input$rev_refresh, refresh())
  shiny::observeEvent(input$sidebar_menu, if (identical(input$sidebar_menu, "tab_review")) refresh())

  shiny::observeEvent(input$rev_prepare, {
    ad <- path.expand(trimws(input$rev_audio_dir))
    if (!dir.exists(ad)) { rv_rev$plog <- "Audio folder not found."; return() }
    au <- list.files(ad, pattern = REVIEW_AUDIO_EXT, full.names = TRUE, ignore.case = TRUE)
    msgs <- character()
    lg <- function(...) msgs <<- c(msgs, paste0(...))
    shiny::withProgress(message = "Cutting excerpts...", {
      review_prepare(au, path.expand(input$rev_dir), input$rev_n, input$rev_len, input$rev_seed,
                     path.expand(trimws(input$rev_words_dir)), log = lg)
    })
    rv_rev$plog <- paste(c(sprintf("%d audio files.", length(au)), msgs), collapse = "\n")
    refresh()
  })

  shiny::observeEvent(input$rev_eval, {
    msgs <- character(); lg <- function(...) msgs <<- c(msgs, paste0(...))
    dirs <- strsplit(input$rev_hyp_dirs, "\n")[[1]]
    r <- tryCatch(review_evaluate(path.expand(input$rev_dir), dirs, input$rev_collar,
                                  isTRUE(input$rev_overlap), log = lg),
                  error = function(e) { lg("ERROR: ", conditionMessage(e)); NULL })
    rv_rev$eval <- r
    rv_rev$elog <- paste(c(if (!is.null(r)) sprintf("%d excerpts evaluated.", length(r$segs)), msgs),
                         collapse = "\n")
    if (!is.null(r)) shiny::updateSelectInput(session, "rev_plot_id", choices = names(r$segs))
  })

  output$rev_prepare_log <- shiny::renderText(rv_rev$plog)
  output$rev_eval_log    <- shiny::renderText(rv_rev$elog)
  output$rev_status <- DT::renderDT({
    s <- rv_rev$status; shiny::req(s)
    DT::datatable(s[, c("id", "start", "end", "status", "intervals", "speech_s", "speakers", "created")],
                  rownames = FALSE, options = list(pageLength = 10))
  })
  output$rev_summary <- DT::renderDT({
    shiny::req(rv_rev$eval)
    DT::datatable(rv_rev$eval$summary, rownames = FALSE, options = list(dom = "t", pageLength = 50))
  })
  output$rev_detail <- DT::renderDT({
    shiny::req(rv_rev$eval)
    d <- rv_rev$eval$detail; d[, c("ref_s", "miss_s", "fa_s", "conf_s")] <- round(d[, c("ref_s", "miss_s", "fa_s", "conf_s")], 1)
    DT::datatable(d, rownames = FALSE, options = list(pageLength = 15))
  })
  output$rev_timeline <- plotly::renderPlotly({
    shiny::req(rv_rev$eval, input$rev_plot_id)
    review_timeline(rv_rev$eval$segs[[input$rev_plot_id]])
  })
  output$rev_download <- shiny::downloadHandler(
    filename = function() sprintf("acustera_diarization_review_%s.csv", format(Sys.Date())),
    content = function(f) {
      shiny::req(rv_rev$eval)
      utils::write.csv(rv_rev$eval$detail, f, row.names = FALSE)
    })
}
