# R/fn_examples2.R — New "Prosodic examples" section v2
# Design: card-style browser, rich filters, selection for report,
# inline audio, F0 sparkline for context, tooltip with acoustic values.

# Builds a rich examples dataframe without filtering by "moment".
# Allows filtering by: parameter, direction, percentile, quartile,
# session range, speaker, couple, required POS (if tokens available).
build_examples_v2 <- function(df, params = "f0_mean",
                              direction = c("both","high","low"),
                              percentil = 0.15,
                              quantiles = NULL, sesiones_range = NULL,
                              speakers = c("man","woman","therapist"),
                              couples = NULL, tokens_df = NULL,
                              upos_required = NULL,
                              min_chars = 3) {
  direction <- match.arg(direction)
  if (is.null(df) || nrow(df) == 0) return(tibble::tibble())
  req_cols <- c("text","speaker","therapy_num","couple","seg_start","seg_end")
  if (!all(req_cols %in% names(df))) return(tibble::tibble())
  params <- intersect(params, names(df))
  if (length(params) == 0) params <- "f0_mean"

  out <- df |>
    dplyr::filter(speaker %in% speakers,
                  !is.na(text), nchar(text) >= min_chars)
  if (!is.null(couples) && length(couples) > 0 && !"All" %in% couples)
    out <- dplyr::filter(out, couple %in% couples)
  if (!is.null(quantiles) && "quantile" %in% names(out) && length(quantiles) > 0)
    out <- dplyr::filter(out, as.character(quantile) %in% as.character(quantiles))
  if (!is.null(sesiones_range) && length(sesiones_range) == 2)
    out <- dplyr::filter(out, therapy_num >= sesiones_range[1],
                              therapy_num <= sesiones_range[2])
  if (nrow(out) == 0) return(tibble::tibble())

  # Per parameter: z-score within speaker → global extremes.
  # If multiple params, take the maximum |z| per row.
  zmat <- sapply(params, function(p) {
    x <- out[[p]]
    ave(x, out$speaker, FUN = function(v) {
      m <- mean(v, na.rm = TRUE); s <- sd(v, na.rm = TRUE)
      if (is.na(s) || s == 0) return(rep(0, length(v)))
      (v - m) / s
    })
  })
  if (is.null(dim(zmat))) zmat <- matrix(zmat, ncol = 1)
  colnames(zmat) <- params
  out$.zmax <- apply(abs(zmat), 1, max, na.rm = TRUE)
  out$.zsign <- sign(apply(zmat, 1, function(r) r[which.max(abs(r))]))
  out$.param_pico <- params[apply(abs(zmat), 1, which.max)]
  # Umbral por percentil
  thr <- quantile(out$.zmax, 1 - percentil, na.rm = TRUE)
  out <- dplyr::filter(out, .zmax >= thr)

  if (direction == "high") out <- dplyr::filter(out, .zsign >= 0)
  if (direction == "low") out <- dplyr::filter(out, .zsign < 0)

  # POS filter: the row must contain at least one word with upos in the list
  if (!is.null(tokens_df) && !is.null(upos_required) && length(upos_required) > 0) {
    ok_ids <- tokens_df |>
      dplyr::filter(upos %in% upos_required) |>
      dplyr::inner_join(
        out |> dplyr::mutate(.key = paste(couple, therapy_num, seg_start)),
        by = intersect(c("couple","therapy_num","seg_start"), names(tokens_df))
      ) |> dplyr::pull(.key) |> unique()
    out <- out |>
      dplyr::mutate(.key = paste(couple, therapy_num, seg_start)) |>
      dplyr::filter(.key %in% ok_ids) |>
      dplyr::select(-.key)
  }

  out |>
    dplyr::arrange(dplyr::desc(.zmax)) |>
    dplyr::mutate(
      .id = paste(couple, therapy_num, round(seg_start, 2), sep = "|"),
      direccion = ifelse(.zsign >= 0, "↑", "↓"),
      z_abs     = round(.zmax, 2),
      f0_mean   = round(f0_mean, 1),
      intensity_mean = round(intensity_mean, 1),
      seg_start = round(seg_start, 2),
      seg_end   = round(seg_end, 2)
    )
}

# Renders an HTML card for an example.
# sel_ids: vector of already-selected ids to show state.
.speaker_color <- function(h) {
  switch(as.character(h),
         man = "#2C7BB6", woman = "#c0392b",
         therapist = "#27ae60", "#888")
}

render_example_card <- function(row, sel_ids = character(0),
                                audio_url = NULL) {
  id <- row$.id
  sel <- id %in% sel_ids
  col <- .speaker_color(row$speaker)
  btn_sel <- sprintf(
    '<button class="ej2-sel-btn" data-id="%s" style="background:%s;color:#fff;border:0;border-radius:3px;padding:3px 8px;font-size:11px;cursor:pointer">%s</button>',
    htmltools::htmlEscape(id),
    if (sel) "#27ae60" else "#555",
    if (sel) "✓ in report" else "+ mark")
  audio_tag <- if (!is.null(audio_url))
    sprintf('<audio controls preload="none" src="%s#t=%.2f,%.2f" style="width:100%%;height:28px;margin-top:6px"></audio>',
            audio_url, row$seg_start, row$seg_end) else ""
  sprintf('
<div class="ej2-card" data-id="%s" style="border-left:3px solid %s;background:#fff;border-radius:4px;padding:10px 12px;margin-bottom:10px;box-shadow:0 1px 2px rgba(0,0,0,.04)">
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#666;margin-bottom:4px">
    <div>
      <b style="color:%s;text-transform:uppercase;letter-spacing:.04em">%s</b>
      · Couple %s · S%s · [%.2fs – %.2fs]
      · <span title="%s (z=%.2f)">%s %s</span>
      · F0 <b>%.1f</b>Hz · Int <b>%.1f</b>dB
    </div>
    <div>%s</div>
  </div>
  <div style="font-family:\'IBM Plex Sans\',sans-serif;font-size:13px;color:#222;line-height:1.5">%s</div>
  %s
</div>',
    htmltools::htmlEscape(id), col, col,
    row$speaker, row$couple, row$therapy_num,
    row$seg_start, row$seg_end,
    row$.param_pico, row$z_abs, row$direccion, row$.param_pico,
    row$f0_mean, row$intensity_mean,
    btn_sel,
    htmltools::htmlEscape(row$text),
    audio_tag)
}

# mm:ss format.
.fmt_mmss <- function(sec) {
  if (is.na(sec)) return("--:--")
  m <- floor(sec / 60); s <- round(sec - m * 60)
  if (s == 60) { m <- m + 1; s <- 0 }
  sprintf("%02d:%02d", m, s)
}

# Explains why the example is relevant (prosodic values) in prose.
.param_human <- c(f0_mean = "mean F0", f0_sd = "F0 variability",
  f0_range = "F0 range", f0_st = "F0 range (semitones)",
  intensity_mean = "mean intensity", intensity_sd = "intensity variability",
  hnr = "HNR", jitter_local = "jitter", shimmer_local = "shimmer",
  cpp = "CPP", speech_rate = "speech rate",
  zcr = "ZCR", rms = "RMS", spectral_centroid = "spectral centroid",
  mfcc1 = "MFCC-1", mfcc2 = "MFCC-2", mfcc3 = "MFCC-3")
.param_unit <- c(f0_mean = "Hz", f0_sd = "Hz", f0_range = "Hz", f0_st = "st",
  intensity_mean = "dB", intensity_sd = "dB", hnr = "dB",
  speech_rate = "voice frames/s", spectral_centroid = "Hz")

render_example_rationale <- function(row) {
  p <- row$.param_pico; z <- row$z_abs; s <- row$direccion
  nm <- .param_human[p] %||% p
  unit <- .param_unit[p] %||% ""
  val <- tryCatch(row[[p]], error = function(e) NA)
  val_txt <- if (!is.na(val)) sprintf("%.1f %s", val, unit) else "n/d"
  dir_txt <- if (identical(s, "↑")) "well above" else "well below"
  sprintf(
    "This turn is selected for presenting %s = %s, %s the speaker's mean (z = %s%.2f SD).",
    nm, val_txt, dir_txt, ifelse(identical(s,"↑"),"+","-"), z)
}

# Groups consecutive turns by the same speaker and builds a dialogue like:
#   Therapist: turn1 · turn2 · turn3
#   Man:       turnA · turnB
# ctx_df: dataframe of turns adjacent to the target turn, already sorted.
# highlight_id: id of the target turn (to highlight it).
render_conversation_context <- function(ctx_df, highlight_id = NULL) {
  if (is.null(ctx_df) || nrow(ctx_df) == 0) return("")
  # Collapse consecutive turns by the same speaker
  grp <- c(1, 1 + cumsum(diff(as.integer(factor(ctx_df$speaker))) != 0))
  ctx_df$.g <- grp
  ids <- if (!is.null(ctx_df$.id)) ctx_df$.id else
           paste(ctx_df$couple, ctx_df$therapy_num, round(ctx_df$seg_start, 2), sep = "|")
  rows <- lapply(split(seq_len(nrow(ctx_df)), ctx_df$.g), function(ix) {
    bloque <- ctx_df[ix, , drop = FALSE]
    hab <- as.character(bloque$speaker[1])
    col <- .speaker_color(hab)
    label <- switch(hab, man = "Man", woman = "Woman",
                    therapist = "Therapist", tools::toTitleCase(hab))
    partes <- vapply(seq_len(nrow(bloque)), function(k) {
      txt <- gsub("^\\s+|\\s+$", "", bloque$text[k] %||% "")
      is_hi <- !is.null(highlight_id) && ids[ix[k]] == highlight_id
      if (is_hi) sprintf('<mark style="background:#fff4c4;padding:1px 2px">%s</mark>',
                         htmltools::htmlEscape(txt))
      else htmltools::htmlEscape(txt)
    }, character(1))
    sprintf(
      '<div style="margin:2px 0;line-height:1.55;font-size:13px"><b style="color:%s;min-width:80px;display:inline-block">%s:</b> %s</div>',
      col, label, paste(partes, collapse = " / "))
  })
  paste0(
    '<div style="background:#fafbfc;border:1px solid #e4e7eb;border-radius:4px;padding:10px 12px;margin-top:8px;font-family:\'Georgia\',serif">',
    '<div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px">Conversational context</div>',
    paste(rows, collapse = ""),
    '</div>')
}

# "Conversation analysis" format designed for academic papers.
# Includes full citation, temporal location, outlier value and justification.
render_example_paper <- function(row, audio_url = NULL,
                                 sel_ids = character(0),
                                 full_df = NULL, ctx_before = 2, ctx_after = 2) {
  id  <- row$.id
  col <- .speaker_color(row$speaker)
  sel <- id %in% sel_ids
  rol <- c(man = "male couple member",
           woman  = "female couple member",
           therapist = "therapist")[as.character(row$speaker)]
  rol <- rol %||% as.character(row$speaker)
  t_ini <- .fmt_mmss(row$seg_start); t_fin <- .fmt_mmss(row$seg_end)
  cuart <- if (!is.null(row$quantile)) sprintf(" · %s", row$quantile) else ""
  cita_header <- sprintf(
    "(Couple %s, session %s, min %s–%s%s; %s)",
    row$couple, row$therapy_num, t_ini, t_fin, cuart, rol)
  extras <- c()
  if (!is.na(row$f0_mean)) extras <- c(extras, sprintf("F0 = %.1f Hz", row$f0_mean))
  if (!is.na(row$intensity_mean)) extras <- c(extras,
    sprintf("intensity = %.1f dB", row$intensity_mean))
  if (!is.null(row$speech_rate) && !is.na(row$speech_rate))
    extras <- c(extras, sprintf("rate = %.2f voice-frames/s", row$speech_rate))
  prosodia <- paste(extras, collapse = "; ")
  rationale <- render_example_rationale(row)
  audio_tag <- if (!is.null(audio_url))
    sprintf('<audio controls preload="none" src="%s#t=%.2f,%.2f" style="width:100%%;height:28px;margin-top:6px"></audio>',
            audio_url, row$seg_start, row$seg_end) else ""
  btn_sel <- sprintf(
    '<button class="ej2-sel-btn" data-id="%s" style="background:%s;color:#fff;border:0;border-radius:3px;padding:3px 8px;font-size:11px;cursor:pointer">%s</button>',
    htmltools::htmlEscape(id),
    if (sel) "#27ae60" else "#555",
    if (sel) "✓ in report" else "+ mark")
  btn_copy <- sprintf(
    '<button onclick="(function(el){var t=el.closest(\'.ej2-paper\').querySelector(\'.ej2-cite\').innerText;navigator.clipboard.writeText(t);el.innerText=\'✓ copied\';setTimeout(()=>el.innerText=\'Copy citation\',1400);})(this)" style="background:#fff;border:1px solid #bbb;border-radius:3px;padding:3px 8px;font-size:11px;cursor:pointer;margin-left:6px">Copy citation</button>')
  plain_cite <- sprintf('«%s» %s',
    gsub("[\r\n]+"," ", row$text %||% ""), cita_header)

  ctx_html <- ""
  if (!is.null(full_df) && "text" %in% names(full_df)) {
    sub <- full_df[
      as.character(full_df$couple)      == as.character(row$couple) &
      as.integer(full_df$therapy_num)   == as.integer(row$therapy_num), , drop = FALSE]
    if (nrow(sub) > 0) {
      sub <- sub[order(sub$seg_start), , drop = FALSE]
      pos <- which(abs(sub$seg_start - row$seg_start) < 0.01)[1]
      if (!is.na(pos)) {
        rng <- max(1, pos - ctx_before):min(nrow(sub), pos + ctx_after)
        ctx <- sub[rng, c("couple","therapy_num","seg_start","speaker","text"),
                   drop = FALSE]
        ctx$.id <- paste(ctx$couple, ctx$therapy_num,
                         round(ctx$seg_start, 2), sep = "|")
        ctx_html <- render_conversation_context(ctx, highlight_id = id)
      }
    }
  }
  sprintf('
<div class="ej2-card ej2-paper" data-id="%s" style="border-left:3px solid %s;background:#fff;border-radius:4px;padding:12px 14px;margin-bottom:12px;box-shadow:0 1px 2px rgba(0,0,0,.04)">
  <div class="ej2-cite" style="font-family:\'Georgia\',serif;font-size:14px;color:#222;line-height:1.55">
    <span style="color:#555">&laquo;</span><i>%s</i><span style="color:#555">&raquo;</span>
    <span style="color:#444;font-style:normal"> %s</span>
  </div>
  <div style="margin-top:6px;color:#555;font-size:12.5px;line-height:1.5">
    <b style="color:%s">%s.</b> %s Prosodic values for this turn: %s.
  </div>
  %s
  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px">
    <div style="font-size:11px;color:#888">ID <code>%s</code></div>
    <div>%s%s</div>
  </div>
  %s
  <textarea style="display:none">%s</textarea>
</div>',
    htmltools::htmlEscape(id), col,
    htmltools::htmlEscape(row$text %||% ""),
    htmltools::htmlEscape(cita_header),
    col, tools::toTitleCase(as.character(row$speaker)),
    htmltools::htmlEscape(rationale),
    htmltools::htmlEscape(prosodia %||% "n/a"),
    ctx_html,
    htmltools::htmlEscape(id), btn_sel, btn_copy, audio_tag,
    htmltools::htmlEscape(plain_cite))
}

# Builds the audio URL if the file exists in www/audios/
audio_url_for <- function(couple, therapy_num) {
  p <- file.path(getwd(), "www", "audios",
                 sprintf("%s_therapy%s.wav", couple, therapy_num))
  if (file.exists(p)) sprintf("audios/%s_therapy%s.wav", couple, therapy_num)
  else NULL
}
