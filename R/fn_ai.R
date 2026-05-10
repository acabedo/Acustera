# R/fn_ai.R — AI client (Anthropic + OpenAI) and speaker review.
# Keys are not exposed: read from env vars or a local unversioned file.

ai_key_file <- function() file.path(getwd(), ".acustera_keys.json")

ai_load_keys <- function() {
  f <- ai_key_file()
  empty <- list(anthropic = "", openai = "", hf = "")
  if (!file.exists(f)) return(empty)
  cur <- tryCatch(jsonlite::fromJSON(f), error = function(e) empty)
  for (k in names(empty)) if (is.null(cur[[k]])) cur[[k]] <- ""
  cur
}

ai_save_keys <- function(anthropic = NULL, openai = NULL, hf = NULL) {
  cur <- ai_load_keys()
  if (!is.null(anthropic)) cur$anthropic <- anthropic
  if (!is.null(openai))    cur$openai    <- openai
  if (!is.null(hf))        cur$hf        <- hf
  jsonlite::write_json(cur, ai_key_file(), auto_unbox = TRUE)
  Sys.chmod(ai_key_file(), mode = "0600")
  invisible(TRUE)
}

ai_get_key <- function(provider) {
  k <- ai_load_keys()
  env <- switch(provider,
    anthropic = Sys.getenv("ANTHROPIC_API_KEY"),
    openai    = Sys.getenv("OPENAI_API_KEY"),
    hf        = { x <- Sys.getenv("HF_TOKEN"); if (!nzchar(x)) Sys.getenv("HUGGINGFACE_TOKEN") else x },
    "")
  if (nzchar(env)) return(env)
  k[[provider]] %||% ""
}

# Calls an LLM. provider: "anthropic" | "openai".
# system: system prompt. user: user message. model: optional.
# Returns list(ok, text, raw, error)
ai_call <- function(provider = c("anthropic","openai"),
                    system = "", user = "",
                    model = NULL, max_tokens = 2000,
                    temperature = 0.2, timeout = 90) {
  provider <- match.arg(provider)
  if (!requireNamespace("httr",     quietly = TRUE) ||
      !requireNamespace("jsonlite", quietly = TRUE))
    return(list(ok = FALSE, error = "Install httr and jsonlite."))

  key <- ai_get_key(provider)
  if (!nzchar(key))
    return(list(ok = FALSE, error = paste0("Missing API key for ", provider,
                                           ". Add it in Settings.")))

  if (provider == "anthropic") {
    model <- model %||% "claude-sonnet-4-6"
    body <- list(model = model, max_tokens = max_tokens,
                 temperature = temperature, system = system,
                 messages = list(list(role = "user", content = user)))
    r <- tryCatch(
      httr::POST("https://api.anthropic.com/v1/messages",
                 httr::add_headers(`x-api-key` = key,
                                   `anthropic-version` = "2023-06-01",
                                   `content-type` = "application/json"),
                 body = jsonlite::toJSON(body, auto_unbox = TRUE),
                 httr::timeout(timeout)),
      error = function(e) e)
    if (inherits(r, "error"))
      return(list(ok = FALSE, error = conditionMessage(r)))
    if (httr::status_code(r) >= 400)
      return(list(ok = FALSE, error = httr::content(r, "text",
                                                    encoding = "UTF-8")))
    j <- httr::content(r, as = "parsed", type = "application/json")
    txt <- tryCatch(j$content[[1]]$text, error = function(e) "")
    return(list(ok = TRUE, text = txt %||% "", raw = j))
  }

  # OpenAI
  model <- model %||% "gpt-4o-mini"
  body <- list(model = model, max_tokens = max_tokens,
               temperature = temperature,
               messages = list(
                 list(role = "system", content = system),
                 list(role = "user",   content = user)))
  r <- tryCatch(
    httr::POST("https://api.openai.com/v1/chat/completions",
               httr::add_headers(Authorization = paste("Bearer", key),
                                 `content-type` = "application/json"),
               body = jsonlite::toJSON(body, auto_unbox = TRUE),
               httr::timeout(timeout)),
    error = function(e) e)
  if (inherits(r, "error"))
    return(list(ok = FALSE, error = conditionMessage(r)))
  if (httr::status_code(r) >= 400)
    return(list(ok = FALSE, error = httr::content(r, "text",
                                                  encoding = "UTF-8")))
  j <- httr::content(r, as = "parsed", type = "application/json")
  txt <- tryCatch(j$choices[[1]]$message$content, error = function(e) "")
  list(ok = TRUE, text = txt %||% "", raw = j)
}

# ────────────────────────────────────────────────────────────
# Heuristic speaker attribution review (without AI)
# Returns columns: flag, reason, heur_confidence
# ────────────────────────────────────────────────────────────
speaker_heuristic_review <- function(features, transcript = NULL) {
  if (is.null(features) || nrow(features) == 0) return(NULL)
  df <- features
  if (!"text" %in% names(df) && !is.null(transcript))
    df <- dplyr::left_join(df, transcript |>
      dplyr::select(dplyr::any_of(c("couple","therapy_num","seg_start","seg_end","text"))),
      by = intersect(names(df),
        c("couple","therapy_num","seg_start","seg_end")))
  if (!"text" %in% names(df)) df$text <- NA_character_

  # Medianas F0 por rol esperado
  med <- df |>
    dplyr::filter(speaker %in% c("man","woman","therapist")) |>
    dplyr::group_by(speaker) |>
    dplyr::summarise(f0_med = median(f0_mean, na.rm = TRUE),
                     f0_sd  = sd(f0_mean, na.rm = TRUE), .groups = "drop")

  pat_therapist <- "(?i)\\b(¿cómo te sientes|entiendo que|lo que escucho|permitidme|quisiera preguntar|quieres contarme|qué sientes cuando)\\b"

  df$bandera   <- FALSE
  df$motivo    <- ""
  df$confianza_heur <- 1.0

  for (i in seq_len(nrow(df))) {
    motivos <- character(0); conf <- 1.0
    h <- df$speaker[i]; f0 <- df$f0_mean[i]
    if (h %in% med$speaker && !is.na(f0)) {
      ref <- med[med$speaker == h, ]
      if (!is.na(ref$f0_sd) && ref$f0_sd > 0 &&
          abs(f0 - ref$f0_med) > 2.5 * ref$f0_sd) {
        motivos <- c(motivos, sprintf("F0=%.0fHz atypical for %s (med=%.0f)",
                                      f0, h, ref$f0_med))
        conf <- conf * 0.6
      }
    }
    txt <- df$text[i]
    if (!is.na(txt) && h != "therapist" && grepl(pat_therapist, txt)) {
      motivos <- c(motivos, "Typical therapist lexicon")
      conf <- conf * 0.5
    }
    if (!is.na(txt) && h == "therapist" && nchar(trimws(txt)) < 4) {
      motivos <- c(motivos, "Very short turn for therapist")
      conf <- conf * 0.8
    }
    if (length(motivos) > 0) {
      df$bandera[i] <- TRUE
      df$motivo[i]  <- paste(motivos, collapse = "; ")
      df$confianza_heur[i] <- round(conf, 2)
    }
  }
  df |> dplyr::filter(bandera) |>
    dplyr::select(dplyr::any_of(c("couple","therapy_num","seg_start","seg_end",
                                  "speaker","text","f0_mean","motivo",
                                  "confianza_heur")))
}

# ────────────────────────────────────────────────────────────
# LLM review: sends only heuristic flags in batches.
# Returns df with: sugerencia, confianza_ia, razonamiento
# ────────────────────────────────────────────────────────────
speaker_llm_review <- function(flagged_df, full_features, provider = "anthropic",
                               ctx_before = 2, ctx_after = 2,
                               batch_size = 10, progress = NULL) {
  if (is.null(flagged_df) || nrow(flagged_df) == 0) return(flagged_df)
  out <- flagged_df
  out$sugerencia    <- NA_character_
  out$confianza_ia  <- NA_real_
  out$razonamiento  <- NA_character_

  build_prompt <- function(row) {
    ctx <- NULL
    if (!is.null(full_features) && "text" %in% names(full_features)) {
      sub <- full_features |>
        dplyr::filter(couple == row$couple, therapy_num == row$therapy_num) |>
        dplyr::arrange(seg_start)
      idx <- which(sub$seg_start == row$seg_start)[1]
      if (!is.na(idx)) {
        rng <- max(1, idx - ctx_before):min(nrow(sub), idx + ctx_after)
        ctx <- sub[rng, c("speaker","text")]
      }
    }
    ctx_txt <- if (!is.null(ctx))
      paste(sprintf("[%s] %s", ctx$speaker,
                    substr(ctx$text %||% "", 1, 180)), collapse = "\n")
      else "(no context)"
    sprintf(
"Analyze whether the speaker attribution is correct.
Current label: %s
Turn (transcribed text): %s
Reason for suspicion: %s
Conversational context (before/after):
%s

Return ONLY a valid JSON with:
{\"sugerencia\":\"man|woman|therapist|mantener\",\"confianza\":0-1,\"razonamiento\":\"<1 sentence>\"}",
      row$speaker %||% "?", substr(row$text %||% "", 1, 400),
      row$motivo %||% "", ctx_txt)
  }

  parse_json <- function(txt) {
    m <- regmatches(txt, regexpr("\\{[^}]*\\}", txt))
    if (length(m) == 0) return(list(sugerencia=NA, confianza=NA, razonamiento=txt))
    tryCatch(jsonlite::fromJSON(m),
             error = function(e) list(sugerencia=NA, confianza=NA, razonamiento=txt))
  }

  sys <- "You are a linguistic annotator expert in couple therapy diarization. Respond only with JSON."
  for (i in seq_len(nrow(out))) {
    if (!is.null(progress)) progress(i / nrow(out),
      sprintf("Reviewing row %d/%d...", i, nrow(out)))
    r <- ai_call(provider, system = sys, user = build_prompt(out[i, , drop=FALSE]),
                 max_tokens = 200, temperature = 0.1)
    if (!r$ok) {
      out$razonamiento[i] <- paste("Error:", substr(r$error, 1, 200))  # keep "Error:" as technical label
      next
    }
    p <- parse_json(r$text)
    out$sugerencia[i]    <- p$sugerencia    %||% NA_character_
    out$confianza_ia[i]  <- suppressWarnings(as.numeric(p$confianza))
    out$razonamiento[i]  <- p$razonamiento  %||% ""
  }
  out
}

# ────────────────────────────────────────────────────────────
# Full LLM review: reads ALL segments with conversational
# context (±k turns) and reviews each attribution.
# Processed in batches of N turns to reduce number of calls.
# ────────────────────────────────────────────────────────────
speaker_llm_review_full <- function(features, provider = "anthropic",
                                    ctx = 1, batch_size = 8,
                                    confidence_threshold = 0.7,
                                    progress = NULL,
                                    only_changes = TRUE) {
  if (is.null(features) || nrow(features) == 0) return(NULL)
  df <- features
  if (!"text" %in% names(df)) df$text <- NA_character_
  df <- df |>
    dplyr::filter(!is.na(text), nchar(trimws(text)) > 0) |>
    dplyr::arrange(couple, therapy_num, seg_start) |>
    dplyr::mutate(.row = dplyr::row_number())

  out <- df
  out$sugerencia   <- NA_character_
  out$confianza_ia <- NA_real_
  out$razonamiento <- NA_character_

  sys_msg <- paste(
    "You are a linguistic annotator specialised in couple therapy diarization.",
    "Given a batch of consecutive turns with the current speaker label",
    "(man|woman|therapist) and their transcription, review whether the attribution is correct",
    "using as evidence: lexicon, register, pronouns, question type,",
    "coherence with the conversational thread and adjacent turns.",
    "Respond ONLY with a JSON array with one entry per turn in the batch:",
    '[{"idx":<index>,"sugerencia":"man|woman|therapist|mantener","confianza":0-1,"razonamiento":"<1 sentence>"}, ...]')

  idxs <- split(df$.row, ceiling(seq_along(df$.row) / batch_size))
  n_batches <- length(idxs)
  for (b in seq_along(idxs)) {
    if (!is.null(progress)) progress(b / n_batches,
      sprintf("Batch %d/%d (%d turns)...", b, n_batches, length(idxs[[b]])))
    rng <- idxs[[b]]
    # Include ctx turns before and after as context (without requesting review)
    all_rng <- unique(c(
      max(1, min(rng) - ctx):(min(rng) - 1),
      rng,
      (max(rng) + 1):min(nrow(df), max(rng) + ctx)))
    all_rng <- all_rng[all_rng >= 1 & all_rng <= nrow(df)]
    slice <- df[all_rng, , drop = FALSE]
    body <- paste(vapply(seq_len(nrow(slice)), function(k) {
      marker <- if (slice$.row[k] %in% rng) "[REVISAR]" else "[context]"
      sprintf("%s idx=%d couple=%s S%s speaker=%s: %s",
              marker, slice$.row[k], slice$couple[k],
              slice$therapy_num[k], slice$speaker[k],
              substr(slice$text[k], 1, 260))
    }, character(1)), collapse = "\n")
    user_msg <- paste0(
      "Review ONLY the turns marked [REVISAR]. ",
      "Use the [context] turns to understand the thread. ",
      "Respond with a JSON array as specified.\n\n", body)
    r <- ai_call(provider, system = sys_msg, user = user_msg,
                 max_tokens = 1200, temperature = 0.1)
    if (!r$ok) next
    txt <- r$text
    m <- regmatches(txt, regexpr("\\[.*\\]", txt, perl = TRUE))
    if (length(m) == 0) next
    parsed <- tryCatch(jsonlite::fromJSON(m, simplifyVector = TRUE),
                       error = function(e) NULL)
    if (is.null(parsed) || !"idx" %in% names(parsed)) next
    for (k in seq_len(nrow(parsed))) {
      ix <- parsed$idx[k]
      pos <- which(out$.row == ix)
      if (length(pos) == 0) next
      out$sugerencia[pos]   <- parsed$sugerencia[k]   %||% NA_character_
      out$confianza_ia[pos] <- suppressWarnings(as.numeric(parsed$confianza[k]))
      out$razonamiento[pos] <- parsed$razonamiento[k] %||% ""
    }
  }

  sel_cols <- intersect(c("couple","therapy_num","seg_start","seg_end",
                          "speaker","text","sugerencia","confianza_ia",
                          "razonamiento"), names(out))
  res <- out[, sel_cols, drop = FALSE]
  if (only_changes) {
    res <- res[!is.na(res$sugerencia) &
               res$sugerencia != "mantener" &
               res$sugerencia != res$speaker &
               (is.na(res$confianza_ia) | res$confianza_ia >= confidence_threshold), ,
               drop = FALSE]
  }
  res
}
