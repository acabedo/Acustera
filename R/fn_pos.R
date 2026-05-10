# R/fn_pos.R — Morphosyntactic annotation (udpipe) and lexical analysis
# Lazy load: the model is only downloaded when first used.

.udpipe_model <- NULL
.udpipe_lang  <- "spanish"

udpipe_available <- function() requireNamespace("udpipe", quietly = TRUE)

# Loads (and downloads if needed) the Spanish UDPipe model.
# model_dir: folder to save the .udpipe file (default .acustera_cache/)
get_udpipe_model <- function(lang = "spanish", model_dir = NULL, log = message) {
  if (!udpipe_available()) {
    log("udpipe is not installed. Run: install.packages('udpipe')")
    return(NULL)
  }
  if (!is.null(.udpipe_model) && .udpipe_lang == lang) return(.udpipe_model)
  if (is.null(model_dir)) {
    model_dir <- file.path(getwd(), ".acustera_cache")
    if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)
  }
  existing <- list.files(model_dir, pattern = paste0("^", lang, "-.*\\.udpipe$"),
                         full.names = TRUE)
  if (length(existing) > 0) {
    m <- udpipe::udpipe_load_model(existing[1])
  } else {
    log(sprintf("Downloading UDPipe model for %s (~50MB, once)...", lang))
    dl <- udpipe::udpipe_download_model(language = lang, model_dir = model_dir)
    m  <- udpipe::udpipe_load_model(dl$file_model)
  }
  .udpipe_model <<- m
  .udpipe_lang  <<- lang
  m
}

# Annotates a data.frame with a `text` column and returns long tokens:
# couple, therapy_num, speaker, quantile, token, lemma, upos, feats, doc_id
annotate_pos <- function(features, text_col = "text", lang = "spanish",
                         log = message) {
  if (!udpipe_available()) return(NULL)
  if (!text_col %in% names(features)) return(NULL)
  rows <- features |>
    dplyr::mutate(.rowid = dplyr::row_number()) |>
    dplyr::filter(!is.na(.data[[text_col]]),
                  nchar(trimws(.data[[text_col]])) > 1)
  if (nrow(rows) == 0) return(NULL)

  mdl <- get_udpipe_model(lang, log = log)
  if (is.null(mdl)) return(NULL)

  ann <- udpipe::udpipe_annotate(mdl, x = rows[[text_col]],
                                 doc_id = as.character(rows$.rowid)) |>
         as.data.frame()
  keep_cols <- intersect(c("couple", "therapy_num", "speaker", "quantile",
                           "seg_start", "seg_end"), names(rows))
  meta <- rows[, c(".rowid", keep_cols), drop = FALSE]
  meta$.rowid <- as.character(meta$.rowid)
  ann$doc_id  <- as.character(ann$doc_id)
  dplyr::left_join(ann, meta, by = c("doc_id" = ".rowid"))
}

# POS frequency summary by speaker and time quartile
pos_summary <- function(tokens_df, by = c("speaker", "quantile")) {
  by <- intersect(by, names(tokens_df))
  if (length(by) == 0) return(NULL)
  tokens_df |>
    dplyr::filter(!is.na(upos), upos != "PUNCT") |>
    dplyr::group_by(dplyr::across(dplyr::all_of(by)), upos) |>
    dplyr::summarise(n = dplyr::n(), .groups = "drop") |>
    dplyr::group_by(dplyr::across(dplyr::all_of(by))) |>
    dplyr::mutate(pct = 100 * n / sum(n)) |>
    dplyr::ungroup()
}

# Lexical metrics by speaker (TTR, Yule's K, lexical density)
lexical_metrics <- function(tokens_df, by = "speaker") {
  by <- intersect(by, names(tokens_df))
  if (length(by) == 0) return(NULL)
  content_pos <- c("NOUN", "VERB", "ADJ", "ADV")
  tokens_df |>
    dplyr::filter(!is.na(upos), upos != "PUNCT") |>
    dplyr::group_by(dplyr::across(dplyr::all_of(by))) |>
    dplyr::summarise(
      n_tokens   = dplyr::n(),
      n_types    = dplyr::n_distinct(tolower(lemma)),
      TTR        = round(n_types / n_tokens, 3),
      dens_lex   = round(mean(upos %in% content_pos), 3),
      yules_K    = {
        f   <- as.numeric(table(tolower(lemma)))
        M1  <- sum(f); M2 <- sum(f^2)
        round(10000 * (M2 - M1) / (M1^2), 2)
      },
      .groups = "drop"
    )
}

# Top lemmas by group — for word clouds
top_lemmas <- function(tokens_df, by = c("speaker"), upos_filter = NULL,
                       n = 100, min_chars = 3) {
  by <- intersect(by, names(tokens_df))
  out <- tokens_df |>
    dplyr::filter(!is.na(lemma), nchar(lemma) >= min_chars,
                  upos != "PUNCT", !is.na(upos))
  if (!is.null(upos_filter) && length(upos_filter) > 0)
    out <- dplyr::filter(out, upos %in% upos_filter)
  out |>
    dplyr::mutate(lemma = tolower(lemma)) |>
    dplyr::group_by(dplyr::across(dplyr::all_of(by)), lemma) |>
    dplyr::summarise(freq = dplyr::n(), .groups = "drop") |>
    dplyr::group_by(dplyr::across(dplyr::all_of(by))) |>
    dplyr::slice_max(freq, n = n, with_ties = FALSE) |>
    dplyr::ungroup()
}

# Personal pronouns (clinical marker): yo / tú / nosotros
pronoun_markers <- function(tokens_df) {
  if (!"lemma" %in% names(tokens_df)) return(NULL)
  pron <- tokens_df |>
    dplyr::filter(upos == "PRON") |>
    dplyr::mutate(lemma = tolower(lemma),
                  grupo = dplyr::case_when(
                    lemma %in% c("yo","mí","mi","conmigo")     ~ "1st sing",
                    lemma %in% c("nosotros","nosotras","nos")  ~ "1st plur",
                    lemma %in% c("tú","ti","te","contigo","usted") ~ "2nd sing",
                    lemma %in% c("vosotros","vosotras","os","ustedes") ~ "2nd plur",
                    lemma %in% c("él","ella","le","lo","la")   ~ "3rd sing",
                    lemma %in% c("ellos","ellas","les","los","las") ~ "3rd plur",
                    TRUE ~ NA_character_
                  ))
  by <- intersect(c("speaker","quantile","therapy_num"), names(pron))
  pron |>
    dplyr::filter(!is.na(grupo)) |>
    dplyr::group_by(dplyr::across(dplyr::all_of(by)), grupo) |>
    dplyr::summarise(n = dplyr::n(), .groups = "drop")
}

# Extended lexical markers: pronouns, possessives, demonstratives,
# negation, intensifiers, hedges, adversatives, emotion +/-.
# Returns count + total tokens per speaker for relativisation.
lexical_markers <- function(tokens_df,
                            categorias = c("pron_1sg","pron_1pl","pron_2sg",
                                           "pron_3sg","pron_3pl","posesivos",
                                           "demostrativos","negacion",
                                           "intensificadores","atenuadores",
                                           "adversativos","emocion_neg",
                                           "emocion_pos")) {
  if (!"lemma" %in% names(tokens_df)) return(NULL)
  df <- tokens_df
  df$lemma_lc <- tolower(df$lemma %||% "")
  L <- list(
    pron_1sg       = list(lemas = c("yo","mí","mi","conmigo"), upos = "PRON"),
    pron_1pl       = list(lemas = c("nosotros","nosotras","nos"), upos = "PRON"),
    pron_2sg       = list(lemas = c("tú","ti","te","contigo","usted"), upos = "PRON"),
    pron_3sg       = list(lemas = c("él","ella","le","lo","la"), upos = "PRON"),
    pron_3pl       = list(lemas = c("ellos","ellas","les","los","las"), upos = "PRON"),
    posesivos      = list(lemas = c("mi","mío","mía","nuestro","nuestra",
                                    "tu","tuyo","tuya","su","suyo","suya"),
                          upos = c("DET","PRON")),
    demostrativos  = list(lemas = c("este","esta","esto","estos","estas",
                                    "ese","esa","eso","esos","esas",
                                    "aquel","aquella","aquello"),
                          upos = c("DET","PRON")),
    negacion       = list(lemas = c("no","nunca","jamás","nada","nadie",
                                    "ningún","ninguno","ninguna","tampoco"),
                          upos = NULL),
    intensificadores = list(lemas = c("muy","mucho","demasiado","bastante",
                                      "tan","tanto","totalmente","completamente",
                                      "absolutamente","siempre"), upos = NULL),
    atenuadores    = list(lemas = c("quizá","quizás","creo","parecer",
                                    "supongo","suponer","igual","acaso",
                                    "probablemente","posiblemente","algo"),
                          upos = NULL),
    adversativos   = list(lemas = c("pero","sino","aunque"), upos = NULL),
    emocion_neg    = list(lemas = c("odiar","enfadar","enfado","rabia","miedo",
                                    "triste","tristeza","dolor","doler","culpa",
                                    "frustración","frustrar","ansiedad","ansioso",
                                    "preocupar","preocupación","harto","cansado",
                                    "cansar"), upos = NULL),
    emocion_pos    = list(lemas = c("amor","amar","querer","feliz","felicidad",
                                    "alegría","alegre","ilusión","ilusionar",
                                    "tranquilo","calma","agradecer"), upos = NULL)
  )
  categorias <- intersect(categorias, names(L))
  rows <- lapply(categorias, function(cat) {
    spec <- L[[cat]]
    sel <- df$lemma_lc %in% spec$lemas
    if (!is.null(spec$upos)) sel <- sel & (df$upos %in% spec$upos)
    if (!any(sel)) return(NULL)
    sub <- df[sel, , drop = FALSE]
    sub$categoria <- cat
    sub
  })
  hit <- do.call(rbind, Filter(Negate(is.null), rows))
  if (is.null(hit) || nrow(hit) == 0) return(NULL)
  by  <- intersect(c("speaker","quantile","therapy_num"), names(hit))
  cnt <- hit |>
    dplyr::group_by(dplyr::across(dplyr::all_of(by)), categoria) |>
    dplyr::summarise(n = dplyr::n(), .groups = "drop")
  tot <- df |>
    dplyr::group_by(dplyr::across(dplyr::all_of(by))) |>
    dplyr::summarise(n_tot = dplyr::n(), .groups = "drop")
  dplyr::left_join(cnt, tot, by = by) |>
    dplyr::mutate(pct = 100 * n / pmax(n_tot, 1))
}

# Generates a minimal HTML word cloud (no extra dependency)
# Font size proportional to log(freq). Optional group colours.
wordcloud_html <- function(lemmas_df, group_col = NULL,
                           palette = c("#2C7BB6","#c0392b","#27ae60","#f39c12"),
                           max_size = 42, min_size = 12) {
  if (is.null(lemmas_df) || nrow(lemmas_df) == 0)
    return("<p style='color:#888'>Insufficient data.</p>")
  fmin <- min(lemmas_df$freq); fmax <- max(lemmas_df$freq)
  lemmas_df$.size <- if (fmax == fmin) (max_size + min_size)/2
                    else min_size + (max_size - min_size) *
                      (log1p(lemmas_df$freq) - log1p(fmin)) /
                      (log1p(fmax) - log1p(fmin))
  grp_colors <- NULL
  if (!is.null(group_col) && group_col %in% names(lemmas_df)) {
    grps <- unique(as.character(lemmas_df[[group_col]]))
    grp_colors <- setNames(rep(palette, length.out = length(grps)), grps)
  }
  spans <- vapply(seq_len(nrow(lemmas_df)), function(i) {
    col <- if (!is.null(grp_colors))
             grp_colors[[ as.character(lemmas_df[[group_col]][i]) ]]
           else "#2C7BB6"
    grp_txt <- if (!is.null(group_col))
      sprintf(" (%s)", as.character(lemmas_df[[group_col]][i])) else ""
    sprintf('<span style="display:inline-block;margin:4px 8px;font-size:%.0fpx;color:%s;font-weight:500" title="freq=%d%s">%s</span>',
            lemmas_df$.size[i], col, lemmas_df$freq[i],
            htmltools::htmlEscape(grp_txt),
            htmltools::htmlEscape(lemmas_df$lemma[i]))
  }, character(1))
  legend_html <- ""
  if (!is.null(grp_colors)) {
    chips <- vapply(names(grp_colors), function(g)
      sprintf('<span style="display:inline-flex;align-items:center;gap:6px;margin:0 10px;font-size:13px;color:#333"><span style="display:inline-block;width:14px;height:14px;background:%s;border-radius:3px"></span>%s</span>',
              grp_colors[[g]], htmltools::htmlEscape(g)), character(1))
    legend_html <- paste0(
      '<div style="text-align:center;padding:6px 0 10px 0;border-bottom:1px dashed #ddd;margin-bottom:8px">',
      '<b style="color:#555;margin-right:8px">Speakers:</b>',
      paste(chips, collapse = ""), '</div>')
  }
  paste0('<div style="padding:8px">', legend_html,
         '<div style="line-height:1.6;text-align:center;padding:14px">',
         paste(spans, collapse = ""), "</div></div>")
}
