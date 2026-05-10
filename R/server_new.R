# R/server_new.R — Server logic for new features.
# Invoked from server() with: input, output, session, rv.

attach_new_server <- function(input, output, session, rv) {

  # Local helper: datatable with CSV/Excel buttons that download the full dataset.
  # Defensive duplicate in case not loaded in the global environment.
  dt_analysis <- function(df, ...) {
    DT::datatable(df, extensions = "Buttons",
      options = list(pageLength = 15, dom = "Blfrtip",
        buttons = list("copy",
          list(extend="csv",   text="⬇ Full CSV",
               exportOptions = list(modifier = list(search = "none"))),
          list(extend="excel", text="⬇ Excel",
               exportOptions = list(modifier = list(search = "none")))),
        scrollX = TRUE),
      rownames = FALSE, ...)
  }


  # ═══════════════════════════════════════════════════════════
  #  SAVED SESSIONS (single-file save/load)
  # ═══════════════════════════════════════════════════════════
  sess_refresh <- shiny::reactiveVal(0)
  shiny::observeEvent(input$btn_sessions_refresh, sess_refresh(sess_refresh() + 1))

  sess_list <- shiny::reactive({
    sess_refresh()
    tryCatch(acustera_session_list(), error = function(e) NULL)
  })

  output$ui_sessions_picker <- shiny::renderUI({
    df <- sess_list()
    if (is.null(df) || nrow(df) == 0) {
      return(shiny::tags$div(class = "callout callout-info",
        shiny::icon("circle-info"),
        " No saved analyses yet. Process a couple and press ",
        shiny::tags$b("Save analysis"), " at the end of the pipeline."))
    }
    labels <- sprintf(
      "%s  ·  %s  ·  %s  ·  %d couples · %d sessions · %d segs%s%s",
      format(df$saved_at, "%Y-%m-%d %H:%M"),
      df$label, df$pairs, df$n_pairs, df$n_sess, df$n_seg,
      ifelse(df$has_trans, " · +trans", ""),
      ifelse(df$has_tok,   " · +POS",   ""))
    shiny::selectizeInput("sessions_picked", "Saved analyses",
      choices = setNames(df$file, labels), multiple = TRUE,
      options = list(placeholder = "Select one or more…"))
  })

  shiny::observeEvent(input$btn_session_save, {
    tryCatch({
      path <- acustera_session_save(rv, label = input$session_save_label %||% "")
      sess_refresh(sess_refresh() + 1)
      shiny::showNotification(
        paste0("✓ Analysis saved: ", basename(path)),
        type = "message", duration = 4)
    }, error = function(e)
      shiny::showNotification(paste("Error saving:", conditionMessage(e)),
                              type = "error"))
  })

  shiny::observeEvent(input$btn_sessions_load, {
    paths <- input$sessions_picked
    if (is.null(paths) || length(paths) == 0) {
      shiny::showNotification("Select at least one analysis.",
                              type = "warning"); return()
    }
    tryCatch({
      out <- acustera_session_load(paths)
      rv$features   <- out$features
      rv$transcript <- out$transcript
      rv$overlaps   <- out$overlaps
      rv$sess_meta  <- out$sess_meta
      rv$spk_meta   <- out$spk_meta
      if (!is.null(out$tokens)) rv$tokens <- out$tokens
      shiny::showNotification(
        sprintf("✓ Loaded %d analyses · %d segments · %d couples.",
          length(paths), nrow(out$features %||% data.frame()),
          length(unique(out$features$couple))),
        type = "message", duration = 5)
      shinydashboard::updateTabItems(session, "sidebar_menu", "tab_meta")
    }, error = function(e)
      shiny::showNotification(paste("Error loading:", conditionMessage(e)),
                              type = "error", duration = 6))
  })

  shiny::observeEvent(input$btn_sessions_delete, {
    paths <- input$sessions_picked
    if (is.null(paths) || length(paths) == 0) return()
    for (p in paths) tryCatch(acustera_session_delete(p),
                              error = function(e) NULL)
    sess_refresh(sess_refresh() + 1)
    shiny::showNotification(sprintf("✓ %d analyses deleted.", length(paths)),
                            type = "message")
  })

  # ═══════════════════════════════════════════════════════════
  #  LINGUISTICS / POS
  # ═══════════════════════════════════════════════════════════
  shiny::observe({
    shiny::req(rv$features)
    shiny::updateSelectInput(session, "ling_couple",
      choices = c("All", sort(unique(as.character(rv$features$couple)))))
  })

  output$ui_pos_status <- shiny::renderUI({
    if (is.null(rv$tokens)) {
      shiny::tags$div(style = "color:#888;font-size:12px;padding-top:6px",
        shiny::icon("circle-info"), " Not yet annotated.")
    } else {
      shiny::tags$div(style = "color:#27ae60;font-size:12px;padding-top:6px",
        shiny::icon("check"), sprintf(" %d tokens annotated · %d unique lemmas.",
          nrow(rv$tokens), length(unique(rv$tokens$lemma))))
    }
  })

  shiny::observeEvent(input$btn_pos_run, {
    shiny::req(rv$features)
    if (!udpipe_available()) {
      shiny::showNotification(
        "Install udpipe with: install.packages('udpipe')",
        type = "error", duration = 8); return()
    }
    if (!"text" %in% names(rv$features) ||
        !any(!is.na(rv$features$text))) {
      shiny::showNotification("No transcription available.",
        type = "warning"); return()
    }
    shiny::withProgress(message = "Annotating POS...", value = 0.2, {
      tryCatch({
        tok <- annotate_pos(rv$features, text_col = "text",
          log = function(m) shiny::incProgress(0.1, detail = m))
        if (is.null(tok) || nrow(tok) == 0) {
          shiny::showNotification("No tokens obtained.", type = "warning")
          return()
        }
        rv$tokens <- tok
        shiny::incProgress(1, detail = "Done")
        shiny::showNotification(
          sprintf("✓ %d tokens annotated.", nrow(tok)), type = "message")
      }, error = function(e) {
        shiny::showNotification(paste("Error udpipe:", conditionMessage(e)),
          type = "error", duration = 8)
      })
    })
  })

  # Reactive filter on tokens
  tokens_filt <- shiny::reactive({
    shiny::req(rv$tokens)
    tok <- rv$tokens
    if (!is.null(input$ling_speaker) && input$ling_speaker != "All")
      tok <- dplyr::filter(tok, speaker == input$ling_speaker)
    if (!is.null(input$ling_couple) && input$ling_couple != "All")
      tok <- dplyr::filter(tok, couple == input$ling_couple)
    if (!is.null(input$ling_quantile) && input$ling_quantile != "All" &&
        "quantile" %in% names(tok))
      tok <- dplyr::filter(tok, as.character(quantile) == input$ling_quantile)
    tok
  }) |> shiny::bindEvent(input$btn_ling_refresh, rv$tokens,
                          input$ling_speaker, input$ling_couple,
                          input$ling_quantile, ignoreNULL = FALSE)

  output$ui_wordcloud <- shiny::renderUI({
    tok <- tokens_filt()
    if (is.null(tok) || nrow(tok) == 0)
      return(shiny::tags$p(style="color:#888",
        "Click 'Annotate transcription' and then 'Refresh'."))
    facet <- input$ling_facet %||% "none"
    make_cloud <- function(tok_sub, titulo = NULL) {
      lemas <- top_lemmas(tok_sub, by = "speaker",
                          upos_filter = input$ling_upos,
                          n = input$ling_top %||% 80)
      if (is.null(lemas) || nrow(lemas) == 0)
        return(shiny::tags$div(style="color:#888;padding:8px",
                               if (!is.null(titulo)) shiny::tags$h5(titulo),
                               "No lemmas."))
      shiny::tags$div(style="padding:6px 8px",
        if (!is.null(titulo))
          shiny::tags$h5(titulo,
            style="margin:4px 0 6px;font-size:13px;color:#2C7BB6;border-bottom:1px solid #e0e0e0;padding-bottom:3px"),
        shiny::HTML(wordcloud_html(lemas, group_col = "speaker")))
    }
    if (identical(facet, "none") || !(facet %in% names(tok)))
      return(make_cloud(tok))
    niveles <- sort(unique(tok[[facet]]))
    niveles <- niveles[!is.na(niveles)]
    if (length(niveles) == 0) return(make_cloud(tok))
    etiqueta <- c(couple = "Couple", therapy_num = "Session",
                  quantile = "Quartile", speaker = "Speaker")[facet]
    # Two per row if few facets; one column if many
    w <- if (length(niveles) <= 2) 6 else if (length(niveles) <= 4) 6 else 4
    cols <- lapply(niveles, function(lv) {
      sub <- tok[as.character(tok[[facet]]) == as.character(lv), , drop = FALSE]
      shiny::column(w, make_cloud(sub,
        titulo = sprintf("%s: %s", etiqueta %||% facet, lv)))
    })
    do.call(shiny::fluidRow, cols)
  })

  output$p_pos <- plotly::renderPlotly({
    tok <- tokens_filt()
    shiny::req(tok, nrow(tok) > 0)
    sm <- pos_summary(tok, by = c("speaker","quantile"))
    shiny::req(sm)
    plotly::plot_ly(sm, x = ~quantile, y = ~pct, color = ~upos, type = "bar") |>
      plotly::layout(barmode = "stack", yaxis = list(title = "% tokens"),
                     xaxis = list(title = "Quartile"),
                     legend = list(orientation = "h"))
  })

  output$p_pron <- plotly::renderPlotly({
    tok <- tokens_filt()
    shiny::req(tok, nrow(tok) > 0)
    cats <- input$ling_marker_cats %||% c("pron_1sg","pron_1pl","pron_2sg",
                                          "pron_3sg","pron_3pl")
    unit <- input$ling_marker_unit %||% "pct"
    pm <- lexical_markers(tok, categorias = cats)
    shiny::req(pm, nrow(pm) > 0)
    # Correct denominator: total tokens per speaker (unique, not broken down
    # by quartile or session, so it does not depend on where the category appears).
    n_tot_hab <- tok |>
      dplyr::group_by(speaker) |>
      dplyr::summarise(n_tot = dplyr::n(), .groups = "drop")
    pm_ag <- pm |>
      dplyr::group_by(speaker, categoria) |>
      dplyr::summarise(n = sum(n), .groups = "drop") |>
      dplyr::left_join(n_tot_hab, by = "speaker") |>
      dplyr::mutate(pct = 100 * n / pmax(n_tot, 1))
    yvar  <- if (identical(unit, "pct")) "pct" else "n"
    ytit  <- if (identical(unit, "pct")) "% of speaker tokens" else "Frequency"
    hover <- if (identical(unit, "pct"))
      sprintf("<b>%s</b><br>%s: %.2f%% (%d/%d tokens)",
              pm_ag$speaker, pm_ag$categoria, pm_ag$pct, pm_ag$n, pm_ag$n_tot)
    else
      sprintf("<b>%s</b><br>%s: %d (de %d tokens)",
              pm_ag$speaker, pm_ag$categoria, pm_ag$n, pm_ag$n_tot)
    plotly::plot_ly(pm_ag, x = ~categoria, y = pm_ag[[yvar]],
                    color = ~speaker, type = "bar",
                    text = hover, hoverinfo = "text") |>
      plotly::layout(barmode = "group",
        yaxis = list(title = ytit),
        xaxis = list(title = "Lexical category"))
  })

  output$dt_lex <- DT::renderDT({
    tok <- tokens_filt()
    shiny::req(tok, nrow(tok) > 0)
    lm <- lexical_metrics(tok, by = "speaker")
    dt_analysis(lm)
  })

  output$dl_tokens_csv <- shiny::downloadHandler(
    filename = function() paste0("acustera_tokens_",
      format(Sys.time(),"%Y%m%d_%H%M"), ".csv"),
    content  = function(f) {
      shiny::req(rv$tokens)
      readr::write_csv(rv$tokens, f)
    }
  )

  # ═══════════════════════════════════════════════════════════
  #  EXAMPLES V2
  # ═══════════════════════════════════════════════════════════
  shiny::observe({
    shiny::req(rv$features)
    shiny::updateSelectInput(session, "ej2_couple",
      choices = c("All", sort(unique(as.character(rv$features$couple)))),
      selected = "All")
  })

  output$ui_ej2_sesrange <- shiny::renderUI({
    shiny::req(rv$features)
    rng <- range(rv$features$therapy_num, na.rm = TRUE)
    if (any(!is.finite(rng))) rng <- c(1, 20)
    shiny::sliderInput("ej2_sesrange","Session range",
      min = rng[1], max = rng[2], value = rng, step = 1)
  })

  # Debounce on percentile
  pct_d <- shiny::reactive(input$ej2_pct) |> shiny::debounce(400)

  examples2_df <- shiny::reactive({
    shiny::req(rv$features)
    if (!"text" %in% names(rv$features)) return(NULL)
    build_examples_v2(
      rv$features,
      params        = input$ej2_param %||% "f0_mean",
      direction     = input$ej2_dir   %||% "both",
      percentil     = (pct_d() %||% 15) / 100,
      quantiles     = input$ej2_quantile,
      sesiones_range = input$ej2_sesrange,
      speakers     = if (is.null(input$ej2_speaker) || input$ej2_speaker=="All")
                        c("man","woman","therapist") else input$ej2_speaker,
      couples       = input$ej2_couple,
      tokens_df     = rv$tokens,
      upos_required = input$ej2_upos
    )
  }) |> shiny::bindEvent(input$btn_ej2_refresh, rv$features,
                          ignoreNULL = FALSE)

  output$ui_ej2_summary <- shiny::renderUI({
    df <- examples2_df()
    n <- if (is.null(df)) 0 else nrow(df)
    shiny::tags$div(style="font-size:12px;color:#666;margin-bottom:8px",
      shiny::icon("list"), " ", shiny::tags$b(n), " examples found · ",
      "selected: ", shiny::tags$b(length(rv$ej2_selected %||% character(0))))
  })

  output$ui_ej2_cards <- shiny::renderUI({
    df <- examples2_df()
    if (is.null(df) || nrow(df) == 0)
      return(shiny::tags$p(style="color:#888;font-style:italic",
        "No examples. Adjust filters."))
    npg <- input$ej2_npage %||% 30
    df <- head(df, npg)
    sel <- rv$ej2_selected %||% character(0)
    rows <- lapply(seq_len(nrow(df)), function(i) {
      r <- df[i, , drop = FALSE]
      au <- audio_url_for(r$couple, r$therapy_num)
      shiny::HTML(render_example_paper(r, sel_ids = sel, audio_url = au,
                                       full_df = rv$features,
                                       ctx_before = 2, ctx_after = 2))
    })
    shiny::tagList(rows)
  })

  # Selection toggle
  shiny::observeEvent(input$ej2_toggle, {
    id <- input$ej2_toggle$id
    if (is.null(id)) return()
    sel <- rv$ej2_selected %||% character(0)
    if (id %in% sel) sel <- setdiff(sel, id) else sel <- c(sel, id)
    rv$ej2_selected <- sel
  })

  shiny::observeEvent(input$btn_ej2_clear, { rv$ej2_selected <- character(0) })

  output$ui_ej2_selected <- shiny::renderUI({
    sel <- rv$ej2_selected %||% character(0)
    if (length(sel) == 0)
      return(shiny::tags$p(style="color:#888;font-size:12px",
        "No marked examples. Click + mark on a card."))
    df <- examples2_df()
    if (is.null(df)) return(NULL)
    picks <- df[df$.id %in% sel, , drop = FALSE]
    if (nrow(picks) == 0)
      return(shiny::tags$p(style="color:#888;font-size:12px",
        "Current selection does not match the filters."))
    items <- lapply(seq_len(nrow(picks)), function(i) {
      r <- picks[i, , drop = FALSE]
      col <- switch(as.character(r$speaker),
        man = "#2C7BB6", woman = "#c0392b",
        therapist = "#27ae60", "#888")
      shiny::HTML(sprintf(
        '<div style="border-left:2px solid %s;padding:6px 8px;margin-bottom:6px;font-size:11px">
          <b style="color:%s">%s</b> · Couple %s · S%s<br>
          <span style="color:#333">%s</span>
         </div>',
        col, col, r$speaker, r$couple, r$therapy_num,
        htmltools::htmlEscape(substr(r$text, 1, 160))))
    })
    shiny::tagList(items)
  })

  output$dl_ej2_csv <- shiny::downloadHandler(
    filename = function() paste0("acustera_examples_",
      format(Sys.time(),"%Y%m%d_%H%M"), ".csv"),
    content = function(f) {
      df <- examples2_df()
      sel <- rv$ej2_selected %||% character(0)
      if (!is.null(df) && length(sel) > 0)
        df <- df[df$.id %in% sel, , drop = FALSE]
      readr::write_csv(df %||% data.frame(), f)
    }
  )

  # ═══════════════════════════════════════════════════════════
  #  AI SPEAKER REVIEW
  # ═══════════════════════════════════════════════════════════
  shiny::observeEvent(input$btn_heur_run, {
    shiny::req(rv$features)
    shiny::withProgress(message = "Detecting suspicious turns...", value = 0.3, {
      flagged <- speaker_heuristic_review(rv$features, rv$transcript)
      rv$ai_review <- flagged
      shiny::incProgress(1)
    })
    n <- if (is.null(rv$ai_review)) 0 else nrow(rv$ai_review)
    shiny::showNotification(sprintf("✓ %d turns flagged.", n), type = "message")
  })

  shiny::observeEvent(input$btn_ai_run, {
    shiny::req(rv$ai_review)
    if (nrow(rv$ai_review) == 0) {
      shiny::showNotification("No flagged turns. Run the heuristic first.",
        type = "warning"); return()
    }
    shiny::withProgress(message = "Consulting AI...", value = 0, {
      prog <- function(p, msg) shiny::setProgress(value = p, detail = msg)
      rv$ai_review <- speaker_llm_review(rv$ai_review, rv$features,
        provider = input$ai_rev_provider %||% "anthropic", progress = prog)
    })
    shiny::showNotification("✓ AI review completed.", type = "message")
  })

  shiny::observeEvent(input$btn_ai_run_full, {
    shiny::req(rv$features)
    if (!"text" %in% names(rv$features) ||
        all(is.na(rv$features$text))) {
      shiny::showNotification("No transcription available in features.",
        type = "warning"); return()
    }
    shiny::withProgress(message = "Full AI review...", value = 0, {
      prog <- function(p, msg) shiny::setProgress(value = p, detail = msg)
      res <- tryCatch(speaker_llm_review_full(rv$features,
        provider = input$ai_rev_provider %||% "anthropic",
        ctx = as.integer(input$ai_rev_ctx %||% 1),
        batch_size = as.integer(input$ai_rev_batch %||% 8),
        confidence_threshold = as.numeric(input$ai_rev_conf %||% 0.7),
        only_changes = isTRUE(input$ai_rev_only_changes),
        progress = prog),
        error = function(e) { shiny::showNotification(
          paste("AI error:", conditionMessage(e)), type = "error"); NULL })
      if (!is.null(res)) rv$ai_review <- res
    })
    n <- if (is.null(rv$ai_review)) 0 else nrow(rv$ai_review)
    shiny::showNotification(sprintf("✓ Full AI review: %d proposed changes.", n),
                            type = "message", duration = 5)
  })

  output$dt_ai_review <- DT::renderDT({
    df <- rv$ai_review
    if (is.null(df) || nrow(df) == 0) return(NULL)
    dt_analysis(df, filter = "top", selection = "multiple")
  })

  shiny::observeEvent(input$btn_ai_apply, {
    df <- rv$ai_review
    if (is.null(df) || !"sugerencia" %in% names(df)) {
      shiny::showNotification("No AI suggestions to apply.",
        type = "warning"); return()
    }
    sel <- input$dt_ai_review_rows_selected
    if (is.null(sel) || length(sel) == 0) sel <- seq_len(nrow(df))
    chg <- df[sel, , drop = FALSE]
    chg <- chg[!is.na(chg$sugerencia) &
               chg$sugerencia %in% c("man","woman","therapist"), ,
               drop = FALSE]
    if (nrow(chg) == 0) {
      shiny::showNotification("No applicable suggestions.",
        type = "warning"); return()
    }
    f <- rv$features
    for (i in seq_len(nrow(chg))) {
      idx <- which(f$couple == chg$couple[i] &
                   f$therapy_num == chg$therapy_num[i] &
                   abs(f$seg_start - chg$seg_start[i]) < 0.01)
      if (length(idx) > 0) f$speaker[idx] <- chg$sugerencia[i]
    }
    rv$features <- f
    shiny::showNotification(
      sprintf("✓ %d corrections applied.", nrow(chg)), type = "message")
  })

  # ═══════════════════════════════════════════════════════════
  #  API KEYS (config)
  # ═══════════════════════════════════════════════════════════
  output$ui_api_keys <- shiny::renderUI({
    k <- ai_load_keys()
    shiny::tagList(
      shiny::passwordInput("api_anthropic","API key Anthropic",
        value = if (nzchar(k$anthropic %||% "")) "••••••••" else ""),
      shiny::passwordInput("api_openai","API key OpenAI",
        value = if (nzchar(k$openai %||% "")) "••••••••" else ""),
      shiny::passwordInput("api_hf","Token HuggingFace",
        value = if (nzchar(k$hf %||% "")) "••••••••" else "",
        placeholder = "hf_xxxxxxxxxxxxxxxxxxxx"),
      shiny::actionButton("btn_save_keys","Save keys",
        class = "btn btn-primary btn-sm w-100"),
      shiny::tags$p(style="font-size:11px;color:#888;margin-top:8px",
        shiny::icon("lock")," Stored in ",
        shiny::tags$code(".acustera_keys.json"),
        " with 600 permissions. Alternative: ",
        shiny::tags$code("ANTHROPIC_API_KEY"),", ",
        shiny::tags$code("OPENAI_API_KEY"),", ",
        shiny::tags$code("HF_TOKEN")," as environment variables.")
    )
  })

  shiny::observeEvent(input$btn_save_keys, {
    a <- input$api_anthropic; o <- input$api_openai; h <- input$api_hf
    if (identical(a, "••••••••")) a <- NULL
    if (identical(o, "••••••••")) o <- NULL
    if (identical(h, "••••••••")) h <- NULL
    ai_save_keys(anthropic = a, openai = o, hf = h)
    # Preload hf_token in the Process tab when a new token was saved
    if (!is.null(h) && nzchar(h))
      shiny::updateTextInput(session, "hf_token", value = h)
    shiny::showNotification("✓ Keys saved.", type = "message")
  })

  # Preload hf_token in the Process tab at session start
  shiny::observe({
    k <- ai_load_keys()
    if (nzchar(k$hf %||% "") && !nzchar(shiny::isolate(input$hf_token %||% "")))
      shiny::updateTextInput(session, "hf_token", value = k$hf)
  })

  # ═══════════════════════════════════════════════════════════
  #  REPORT V2 + send to AI
  # ═══════════════════════════════════════════════════════════
  build_report_payload <- function() {
    tok <- rv$tokens
    lex <- if (!is.null(tok)) lexical_metrics(tok, by = "speaker") else NULL
    possum <- if (!is.null(tok)) pos_summary(tok, by = c("speaker","quantile"))
              else NULL
    sel_df <- NULL
    ex_html <- NULL
    df2 <- tryCatch(build_examples_v2(rv$features,
      params = c("f0_mean","intensity_mean"), percentil = 0.15),
      error = function(e) NULL)
    if (!is.null(df2) && !is.null(rv$ej2_selected) && length(rv$ej2_selected) > 0) {
      sel_df <- df2[df2$.id %in% rv$ej2_selected, , drop = FALSE]
      if (!is.null(sel_df) && nrow(sel_df) > 0) {
        cards <- vapply(seq_len(nrow(sel_df)), function(i)
          render_example_paper(sel_df[i, , drop=FALSE],
                               full_df = rv$features,
                               ctx_before = 2, ctx_after = 2), character(1))
        ex_html <- paste(cards, collapse = "")
      }
    }
    list(lex = lex, possum = possum, sel_df = sel_df, ex_html = ex_html)
  }

  output$btn_dl_v2 <- shiny::downloadHandler(
    filename = function() paste0("acustera_v2_",
      format(Sys.time(),"%Y%m%d_%H%M"), ".html"),
    content = function(f) {
      pl <- build_report_payload()
      txt <- tryCatch(
        build_text_block(rv$features, rv$overlaps, rv$lmm, rv$posthoc,
          input$active_params %||% names(PARAM_LABELS)),
        error = function(e) "")
      html <- build_report_v2(
        text_block = txt, features = rv$features,
        overlaps = rv$overlaps, lmm = rv$lmm, posthoc = rv$posthoc,
        notes = input$rep_notes %||% "", sess_meta = rv$sess_meta,
        conv_lmm = rv$conv_lmm, lex_metrics = pl$lex,
        pos_summary_df = pl$possum, selected_examples = pl$sel_df,
        examples_html = pl$ex_html,
        ai_interpretation = rv$ai_interp %||% NULL)
      writeLines(html, f)
    })

  output$ui_dl_v2 <- shiny::renderUI({
    shiny::req(rv$features)
    shiny::downloadButton("btn_dl_v2","⬇ Report v2 (HTML with AI hints)",
      class = "btn btn-primary w-100")
  })

  shiny::observeEvent(input$btn_ai_interpret, {
    shiny::req(rv$features)
    pl <- build_report_payload()
    ctx <- ai_context_json(rv$features, rv$lmm, rv$posthoc, rv$conv_lmm,
      pl$lex, pl$sel_df, input$rep_notes %||% "", rv$sess_meta)
    sys <- "You are a clinical methodological reviewer. Reply in English, 500-800 words, with subheadings."
    user <- paste0(ai_prompt_template(),
      "\n\n--- Context JSON ---\n", ctx)
    shiny::withProgress(message = "Requesting AI interpretation...", value = 0.3, {
      r <- ai_call(input$ai_report_provider %||% "anthropic",
        system = sys, user = user, max_tokens = 2500)
      shiny::incProgress(1)
      if (!r$ok) {
        shiny::showNotification(paste("AI error:", substr(r$error, 1, 200)),
          type = "error", duration = 10); return()
      }
      rv$ai_interp <- r$text
      shiny::showNotification("✓ AI interpretation ready.", type = "message")
    })
  })

  output$ui_ai_interp <- shiny::renderUI({
    if (is.null(rv$ai_interp))
      return(shiny::tags$p(style="color:#888;font-style:italic",
        "Click 'Request AI interpretation' to generate an automatic comment."))
    shiny::tags$div(style="background:#f7f9fc;border:1px solid #d7e3f0;padding:14px;border-radius:4px;white-space:pre-wrap",
      rv$ai_interp)
  })
}
