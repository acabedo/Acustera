# R/ui_tabs_new.R — Content for the new tabs.
# Return shinydashboard::tabItem(...) ready to insert into tabItems().

ui_tab_linguistica <- function() {
  shinydashboard::tabItem("tab_ling",
    shiny::fluidRow(
      shinydashboard::box(
        title = "Morphosyntactic annotation (udpipe)", width = 12, status = "primary",
        shiny::div(class = "callout callout-info",
          "Applies the Spanish UDPipe model to transcriptions to obtain ",
          shiny::tags$b("lemma"), " and ", shiny::tags$b("POS category"), " per token. ",
          "The model (~50 MB) is downloaded once and cached."),
        shiny::fluidRow(
          shiny::column(4,
            shiny::actionButton("btn_pos_run",
              shiny::tagList(shiny::icon("language"), " Annotate transcription"),
              class = "btn btn-primary w-100")),
          shiny::column(4,
            shiny::uiOutput("ui_pos_status")),
          shiny::column(4,
            shiny::downloadButton("dl_tokens_csv", "⬇ Tokens CSV",
              class = "btn btn-outline-secondary btn-sm w-100"))
        )
      )
    ),
    shiny::fluidRow(
      shinydashboard::box(
        title = "Filters", width = 3, status = "primary",
        shiny::selectInput("ling_speaker", "Speaker",
          c("All","man","woman","therapist"), "All"),
        shiny::selectInput("ling_couple", "Couple", c("All")),
        shiny::selectInput("ling_quantile", "Time quartile",
          c("All","Q1","Q2","Q3","Q4"), "All"),
        shiny::checkboxGroupInput("ling_upos", "POS categories (for cloud)",
          choices = c("NOUN","VERB","ADJ","ADV","PRON","DET","ADP","CCONJ","SCONJ"),
          selected = c("NOUN","VERB","ADJ","ADV")),
        shiny::sliderInput("ling_top", "Top lemmas", 20, 200, 80, 10),
        shiny::hr(),
        shiny::actionButton("btn_ling_refresh",
          shiny::tagList(shiny::icon("rotate")," Refresh"),
          class = "btn btn-primary btn-sm w-100")
      ),
      shinydashboard::box(
        title = "Lemma cloud", width = 9, status = "primary",
        collapsible = TRUE,
        style = "background:#fff;min-height:360px",
        shiny::fluidRow(
          shiny::column(4,
            shiny::selectInput("ling_facet", "Facet",
              c("No facet"     = "none",
                "By couple"    = "couple",
                "By session"   = "therapy_num",
                "By quartile"  = "quantile",
                "By speaker"   = "speaker"),
              selected = "none")),
          shiny::column(8,
            shiny::helpText(style="font-size:11px;color:#777;margin-top:24px",
              "If you choose a facet, one cloud is shown per value of that variable. Colors follow the speaker."))
        ),
        shiny::uiOutput("ui_wordcloud")
      )
    ),
    shiny::fluidRow(
      shinydashboard::box(
        title = "POS distribution by speaker and quartile", width = 6,
        plotly::plotlyOutput("p_pos", height = "340px")),
      shinydashboard::box(
        title = "Lexical markers by speaker", width = 6,
        collapsible = TRUE,
        shiny::div(class = "callout callout-warning", style="font-size:11.5px",
          shiny::tags$b("Methodological note."),
          " Categories are ", shiny::tags$b("ad-hoc lexical lists"),
          " defined in ",
          shiny::tags$code("R/fn_pos.R → lexical_markers()"),
          ". They do not come from a validated dictionary such as LIWC/NRC.",
          shiny::tags$br(),
          shiny::tags$b("What each category captures:"),
          shiny::tags$ul(style="margin:4px 0 0 18px;padding:0",
            shiny::tags$li(shiny::tags$b("Pron. 1st/2nd/3rd:"),
              " PRON lemmas by person and number (yo/mí/me; tú/ti/te; él/ella/le…)."),
            shiny::tags$li(shiny::tags$b("Possessives / Demonstratives:"),
              " determiners and pronouns (mi, tu, su, nuestro / este, ese, aquel…)."),
            shiny::tags$li(shiny::tags$b("Negation:"),
              " no, nunca, jamás, nada, nadie, ningún/a/o, tampoco."),
            shiny::tags$li(shiny::tags$b("Intensifiers:"),
              " muy, mucho, demasiado, bastante, tan, totalmente, siempre…"),
            shiny::tags$li(shiny::tags$b("Hedges:"),
              " creo, quizás, supongo, parecer, probablemente, algo…"),
            shiny::tags$li(shiny::tags$b("Adversatives:"),
              " pero, sino, aunque."),
            shiny::tags$li(shiny::tags$b("Emotion (−/+):"),
              " short affective-core lists (odiar, miedo, triste, culpa, frustrar… / amor, feliz, alegría, tranquilo…).")
          ),
          shiny::tags$b("Relativisation:"),
          " % is calculated over the ", shiny::tags$b("total tokens of the speaker"),
          " under the current filters. The denominator is the same for all categories."),
        shiny::fluidRow(
          shiny::column(7,
            shiny::selectInput("ling_marker_cats",
              "Categories",
              choices = c("1st sing pron." = "pron_1sg",
                          "1st plur pron." = "pron_1pl",
                          "2nd sing pron." = "pron_2sg",
                          "3rd sing pron." = "pron_3sg",
                          "3rd plur pron." = "pron_3pl",
                          "Possessives"      = "posesivos",
                          "Demonstratives"  = "demostrativos",
                          "Negation"       = "negacion",
                          "Intensifiers" = "intensificadores",
                          "Hedges"    = "atenuadores",
                          "Adversatives"   = "adversativos",
                          "Emotion (−)"    = "emocion_neg",
                          "Emotion (+)"    = "emocion_pos"),
              selected = c("pron_1sg","pron_1pl","pron_2sg",
                           "pron_3sg","pron_3pl"),
              multiple = TRUE)),
          shiny::column(5,
            shiny::radioButtons("ling_marker_unit","Unit",
              c("% of speaker tokens" = "pct",
                "Absolute frequency"         = "n"),
              selected = "pct"))
        ),
        plotly::plotlyOutput("p_pron", height = "340px"))
    ),
    shiny::fluidRow(
      shinydashboard::box(
        title = "Lexical metrics (TTR, density, Yule's K)", width = 12,
        collapsible = TRUE,
        shiny::div(class = "callout callout-info",
          shiny::tags$b("What each metric measures:"), shiny::tags$br(),
          shiny::tags$b("TTR (Type-Token Ratio):"),
          " number of distinct lemmas / total number of words. Close to 1 = high lexical variety; close to 0 = high repetition. Sensitive to sample size.",
          shiny::tags$br(),
          shiny::tags$b("Lexical density:"),
          " proportion of content words (NOUN, VERB, ADJ, ADV) over the total. High values = more informative/descriptive speech; low values = more conversational/functional speech.",
          shiny::tags$br(),
          shiny::tags$b("Yule's K:"),
          " lexical repetition index independent of text size. Low K = broad, non-repetitive vocabulary; high K = vocabulary concentrated in a few very frequent forms."),
        DT::DTOutput("dt_lex"))
    )
  )
}

ui_tab_ai_review <- function() {
  shinydashboard::tabItem("tab_ai_rev",
    shiny::fluidRow(
      shinydashboard::box(
        title = "AI review of speaker attribution", width = 12, status = "primary",
        shiny::div(class = "callout callout-warning", shiny::tags$b("Privacy. "), "This option sends data to an external provider (Anthropic or OpenAI): ", "excerpts of the transcript and acoustic summaries leave this computer. ", "Use it only if consent, ethics approval and data protection rules allow it. ", "The rest of AcusTera runs locally."),
        shiny::div(class = "callout callout-info",
          shiny::tags$b("Two modes:"), shiny::tags$br(),
          shiny::tags$b("1) Fast heuristic:"),
          " automatically flags suspicious turns (atypical F0 or lexicon inconsistent with the role).",
          shiny::tags$br(),
          shiny::tags$b("2) Full AI review:"),
          " the LLM reads ",
          shiny::tags$b("all segments"),
          " with ±N turns of context and checks attribution based on content.",
          shiny::tags$br(),
          shiny::tags$small(style="color:#666",
            "Approx. cost: ~$0.005 per 100 turns with Sonnet / ~$0.001 with GPT-4o-mini."))
      )
    ),
    shiny::fluidRow(
      shinydashboard::box(
        title = "Controls", width = 12, status = "primary",
        shiny::fluidRow(
          shiny::column(3,
            shiny::selectInput("ai_rev_provider","Provider",
              c("Anthropic" = "anthropic", "OpenAI" = "openai"), "anthropic")),
          shiny::column(2,
            shiny::numericInput("ai_rev_ctx","Context ± (turns)", 1, 0, 4, 1)),
          shiny::column(2,
            shiny::numericInput("ai_rev_batch","Batch", 8, 3, 20, 1)),
          shiny::column(2,
            shiny::sliderInput("ai_rev_conf","Minimum confidence",
              0.5, 0.95, 0.7, 0.05)),
          shiny::column(3,
            shiny::checkboxInput("ai_rev_only_changes",
              "Show only changes", TRUE))
        ),
        shiny::fluidRow(
          shiny::column(3,
            shiny::actionButton("btn_heur_run",
              shiny::tagList(shiny::icon("magnifying-glass")," Heuristic (fast)"),
              class = "btn btn-outline-primary w-100")),
          shiny::column(3,
            shiny::actionButton("btn_ai_run_full",
              shiny::tagList(shiny::icon("robot")," Full AI review"),
              class = "btn btn-warning w-100")),
          shiny::column(3,
            shiny::actionButton("btn_ai_run",
              shiny::tagList(shiny::icon("robot")," Review only suspects"),
              class = "btn btn-outline-warning w-100")),
          shiny::column(3,
            shiny::actionButton("btn_ai_apply",
              shiny::tagList(shiny::icon("check-double")," Apply suggestions"),
              class = "btn btn-success w-100"))
        )
      )
    ),
    shiny::fluidRow(
      shinydashboard::box(
        title = "Speaker change proposals", width = 12,
        DT::DTOutput("dt_ai_review"))
    )
  )
}

ui_tab_examples_v2 <- function() {
  shinydashboard::tabItem("tab_ej",
    shiny::fluidRow(
      shinydashboard::box(
        title = "Filters", width = 3, status = "primary",
        shiny::uiOutput("ui_examples_guard"),
        shiny::selectInput("ej2_couple","Couple", c("All"),
                           multiple = TRUE, selected = "All"),
        shiny::selectInput("ej2_speaker","Speaker",
                           c("All","man","woman","therapist"), "All"),
        shiny::selectInput("ej2_param","Prosodic parameter",
                           c("F0 (mean)"  = "f0_mean",
                             "F0 (SD)"    = "f0_sd",
                             "F0 (range)" = "f0_range",
                             "Intensity"  = "intensity_mean",
                             "HNR"        = "hnr",
                             "Jitter"     = "jitter_local",
                             "Shimmer"    = "shimmer_local",
                             "Rate"       = "speech_rate"),
                           multiple = TRUE, selected = c("f0_mean","intensity_mean")),
        shiny::radioButtons("ej2_dir","Direction",
                            c("both","high","low"), "both", inline = TRUE),
        shiny::sliderInput("ej2_pct","Extreme percentile (%)",5,40,15,5,post="%"),
        shiny::checkboxGroupInput("ej2_quantile","Time quartile",
                                  c("Q1","Q2","Q3","Q4"),
                                  selected = c("Q1","Q2","Q3","Q4"),
                                  inline = TRUE),
        shiny::uiOutput("ui_ej2_sesrange"),
        shiny::checkboxGroupInput("ej2_upos","Required POS",
                                  c("NOUN","VERB","ADJ","ADV","PRON"),
                                  selected = NULL, inline = TRUE),
        shiny::hr(),
        shiny::sliderInput("ej2_npage","Examples per page",10,100,30,10),
        shiny::actionButton("btn_ej2_refresh",
          shiny::tagList(shiny::icon("rotate")," Refresh"),
          class = "btn btn-primary btn-sm w-100"),
        shiny::hr(),
        shiny::downloadButton("dl_ej2_csv","⬇ Download selected CSV",
                              class = "btn btn-outline-secondary btn-sm w-100")
      ),
      shinydashboard::box(
        title = "Examples", width = 6, status = "primary",
        style = "background:#f8f9fb",
        shiny::uiOutput("ui_ej2_summary"),
        shiny::uiOutput("ui_ej2_cards")
      ),
      shinydashboard::box(
        title = "Selected for report", width = 3, status = "success",
        style = "background:#fff;min-height:360px;max-height:700px;overflow-y:auto",
        shiny::uiOutput("ui_ej2_selected"),
        shiny::hr(),
        shiny::actionButton("btn_ej2_clear","Clear selection",
                            class = "btn btn-outline-danger btn-sm w-100")
      )
    )
  )
}

# JS needed for example selection (click on .ej2-sel-btn)
examples_v2_js <- shiny::tags$script(shiny::HTML("
$(document).on('click','.ej2-sel-btn',function(e){
  e.preventDefault();
  var id = $(this).data('id');
  Shiny.setInputValue('ej2_toggle', {id: id, ts: Date.now()}, {priority:'event'});
});
"))
