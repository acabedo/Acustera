# AcusTera

**AcusTera** is an interactive R/Shiny web application for acoustic and linguistic analysis of couple therapy sessions. It extracts prosodic features from audio recordings, runs statistical models to test hypotheses about prosodic convergence, and integrates large language models for assisted interpretation of results.

The application was developed as part of a longitudinal research project studying how patients' vocal patterns evolve and converge toward the therapist's baseline across therapy sessions.

A demo version of AcusTera — with transcription and Python-dependent modules disabled — is publicly available via ShinyLive on GitHub Pages at https://acabedo.github.io/Acustera/. A set of sample acoustic data is bundled with the demo; to load it, select the desired files and click Load selected.

---

## Research Context

The dataset included in this repository covers **four couples** (Malaga1, Malaga2, Zaragoza1, Zaragoza2) across **three therapy sessions** per couple (sessions 1, 9, and 20), yielding a longitudinal window into early, mid, and late therapeutic process.

The folder [`evaluation/`](evaluation/) contains the materials of the tool evaluation on **complete sessions** (6 couples × 6 sessions, 44.2 h, CrisperWhisper + DiariZen): the role-attribution scripts, the scripts and aggregated results of the sampling-mode evaluation, and anonymised speaker-similarity data (couples coded C1–C6, no transcripts). Segment-level acoustic data are available on request.

Two main hypotheses are operationalised in the app:

- **H1 — Relative evolution**: Patients' prosodic parameters (pitch mean, pitch SD, pitch range, intensity) change significantly across therapy sessions relative to their own session-1 baseline.
- **H2 — Convergence toward the therapist**: The Euclidean distance between each patient's prosodic vector and the therapist's baseline decreases over sessions, indicating acoustic assimilation.

### Data and Anonymisation

This repository includes:

- **Acoustic feature files** (`.csv`): one row per speech segment, containing F0, intensity, HNR, jitter, shimmer, CPP, MFCCs, and spectral features. These are phonetic measurements and do not contain any identifiable verbal content.
- **Praat TextGrid files** (`.TextGrid`): segmentation and timing information.
- **A companion `.rds` document** with the full statistical analysis (LMMs, post-hoc contrasts, effect sizes, convergence indices).

**The transcribed text of therapy participants has been intentionally omitted** to protect the anonymity of all individuals recorded. The `*_transcript.csv` files retain only segment timing and speaker role (`man`, `woman`, `therapist`); all lexical content has been removed. The morphosyntactic and linguistic analyses included in the app can be reproduced by supplying your own audio recordings.

---

## Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| R | 4.2 | 4.4+ |
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 2 GB free | 5 GB free |
| OS | macOS, Linux | macOS 13+ / Ubuntu 22.04+ |

> **macOS note**: The application includes `_mps_patch.py`, which patches PyTorch for Metal Performance Shaders (MPS) to avoid segmentation faults caused by `fork`/`multiprocessing` interactions. Several environment variables (`OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`) are set automatically at startup.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/acustera.git
cd acustera
```

### 2. Create and activate a Python virtual environment

```bash
python3 -m venv acustera-env
source acustera-env/bin/activate          # macOS / Linux
# acustera-env\Scripts\activate.bat       # Windows
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install librosa praat-parselmouth numpy scipy faster-whisper
```

> `faster-whisper` requires `ctranslate2`. On macOS with Apple Silicon, install with:
> ```bash
> pip install faster-whisper --extra-index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Install R dependencies

Open R (or RStudio) in the project directory and run:

```r
install.packages(c(
  # Core Shiny framework
  "shiny", "shinyjs", "shinydashboard", "shinydashboardPlus",

  # Data wrangling & I/O
  "tidyverse", "readr", "jsonlite", "DT",

  # Visualisation
  "plotly", "htmltools",

  # Audio
  "tuneR", "av",

  # Statistical modelling
  "lme4", "lmerTest", "emmeans", "effectsize",

  # Python bridge
  "reticulate",

  # NLP
  "udpipe",

  # Performance
  "memoise", "cachem", "future", "furrr", "digest",

  # Reporting & export
  "officer", "flextable", "base64enc",

  # HTTP (LLM calls)
  "httr"
))
```

### 5. Configure the Python environment in R

```r
library(reticulate)
use_virtualenv("acustera-env", required = TRUE)
```

Or set the path explicitly in the **Configuration** tab of the running application.

### 6. (Optional) API keys for LLM features

Create a file named `.acustera_keys.json` in the project root (it is `.gitignore`d):

```json
{
  "anthropic": "sk-ant-...",
  "openai": "sk-...",
  "huggingface": "hf_..."
}
```

The HuggingFace token is required for speaker diarisation (pyannote.audio). Anthropic/OpenAI keys are only needed for the AI-assisted report generation and speaker attribution review features.

### 7. (Optional) Alternative engines: DiariZen and CrisperWhisper

In **03 · Process** you can choose the diarization engine (pyannote 3.1 or DiariZen) and the transcription engine (Whisper or CrisperWhisper 2.0).

| Engine | Model | Strengths | Costs |
|---|---|---|---|
| pyannote 3.1 (default) | `pyannote/speaker-diarization-3.1` | In-process, light, runs on CPU/MPS; tunable thresholds and embedding model | Less accurate on long recordings with overlap and similar voices |
| DiariZen | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` | WavLM-large end-to-end model; fewer speaker confusions and better overlap handling | Heavy (GPU advised); separate environment (torch 2.8, Python 3.10) |
| Whisper (default) | faster-whisper / mlx-whisper, `large-v3-turbo` | Fast, many languages, word probabilities | Fluent output: drops fillers, repetitions and false starts |
| CrisperWhisper 2.0 | `nyralabs/CrisperWhisper2.0_large` | Verbatim: keeps fillers and disfluencies; more precise word timestamps (better segment boundaries and words/s) | Heavy (GPU advised); separate environment (Python 3.12); no word probabilities |

DiariZen and CrisperWhisper run as subprocesses in their own virtual environments (`~/.venvs/diarizen-mps`, `~/.venvs/crisperwhisper`), which AcusTera installs with [uv](https://docs.astral.sh/uv/) on first use. To use existing environments, set `ACUSTERA_DIARIZEN_PY` / `ACUSTERA_CRISPER_PY` to their Python interpreter.

**Local first.** By default every step of AcusTera runs on your own computer: no audio, transcript or result leaves it. This is the recommended setting for clinical recordings. DiariZen and CrisperWhisper also run locally, but on computers without an NVIDIA GPU (including Apple Silicon Macs, where they use the Apple GPU or the CPU) they are much slower than real time on long sessions; pyannote 3.1 + Whisper are the light alternative.

**Precomputed results (optional, not local).** The heavy engines can run on a remote GPU with the notebook `colab/acustera_colab_gpu.ipynb`. Point *Precomputed results* to the folder with its output: for each audio `<stem>`, AcusTera loads `<stem>_diarizen.json` (or any `<stem>.rttm`) and `<stem>_crisperwhisper.json` instead of running the engines, and does role assignment, acoustic extraction and statistics locally.

> **Privacy warning.** If you use Google Colab (or any cloud service), the audio recordings are uploaded to the provider's servers and the results contain the transcription of what was said. Your data are then **no longer only on your computer**. Use it only if the participants' consent, your ethics approval and your institution's data protection rules allow processing by that provider; give the files neutral names, do not save to Drive unless needed, and delete the runtime when you finish (the notebook includes a cleanup cell).

### 8. Diarization review (local)

The **Diarization review** tab measures how well the automatic diarization works on your own recordings. AcusTera cuts random excerpts (one per session quartile) and writes a WAV and a TextGrid with empty `man` / `woman` / `therapist` tiers (optionally a `words` tier from the transcription), without showing the automatic output. You annotate them in Praat and AcusTera computes, for every automatic output it finds (`<stem>_pyannote.rttm`, `<stem>_diarizen.json`, any `<stem>_<engine>.rttm`, and AcusTera's role TextGrids), the diarization error rate (DER, with optimal speaker mapping and a configurable collar) and the role attribution error. The implementation agrees with `pyannote.metrics` within 0.1 points. Everything stays on your computer.

When a transcription is available, AcusTera also computes `tempo_wps` (words per second of the segment). `speech_rate` remains the voicing rate (voiced frames per second).

---

## Running the Application

```r
library(shiny)
runApp("app.R")
```

Or from the terminal:

```bash
Rscript -e "shiny::runApp('app.R', launch.browser = TRUE)"
```

The app opens at `http://127.0.0.1:<port>` by default. The maximum upload size is set to **2 GB** to accommodate long therapy audio files.

---

## Application Structure

```
app_english/
├── app.R                   # Main application (UI + server, ~5 500 lines)
├── acoustic.py             # Python acoustic analysis module
├── _mps_patch.py           # macOS MPS compatibility patch
├── R/
│   ├── fn_cache.R          # Memoisation utilities
│   ├── fn_pos.R            # Morphosyntactic annotation (UDPipe)
│   ├── fn_ai.R             # LLM client (Anthropic / OpenAI)
│   ├── fn_report2.R        # HTML report generation
│   ├── fn_examples2.R      # Prosodic examples browser
│   ├── fn_sessions.R       # Session save / load
│   ├── fn_textgrid.R       # Praat TextGrid reader / writer
│   ├── server_new.R        # Extended server logic
│   └── ui_tabs_new.R       # Extended UI tabs
├── transcriptions/         # Pre-extracted acoustic feature CSVs + TextGrid imports
├── textgrids/              # Praat TextGrid segmentation files
├── sessions/               # Saved analysis snapshots (.rds)
├── reports/                # Generated HTML reports
└── www/audios/             # (empty) place audio files here
```

---

## Interface Sections

### 01 · Upload Audios

Entry point for new analyses. Upload one or more therapy audio files (WAV, MP3, FLAC), optionally supply reviewed Praat TextGrid files to override automatic segmentation, and load previously saved `.rds` analysis sessions. Detected files and their metadata are listed in a summary table at the bottom of the panel.

### 02 · Metadata

Inspect and correct session-level metadata before running the pipeline. Assign speaker roles (`man`, `woman`, `therapist`) to each audio track, perform mass role reassignment across files, and review the speaker profile (expected gender, role, and couple identifier). Accurate metadata is critical because the statistical models group data by role.

### 03 · Process

Configure and launch the full analysis pipeline:

- **HuggingFace token**: required for speaker diarisation.
- **Diarisation** (pyannote.audio): detects who speaks when, outputs speaker segments.
- **Transcription** (faster-whisper / Whisper): converts speech to text, writes JSON files to `transcriptions/`.
- **Acoustic feature extraction**: calls `acoustic.py` to compute all prosodic features per segment.
- **Automatic variables**: configure session numbering and couple labelling.

A progress log is displayed in real time. Parallelism is handled via the `future`/`furrr` packages (all cores minus two).

### 04 · Results

Multi-panel analysis section with the following sub-tabs:

#### Information
Lists all acoustic variables available for analysis and allows the user to select the subset to include in subsequent statistical models and plots.

#### Descriptive
Box plots and violin plots of any acoustic variable, faceted by couple, session, speaker role, or time quartile. A summary statistics table (mean ± SD, N) is displayed below the chart and can be exported to CSV or Excel.

#### Acoustic Heatmap
Z-score heatmap displaying all selected acoustic parameters simultaneously, with one row per speaker/session cell. Useful for a quick overview of which parameters and which speakers deviate most from the group mean.

#### Trajectories
Line plots of individual or group-mean prosodic trajectories across the three therapy sessions. Supports overlay of confidence intervals and faceting by couple or speaker role.

#### LMM Models
Runs **Linear Mixed Models** (package `lme4` / `lmerTest`) for each selected acoustic parameter. The default formula is:

```
parameter ~ therapy_num * speaker_role + (1 | couple/speaker_id)
```

Outputs include:
- A significance heatmap showing which parameters × effects are significant after Benjamini–Hochberg correction.
- Full coefficient table with t-values, p-values, and degrees of freedom.
- **Post-hoc contrasts** (Tukey, computed with `emmeans`) for pairwise session comparisons per role.

An embedded explanation box describes how to read mixed model output for non-specialist users.

#### Overlappings
Analyses simultaneous speech (overlapping turns). Displays the overlap ratio per session and the turn distribution by speaker role. A detailed table lists all detected overlapping segments with timestamps.

#### Convergence
Core hypothesis-testing panel:

- **H1 — Z-score per session**: normalises each speaker's parameter values relative to session 1 and plots the trajectory, testing whether change is significant via LMMs.
- **H1 — Intra-session range**: plots the within-session variability (max − min) of the prosodic parameter by session.
- **H2 — Euclidean distance to therapist**: computes the multivariate distance between each patient's mean prosodic vector and the therapist's session-1 baseline. A decreasing distance across sessions indicates convergence. The therapist's reference line is fixed at 0.
- **Convergence LMMs**: dedicated mixed models for both hypotheses with effect-size estimates (`effectsize`).

#### Prosodic Examples
Browsable gallery of speech segments at the extremes of the prosodic distribution. Filters allow selecting couple, speaker, session, acoustic parameters, direction (high/low/both), and extreme percentile threshold. Each card displays the segment text, timing, and parameter values. Segments can be selected and exported as a CSV for use in reports.

#### Transcription
Filterable table of all transcribed segments (Whisper output), showing speaker role, session, start/end times, transcribed text, and confidence score (avg log-probability). Supports filtering by couple, session, speaker, and time quartile.

> Note: in the published dataset, the text column is empty to preserve participant anonymity.

#### Segmentation
Displays the raw speaker diarisation output from pyannote.audio, showing which speaker is active at each moment. Useful for diagnosing misattributions before running the statistical pipeline.

#### Diagnostics
Plots F0 distributions by session and speaker role to validate role assignments. If a speaker labelled `therapist` has a pitch distribution inconsistent with the other sessions, this tab helps identify the problem. A **Role Fix by File** panel allows correcting individual files without reprocessing.

#### Prosodic Profile
A condensed heatmap of mean (or z-score) values for all acoustic parameters per speaker, providing a fingerprint-style view of each participant's vocal profile. Supports faceting by couple, session, or quartile.

#### Linguistics (POS)
Morphosyntactic analysis powered by the **Spanish UDPipe model** (downloaded and cached on first use, ~50 MB):

- **Annotation**: lemmatises and POS-tags all transcribed segments.
- **Lemma cloud**: word cloud with optional faceting by couple, session, quartile, or speaker.
- **POS distribution**: stacked bar chart of part-of-speech categories by speaker and time quartile.
- **Lexical markers**: frequency of pragmatically relevant categories per speaker — personal pronouns (1st/2nd/3rd person), possessives, demonstratives, negation, intensifiers, hedges, adversatives, and positive/negative emotion words. Values are expressed as percentage of the speaker's total tokens.
- **Lexical metrics**: table of TTR (Type-Token Ratio), lexical density, and Yule's K per speaker/session.

> Note: linguistic analysis requires transcribed text. In the published dataset the text has been removed; this tab is functional when the app is used with new recordings.

#### AI Speakers Review
AI-assisted speaker attribution auditing. Two modes are available:

1. **Fast heuristic**: flags turns where the acoustic features (F0, lexicon) are inconsistent with the assigned role.
2. **Full AI review**: sends all segments with ± N turns of surrounding context to the selected LLM (Anthropic Claude or OpenAI) for content-based attribution verification.

Proposed corrections are shown in a table and can be applied in bulk with a single click. Estimated cost per 100 turns is displayed in the interface.

### Methods

Reference panel documenting the acoustic analysis pipeline: which Python library extracts each feature, the mathematical definition of each measure, and interpretation guidelines. Intended for readers unfamiliar with acoustic phonetics.

### 05 · Report

Generates a self-contained interactive HTML report summarising the analysis. Options include:

- Selecting which sections and plots to include.
- Generating an **AI narrative** (via Anthropic Claude or OpenAI) that interprets the statistical findings in clinical language. The LLM receives a structured JSON context block, not raw audio or transcripts.
- A **Result block for Claude** panel shows the JSON payload that will be sent to the model, allowing manual inspection before submission.

### Configuration

- **LMM model**: choose the random-effects structure and fixed-effects covariates.
- **Python environment**: specify the virtualenv path and test the Python bridge.
- **System status**: shows package versions, Python version, GPU availability.
- **API keys**: enter or update Anthropic and OpenAI keys at runtime (stored in `.acustera_keys.json`, never committed to the repository).
- **Cache and performance**: displays the current memoise cache size and allows flushing it. Cache TTL is 30 days; maximum size is 500 MB.

---

## Acoustic Features Extracted

Each speech segment yields the following measurements:

| Feature | Description |
|---------|-------------|
| `f0_mean` | Mean fundamental frequency (Hz) |
| `f0_sd` | Standard deviation of F0 |
| `f0_range` | F0 range (max − min) |
| `speech_rate` | Voiced frames per second (proxy for speech rate) |
| `intensity_mean` | Mean RMS energy (dB) |
| `intensity_sd` | SD of RMS energy |
| `hnr` | Harmonics-to-Noise Ratio (voice quality) |
| `jitter_local` | Local jitter — cycle-to-cycle F0 variation |
| `shimmer_local` | Local shimmer — cycle-to-cycle amplitude variation |
| `cpp` | Cepstral Peak Prominence (voice clarity) |
| `mfcc1`–`mfcc13` | Mel-Frequency Cepstral Coefficients 1–13 |
| `spectral_centroid` | Spectral centre of mass (brightness) |
| `zcr` | Zero-crossing rate (noisiness proxy) |
| `rms` | Root mean square amplitude |

F0, jitter, shimmer, and HNR are extracted via **Praat** (parselmouth); spectral and MFCC features via **librosa**.

---

## Key Dependencies

### R Packages

| Package | Role |
|---------|------|
| `shiny`, `shinydashboard`, `shinydashboardPlus` | Web application framework |
| `reticulate` | R–Python bridge |
| `lme4`, `lmerTest` | Linear mixed-effects models |
| `emmeans` | Post-hoc contrasts |
| `effectsize` | Cohen's d, partial η² |
| `plotly` | Interactive visualisations |
| `DT` | Interactive data tables with export |
| `udpipe` | NLP morphosyntactic annotation |
| `memoise`, `cachem` | LMM result caching |
| `future`, `furrr` | Parallel feature extraction |
| `tuneR`, `av` | Audio file I/O |
| `httr`, `jsonlite` | LLM API calls |
| `officer`, `flextable` | Report export (DOCX) |

### Python Packages

| Package | Role |
|---------|------|
| `librosa` | Spectral features, MFCCs, RMS, ZCR |
| `praat-parselmouth` | F0, jitter, shimmer, HNR, CPP |
| `numpy`, `scipy` | Numerical routines |
| `faster-whisper` | Automatic speech recognition (transcription) |
| `pyannote.audio` | Speaker diarisation (requires HuggingFace token) |

---

## Project Layout: Data Files

```
transcriptions/
├── Malaga1_therapy1_features.csv     # Acoustic features, session 1
├── Malaga1_therapy9_features.csv
├── Malaga1_therapy20_features.csv
├── Malaga1_therapy1_transcript.csv   # Timing + speaker role (no text)
├── Malaga1_therapy1_overlaps.csv     # Overlapping speech markers
├── ...                               # Same pattern for Malaga2, Zaragoza1, Zaragoza2
└── *.json                            # Intermediate Whisper output (~180 files)

textgrids/
├── Malaga1_therapy1.TextGrid
├── Malaga1_therapy9.TextGrid
├── Malaga1_therapy20.TextGrid
└── ...                               # 12 files total (4 couples × 3 sessions)

sessions/
└── *.rds                             # Saved analysis snapshots (loadable from tab 01)
```

Feature CSVs contain the following columns:
`couple`, `therapy`, `therapy_num`, `speaker`, `speaker_id`, `seg_start`, `seg_end`, `is_overlap`, `quantile`, `duration`, `n_words`, `avg_logprob`, `pause_pre_total`, `f0_mean`, `f0_sd`, `f0_range`, `speech_rate`, `intensity_mean`, `intensity_sd`, `hnr`, `jitter_local`, `shimmer_local`, `cpp`, `mfcc1`–`mfcc13`, `spectral_centroid`, `zcr`, `rms`.

---

## Citation

If you use AcusTera or the associated dataset in published research, please cite:

> [Citation to be added upon publication]

---

## Licence

The application code is released under the MIT Licence. The acoustic feature data are released for research use only. Audio recordings and session transcripts are not distributed in this repository.
