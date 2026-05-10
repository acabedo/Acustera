options(shiny.maxRequestSize = 2000 * 1024^2)  # 2 GB max per file

# Avoid double loading of libomp (torch + librosa + pyannote)
Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")

# UTF-8 throughout the pipeline (logs, messages, HTML, files)
Sys.setenv(LANG = "en_GB.UTF-8", LC_ALL = "en_GB.UTF-8")
tryCatch({
  Sys.setlocale("LC_CTYPE",    "en_GB.UTF-8")
  Sys.setlocale("LC_COLLATE",  "en_GB.UTF-8")
  Sys.setlocale("LC_MESSAGES", "en_GB.UTF-8")
}, warning = function(w) invisible(NULL),
   error   = function(e) invisible(NULL))
options(encoding = "UTF-8")
suppressPackageStartupMessages({
  library(shiny)
  library(shinyjs)
  library(reticulate)
  library(tuneR)
  library(tidyverse)
  library(DT)
  library(plotly)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(effectsize)
  library(htmltools)
  library(shinydashboard)
  library(shinydashboardPlus)
  library(jsonlite)
  if (requireNamespace("av", quietly=TRUE)) library(av)
  if (requireNamespace("base64enc", quietly=TRUE)) library(base64enc)
  if (requireNamespace("av",        quietly=TRUE)) library(av)
  if (requireNamespace("officer", quietly=TRUE)) library(officer)
  if (requireNamespace("flextable", quietly=TRUE)) library(flextable)
})

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0) a else b

# Helper: DataTable with CSV/Excel buttons that download the ENTIRE dataset
# (not just the visible page). Use for analytical results tables.
dt_analysis <- function(df, ...) {
  DT::datatable(df, extensions = "Buttons",
    options = list(pageLength = 15, dom = "Blfrtip",
      buttons = list(
        "copy",
        list(extend = "csv",   text = "⬇ Full CSV",
             exportOptions = list(modifier = list(search = "none"))),
        list(extend = "excel", text = "⬇ Excel",
             exportOptions = list(modifier = list(search = "none")))
      ),
      scrollX = TRUE),
    rownames = FALSE, ...)
}

# ═════════════════════════════════════════════════════════════
# EXTENDED MODULES (R/) — cache, POS, AI, examples v2, report v2
# ══════════════════════════════════════════════════════════════
suppressPackageStartupMessages({
  if (!requireNamespace("digest",   quietly=TRUE)) install.packages("digest")
  if (requireNamespace("readr",     quietly=TRUE)) library(readr)
  if (requireNamespace("httr",      quietly=TRUE)) library(httr)
  if (requireNamespace("udpipe",    quietly=TRUE)) library(udpipe)
  if (requireNamespace("memoise",   quietly=TRUE)) library(memoise)
  if (requireNamespace("cachem",    quietly=TRUE)) library(cachem)
  if (requireNamespace("future",    quietly=TRUE)) library(future)
  if (requireNamespace("furrr",     quietly=TRUE)) library(furrr)
})
for (.f in c("R/fn_cache.R","R/fn_pos.R","R/fn_ai.R","R/fn_sessions.R",
             "R/fn_textgrid.R",
             "R/fn_examples2.R","R/fn_report2.R",
             "R/ui_tabs_new.R","R/server_new.R")) {
  if (file.exists(.f)) source(.f, local = FALSE, encoding = "UTF-8")
}

# Optional parallel plan (feature extraction)
if (requireNamespace("future", quietly=TRUE)) {
  tryCatch(future::plan(future::multisession,
    workers = max(1, parallel::detectCores() - 2)),
    error = function(e) invisible(NULL))
}

# Cache LMM if memoise is available: the actual body is defined below.

# Activate the wrapper at the end of the file (after defining run_lmm).
.enable_lmm_cache <- function() {
  if (exists("run_lmm", envir = .GlobalEnv) &&
      exists("memo",    envir = .GlobalEnv)) {
    orig <- get("run_lmm", envir = .GlobalEnv)
    if (!isTRUE(attr(orig, "cached"))) {
      wrapped <- memo(orig)
      attr(wrapped, "cached") <- TRUE
      assign("run_lmm", wrapped, envir = .GlobalEnv)
    }
  }
}

# ═════════════════════════════════════════════════════════════
# SECTION 1 — PYTHON
# All Python code is here as a string, written to disk the first time, and loaded with py_run_file().

# disk the first time and loaded with py_run_file(). 
# # ══════════════════════════════════════════════════════════════

ACOUSTIC_PY <- r"(
# AcusTera acoustic module v2 — jitter, shimmer, CPP added
import numpy as np, warnings
warnings.filterwarnings("ignore")
_lib = None
_pm  = None

def _librosa():
    global _lib
    if _lib is None:
        import librosa as l; _lib = l
    return _lib

def _parsel():
    global _pm
    if _pm is None:
        import parselmouth as p; _pm = p
    return _pm

def extract_segment(path, start, end, n_mfcc=13):
    lib = _librosa()
    dur = max(end - start, 0.1)
    y, sr = lib.load(path, offset=float(start), duration=dur,
                     sr=None, mono=True)
    if len(y) < sr * 0.5:
        return _empty(n_mfcc)
    out = {"duration": dur}

    # F0 + speech_rate via Praat
    try:
        snd = _parsel().Sound(y, sr)
        f0  = snd.to_pitch().selected_array["frequency"]
        f0  = f0[f0 > 0]
        out["f0_mean"]     = float(np.mean(f0))   if len(f0) > 3 else float("nan")
        out["f0_sd"]       = float(np.std(f0))    if len(f0) > 3 else float("nan")
        out["f0_range"]    = float(np.ptp(f0))    if len(f0) > 3 else float("nan")
        out["speech_rate"] = float(len(f0) / dur)
    except:
        out.update({"f0_mean": float("nan"), "f0_sd": float("nan"),
                    "f0_range": float("nan"), "speech_rate": float("nan")})

    # Intensity via Praat
    try:
        iv = _parsel().Sound(y, sr).to_intensity().values[0]
        iv = iv[iv > 0]
        out["intensity_mean"] = float(np.mean(iv)) if len(iv) else float("nan")
        out["intensity_sd"]   = float(np.std(iv))  if len(iv) else float("nan")
    except:
        out["intensity_mean"] = out["intensity_sd"] = float("nan")

    # HNR via Praat
    try:
        hv = _parsel().Sound(y, sr).to_harmonicity().values[0]
        hv = hv[hv > -200]
        out["hnr"] = float(np.mean(hv)) if len(hv) else float("nan")
    except:
        out["hnr"] = float("nan")

    # Local jitter: variation of consecutive F0 periods (pure numpy)
    # Definition: mean(|T[i+1]-T[i]|) / mean(T[i]) (simplified RAP)
    try:
        pitch_obj = _parsel().Sound(y, sr).to_pitch()
        periods = pitch_obj.selected_array["frequency"]
        periods = periods[periods > 0]
        if len(periods) > 3:
            period_s = 1.0 / periods
            diffs = np.abs(np.diff(period_s))
            out["jitter_local"] = float(np.mean(diffs) / np.mean(period_s))
        else:
            out["jitter_local"] = float("nan")
    except Exception:
        out["jitter_local"] = float("nan")

    # Local shimmer: variation in the amplitude of sound peaks (pure NumPy)
    # Aproximacion via energia RMS en ventanas de pitch period
    try:
        pitch_obj2 = _parsel().Sound(y, sr).to_pitch()
        f0_arr = pitch_obj2.selected_array["frequency"]
        f0_arr = f0_arr[f0_arr > 0]
        if len(f0_arr) > 3:
            mean_f0 = float(np.mean(f0_arr))
            win = max(1, int(sr / mean_f0))
            amps = np.array([float(np.sqrt(np.mean(y[i:i+win]**2)))
                             for i in range(0, len(y)-win, win//2)])
            amps = amps[amps > 1e-8]
            if len(amps) > 3:
                diffs_a = np.abs(np.diff(amps))
                out["shimmer_local"] = float(np.mean(diffs_a) / np.mean(amps))
            else:
                out["shimmer_local"] = float("nan")
        else:
            out["shimmer_local"] = float("nan")
    except Exception:
        out["shimmer_local"] = float("nan")

    # CPP via librosa cepstrum (Cepstral Peak Prominence proxy)
    # High values ​​= clear voice; low values ​​= strained or crackling voice
    try:
        import numpy as _np2
        n_fft = min(2048, len(y))
        D  = np.abs(np.fft.rfft(y, n=n_fft))
        D  = _np2.where(D > 0, D, 1e-10)
        ceps = np.real(np.fft.irfft(np.log(D)))
        sr_int = int(sr)
        lo = int(sr_int / 500)
        hi = int(sr_int / 60)
        hi = min(hi, len(ceps) - 1)
        if hi > lo:
            peak = float(np.max(np.abs(ceps[lo:hi])))
            out["cpp"] = float(20 * np.log10(peak + 1e-10))
        else:
            out["cpp"] = float("nan")
    except Exception:
        out["cpp"] = float("nan")

    # MFCCs
    try:
        m = lib.feature.mfcc(y=y, sr=sr, n_mfcc=int(n_mfcc))
        for i in range(int(n_mfcc)):
            out[f"mfcc{i+1}"] = float(np.mean(m[i]))
    except:
        for i in range(int(n_mfcc)): out[f"mfcc{i+1}"] = float("nan")

    # Espectro
    try:
        out["spectral_centroid"] = float(np.mean(lib.feature.spectral_centroid(y=y, sr=sr)))
        out["zcr"]               = float(np.mean(lib.feature.zero_crossing_rate(y)))
        out["rms"]               = float(np.mean(lib.feature.rms(y=y)))
    except:
        out["spectral_centroid"] = out["zcr"] = out["rms"] = float("nan")

    return out

def embedding(path, start, end):
    lib = _librosa()
    dur = max(end - start, 0.1)
    try:
        y, sr = lib.load(path, offset=float(start), duration=dur,
                         sr=None, mono=True)
        m = lib.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        v = np.concatenate([np.mean(m, axis=1), np.std(m, axis=1)])
        return v.tolist()
    except:
        return [0.0] * 26

def cosine_sim(a, b):
    va, vb = np.array(a), np.array(b)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0

def _empty(n_mfcc=13):
    d = {k: float("nan") for k in [
        "duration","f0_mean","f0_sd","f0_range","speech_rate",
        "intensity_mean","intensity_sd","hnr",
        "jitter_local","shimmer_local","cpp",
        "spectral_centroid","zcr","rms"]}
    for i in range(int(n_mfcc)): d[f"mfcc{i+1}"] = float("nan")
    return d
)"

# Write acoustic.py next to app.R (overwrite if the module has changed)
.write_acoustic_py <- function() {
  path    <- file.path(getwd(), "acoustic.py")
  current <- if (file.exists(path)) readLines(path, n=1) else ""
  if (!startsWith(current, "# AcusTera acoustic module"))
    writeLines(ACOUSTIC_PY, path)
  path
}

# Configure virtualenv and import the module
setup_python <- function(log = message) {
  Sys.setenv(
    OMP_NUM_THREADS             = "1",
    MKL_NUM_THREADS             = "1",
    OPENBLAS_NUM_THREADS        = "1",
    VECLIB_MAXIMUM_THREADS      = "1",
    NUMEXPR_NUM_THREADS         = "1",
    TOKENIZERS_PARALLELISM      = "false",
    PYTORCH_ENABLE_MPS_FALLBACK = "1",
    KMP_DUPLICATE_LIB_OK        = "TRUE"
  )
venv <- "acustera-env" 
if (!virtualenv_exists(venv)) { 
log("Creating Python virtual environment...") 
virtualenv_create(venv) 
log("Installing librosa and parselmouth (~2 min)...") 
virtualenv_install(venv, 
c("librosa", "praat-parselmouth", "numpy", "scipy", 
"faster-whisper"), 
ignore_installed = FALSE) 
log("✓ Python dependencies installed.") 
}
  use_virtualenv(venv, required = TRUE)

# ── Critical anti-segfault patch (macOS MPS + fork) ─────────────────────
# Must be run BEFORE any imports of librosa/torch/pyannote.
# Replaces DataLoader globally so that it never launches workers with fork().
# Write the anti-segfault patch to disk and run it.
  .patch_py <- file.path(getwd(), "_mps_patch.py")
  writeLines(c(
    "import sys",
    "if sys.platform == 'darwin':",
    "    try:",
    "        import torch.utils.data as tud",
    "        _Orig = tud.DataLoader",
    "        class _SafeDL(_Orig):",
    "            def __init__(self, *a, **kw):",
    "                kw['num_workers'] = 0",
    "                kw.pop('prefetch_factor', None)",
    "                kw.pop('persistent_workers', None)",
    "                super().__init__(*a, **kw)",
    "        tud.DataLoader = _SafeDL",
    "        import torch",
    "        try:",
    "            torch.set_num_threads(1)",
    "        except RuntimeError:",
    "            pass",
    "        try:",
    "            torch.set_num_interop_threads(1)",
    "        except RuntimeError:",
    "            pass",
    "    except Exception:",
    "        pass",
    "    import multiprocessing as mp",
    "    try:",
    "        mp.set_start_method('spawn', force=True)",
    "    except RuntimeError:",
    "        pass"
  ), .patch_py)
  py_run_file(.patch_py)

  py_run_file(.write_acoustic_py())
  log(paste("✓ Python:", py_config()$python))
}

# Wrappers R → Python
py_features <- function(path, start, end, n_mfcc = 13L, log = NULL) {
  r <- tryCatch(
    py$extract_segment(path, start, end, as.integer(n_mfcc)),
    error = function(e) {
      msg <- conditionMessage(e)
      if (!is.null(log)) log(paste0("[py_features ERROR] ", msg))
      else message("[py_features ERROR] ", msg)
      NULL
    })
  if (is.null(r)) return(NULL)
  as.data.frame(lapply(r, function(x)
    if (is.null(x) || (length(x)==1 && is.nan(x))) NA_real_ else x))
}

py_embedding <- function(path, start, end) {
  tryCatch(unlist(py$embedding(path, start, end)), error = \(e) rep(0,26))
}

py_cosine <- function(a, b) {
  tryCatch(py$cosine_sim(as.list(a), as.list(b)), error = \(e) 0)
}

# ═════════════════════════════════════════════════════════════
# SECTION 2 — DIARIZATION (pyannote via reticulate)
# ═════════════════════════════════════════════════════════════

# ensure_pyannote: Installs Pyannote if it's not in the virtual environment.
ensure_pyannote <- function(log = message) {
  ok <- tryCatch({ py_run_string("import pyannote.audio"); TRUE },
                 error = \(e) FALSE)
  if (!ok) {
    log("Installing pyannote.audio...")
    virtualenv_install("acustera-env",
      c("pyannote.audio", "torch", "torchaudio"),
      ignore_installed = FALSE)
    log("✓ pyannote installed.")
  }
  # Detect huggingface_hub / pyannote version incompatibility:
  # pyannote 2.x uses use_auth_token; huggingface_hub >= 0.20 removed it.
  # Only pin huggingface_hub when pyannote 2.x is installed AND hf_hub >= 0.20.
  compat <- tryCatch({
    py_run_string(paste(
      "import pyannote.audio as _pa, huggingface_hub as _hf",
      "_pa_maj = int(_pa.__version__.split('.')[0])",
      "_hf_parts = [int(x) for x in _hf.__version__.split('.')[:2]]",
      "assert not (_pa_maj < 3 and _hf_parts >= [0, 20])",
      sep = "\n"))
    TRUE
  }, error = \(e) FALSE)
  if (!compat) {
    log("Detected huggingface_hub incompatibility with pyannote 2.x — pinning to 0.19.4 ...")
    virtualenv_install("acustera-env", "huggingface_hub==0.19.4",
                       ignore_installed = TRUE)
    log("✓ huggingface_hub pinned.")
  }
}

# init_pyannote_pipelines: Loads both pipelines ONCE on the optimal device.

# Priority: MPS (Apple Silicon) → CUDA (Nvidia) → CPU.

# The pipelines are stored as global variables in the Python environment (_dpipe, _opipe).
init_pyannote_pipelines <- function(hf_token, embedding_model = 'microsoft/wavlm-base-plus-sv', log = message) {
  py$`_hf_token`      <- hf_token
  py$`_emb_model_name` <- embedding_model

  py_run_string('
import warnings
warnings.filterwarnings("ignore")
import torch
from pyannote.audio import Pipeline as _P
from pyannote.audio import Model

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    _dev     = torch.device("mps")
    _emb_dev = torch.device("cpu")
elif torch.cuda.is_available():
    _dev = _emb_dev = torch.device("cuda")
else:
    _dev = _emb_dev = torch.device("cpu")

def _pp_from_pretrained(cls, name, tok):
    # pyannote 3.x usa use_auth_token; pyannote 4.x usa token. Probar ambos.
    try:
        return cls.from_pretrained(name, use_auth_token=tok)
    except TypeError:
        return cls.from_pretrained(name, token=tok)

if "_dpipe" not in dir() or _emb_model_name != getattr(_dpipe, "_emb_model_name_used", ""):
    import traceback as _tb, sys as _sys
    try:
        _dpipe = _pp_from_pretrained(_P, "pyannote/speaker-diarization-3.1", _hf_token)
    except Exception as _e:
        print("=== FULL TRACEBACK DIARIZATION ===", flush=True)
        _tb.print_exc(file=_sys.stdout); _sys.stdout.flush()
        raise
    # Attempt to override the embedding. If it fails (e.g., the model is not
    # a native pyannote checkpoint, such as wavlm-base-plus-sv), is left
    # the default embedding of the pipeline (wespeaker).
    _emb_override_ok = False
    if _emb_model_name and "wespeaker" not in _emb_model_name.lower():
        try:
            _emb_model = _pp_from_pretrained(Model, _emb_model_name, _hf_token).to(_emb_dev)
            _dpipe.embedding = _emb_model
            _dpipe.embedding_batch_size = 1
            _dpipe.embedding_exclude_overlap = True
            _emb_override_ok = True
            print(f"✓ Embedding override aplicado: {_emb_model_name}", flush=True)
        except Exception as _e:
            print("WARN: failed to load " + str(_emb_model_name) +
                  " as a pyannote model; using embedding by default (wespeaker). " +
                  "Explanation: " + type(_e).__name__ + ": " + str(_e),
                  flush=True)
    _dpipe = _dpipe.to(_dev)
    _dpipe._emb_model_name_used = _emb_model_name
    _dpipe._emb_override_ok = _emb_override_ok

_dev_name = str(_dev)
')
  dev <- py$`_dev_name`
  log(sprintf("  ✓ pyannote ready at: %s", dev))
  dev
}

# diarize: uses the already initialized _dpipe, only infers the information.
# Supports both the old (itertracks) and new (DiarizeOutput) APIs
diarize <- function(path, n_speakers = 3L,
                   clustering_threshold   = 0.7,
                   segmentation_threshold = 0.5,
                   log = message) {
  log(paste("  Diarying:", basename(path)))
  # Apply thresholds before inference
  py_run_string(sprintf('
_dpipe.segmentation_onset  = %f
_dpipe.segmentation_offset = %f
try:
    _dpipe._clustering.threshold = %f
except Exception:
    pass
', clustering_threshold, segmentation_threshold - 0.1, clustering_threshold))
  py_run_string(sprintf('
_dr = _dpipe("%s", num_speakers=%d)
_segs = []

def _iter_annotation(ann):
    out = []
    try:
        for t, _, s in ann.itertracks(yield_label=True):
            out.append({"start": t.start, "end": t.end, "speaker": s})
    except Exception as _e:
        print(f"itertracks failed: {_e}")
    return out

if hasattr(_dr, "itertracks"):
    _segs = _iter_annotation(_dr)
else:
    _ann = None
    for _attr in ("speaker_diarization", "diarization", "annotation"):
        if hasattr(_dr, _attr):
            _ann = getattr(_dr, _attr)
            break
    if _ann is None:
        for _attr in (a for a in dir(_dr) if not a.startswith("_")):
            _candidate = getattr(_dr, _attr, None)
            if hasattr(_candidate, "itertracks"):
                _ann = _candidate
                print(f"Annotation encontrada en: {_attr}")
                break
    if _ann is not None:
        _segs = _iter_annotation(_ann)
    else:
        print(f"ERROR: no Annotation. Atributos: {[a for a in dir(_dr) if not a.startswith(chr(95))]}")
', path, as.integer(n_speakers)))
  segs <- py$`_segs`
  if (length(segs) == 0) return(tibble())
  bind_rows(lapply(segs, as.data.frame)) |> mutate(duration = end - start)
}

detect_overlaps <- function(segs, log = message) {
  log("  Detecting overlaps...")
  if (nrow(segs) < 2) return(tibble(ov_start=double(), ov_end=double(), ov_dur=double()))
  segs <- arrange(segs, start)
  ovlps <- list()
  for (i in seq_len(nrow(segs) - 1)) {
    a <- segs[i, ]
    for (j in (i+1):nrow(segs)) {
      b <- segs[j, ]
      if (b$start >= a$end) break
      ov_start <- max(a$start, b$start)
      ov_end   <- min(a$end,   b$end)
      if (ov_end > ov_start && a$speaker != b$speaker)
        ovlps[[length(ovlps)+1]] <- list(ov_start=ov_start, ov_end=ov_end,
                                         ov_dur=ov_end - ov_start)
    }
  }
  if (length(ovlps) == 0)
    return(tibble(ov_start=double(), ov_end=double(), ov_dur=double()))
  bind_rows(lapply(ovlps, as.data.frame))
}

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 3 — ROLE ASSIGNMENT
# ══════════════════════════════════════════════════════════════

assign_roles <- function(path, segs, ref_emb = NULL, log = message) {
  speakers <- unique(segs$speaker)
  roles    <- setNames(rep(NA_character_, length(speakers)), speakers)

  # Embedding the longest segment per speaker
  embs <- lapply(speakers, function(sp) {
    best <- segs |> filter(speaker == sp, duration >= 2) |>
      slice_max(duration, n=1, with_ties=FALSE)
    if (nrow(best) == 0) return(rep(0, 26))
    py_embedding(path, best$start, best$end)
  }) |> setNames(speakers)

  # Step 1: Therapist by cosine similarity with reference
  if (!is.null(ref_emb) && length(ref_emb) == 26) {
    sims <- sapply(embs, \(e) py_cosine(e, ref_emb))
    log(sprintf("    Similitudes: %s",
      paste(sprintf("%s=%.2f", names(sims), sims), collapse="  ")))
    ter <- names(which.max(sims))
    roles[ter] <- "therapist"
    speakers   <- setdiff(speakers, ter)
  }

  # Step 2: male/female by F0 means — only speakers without a role yet
  remaining <- names(roles[is.na(roles)])
  if (length(remaining) >= 1) {
    f0s <- sapply(remaining, function(sp) {
      top3 <- segs |> filter(speaker==sp, duration>=2) |>
        slice_max(duration, n=3, with_ties=FALSE)
      vals <- sapply(seq_len(nrow(top3)), function(i) {
        f <- py_features(path, top3$start[i], top3$end[i], 0L, log)
        if (is.null(f)) NA_real_ else f$f0_mean %||% NA_real_
      })
      mean(vals, na.rm=TRUE)
    })
    valid <- f0s[!is.na(f0s)]
    log(sprintf("    F0: %s",
      paste(sprintf("%s=%.0fHz", names(valid), valid), collapse="  ")))
    if (length(valid) >= 2) {
      sorted <- names(sort(valid))
      roles[sorted[1]]              <- "man"
      roles[sorted[length(sorted)]] <- "woman"
      if (length(sorted) >= 3)
        for (sp in sorted[-c(1, length(sorted))])
          if (is.na(roles[sp])) roles[sp] <- "therapist"
    } else if (length(valid) == 1) {
      sp <- names(valid)
      roles[sp] <- if (valid < 165) "man" else "woman"
    }
  }

  roles[is.na(roles)] <- "unknown"
  log(sprintf("    Roles: %s",
    paste(sprintf("%s→%s", names(roles), roles), collapse="  ")))
  roles
}

# ══════════════════════════════════════════════════════════════
# SECTION 4 — PIPELINE PER SESSION
# ══════════════════════════════════════════════════════════════

process_session <- function(path, couple, therapy, therapy_num,
                             n_speakers, ref_emb,
                             min_seg, n_mfcc, do_overlap, do_quantiles,
                             do_transcribe = FALSE,
                             language = "es",
                             orig_name = NULL,
                             sample_mins = 0,
                             do_pauses = TRUE,
                             clustering_threshold   = 0.7,
                             segmentation_threshold = 0.5,
                             textgrid_path = NULL,
                             log = message) {
  # ── Total audio duration ─────────────────────────────
  total_dur <- tryCatch({
    h <- readWave(path, header=TRUE)
    h$samples / h$sample.rate
  }, error = \(e) {
    # Fallback: use ffprobe if available
    tryCatch({
      out <- system2("ffprobe", c("-v","quiet","-show_entries",
        "format=duration","-of","csv=p=0", path),
        stdout=TRUE, stderr=FALSE)
      as.numeric(trimws(out[1]))
    }, error = \(e2) 3600)  # use 1h if can not be read
  })

  sample_secs     <- sample_mins * 60
  quantile_bounds <- total_dur * c(0, .25, .5, .75, 1)

# ── Revised TextGrid Mode ────────────────────────────────
# If the researcher provided a TextGrid, we use it as the reference truth:
# We skip diary collection and transcription and go directly to the acoustic computation.
  use_textgrid <- !is.null(textgrid_path) && file.exists(textgrid_path)
  if (use_textgrid) {
    log(sprintf("  Using reviewed TextGrid: %s", basename(textgrid_path)))
    tg <- tryCatch(read_textgrid(textgrid_path), error = function(e) NULL)
    if (is.null(tg) || nrow(tg) == 0) {
      log("  Warning: TextGrid empty or corrupted; following automatic dyaring.")
      use_textgrid <- FALSE
    } else {
      # Adapt to the scheme expected by the rest of the pipeline
      segs <- dplyr::transmute(tg,
        start   = start, end = end, duration = duration,
        speaker = speaker, speaker_id = speaker,
        text   = text,
        n_words  = lengths(strsplit(text, "\\s+")),
        avg_logprob = NA_real_)
      segs <- segs[segs$duration > 0, , drop = FALSE]
      # There are no overlaps derivable from the TextGrid itself; we leave it empty.
      ovlps <- tibble(ov_start=double(), ov_end=double(), ov_dur=double())
      log(sprintf("  ✓ %d TextGrid intervals.", nrow(segs)))
    }
  }

  if (use_textgrid) {
    segs_pyannote <- segs
    # is_in_sample is always TRUE (there is no sample mode with manual TextGrid)
    is_in_sample <- function(start, end) TRUE
    log("  Extracting acoustic parameters over the intervals of the TextGrid...")
    rows <- vector("list", nrow(segs))
    for (i in seq_len(nrow(segs))) {
      s <- segs[i,]
      if (s$duration < min_seg) next
      feat <- py_features(path, s$start, s$end, as.integer(n_mfcc), log)
      if (is.null(feat)) next
      mid    <- (s$start + s$end) / 2
      quantile <- if (do_quantiles)
        as.character(cut(mid/total_dur, c(0,.25,.5,.75,1),
                         c("Q1","Q2","Q3","Q4"), include.lowest=TRUE))
      else NA_character_
      rows[[i]] <- bind_cols(
        tibble(couple=couple, therapy=therapy, therapy_num=therapy_num,
               speaker=s$speaker, speaker_id=s$speaker_id %||% s$speaker,
               seg_start=round(s$start,3), seg_end=round(s$end,3),
               is_overlap=FALSE, quantile=quantile),
        feat,
        tibble(text = s$text, n_words = s$n_words,
               avg_logprob = s$avg_logprob))
    }
    features <- bind_rows(compact(rows))
    if (nrow(features) == 0)
      log("  WARNING: 0 segments with features — py_features() returned NULL for all segments. Check Python errors above.")
    else
      log(sprintf("  ✓ %d segments with features (from TextGrid).", nrow(features)))

    # Simplified pauses and ov_sum; skip to the final block.
    if (do_pauses && nrow(features) > 0) {
      features <- features |>
        arrange(seg_start) |>
        group_by(speaker) |>
        mutate(pause_pre_total  = round(seg_start - lag(seg_end, default=NA_real_), 3),
               pause_pre_clean = pause_pre_total) |>
        ungroup()
    } else {
      features <- features |> mutate(pause_pre_total=NA_real_,
                                      pause_pre_clean=NA_real_)
    }

    ov_sum <- tibble(
      couple=couple, therapy=therapy, therapy_num=therapy_num,
      dur_total_s=round(total_dur,1), n_overlappings=0L,
      dur_overlap_s=0, ratio_overlap=0,
      turns_man    = sum(segs$speaker=="man"),
      turns_woman     = sum(segs$speaker=="woman"),
      turns_therapist = sum(segs$speaker=="therapist"),
      dur_man_s = round(sum(segs$duration[segs$speaker=="man"]),1),
      dur_woman_s  = round(sum(segs$duration[segs$speaker=="woman"]),1),
      dur_terap_s  = round(sum(segs$duration[segs$speaker=="therapist"]),1))

    # Save CSVs and TextGrid (for traceability; we re-export the one we used)
    out_dir <- file.path(getwd(), "transcriptions")
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive=TRUE)
    file_stem <- tools::file_path_sans_ext(orig_name %||% basename(path))
    write_csv(features, file.path(out_dir, paste0(file_stem, "_features.csv")))
    write_csv(ov_sum,   file.path(out_dir, paste0(file_stem, "_overlaps.csv")))
    tg_dir <- file.path(getwd(), "textgrids")
    if (!dir.exists(tg_dir)) dir.create(tg_dir, recursive=TRUE)
    file.copy(textgrid_path,
              file.path(tg_dir, paste0(file_stem, ".TextGrid")),
              overwrite = TRUE)
    log(sprintf("  ✓ CSVs stored (TextGrid mode)."))
    return(list(features=features, overlaps=ov_sum,
                transcript=tibble(), segs=segs_pyannote))
  }

# ── Diarization: Chunks or Full Audio ─────────────────
# If sample_mins > 0, extract 4 chunks with ffmpeg and diarize
# each one separately — much faster than diarizing 50 min.
# Timestamps are shifted to the actual value in the original audio.
  if (sample_secs > 0) {
    log(sprintf("  Sample mode: first %.0f min per quantile (%.0f min total).",
                sample_mins, sample_mins * 4))

    tmp_chunks <- character(4)
    segs_all   <- list()

    for (q in 1:4) {
      q_start <- quantile_bounds[q]
      chunk_f  <- tempfile(fileext=".wav")
      tmp_chunks[q] <- chunk_f

      # Extract chunk with ffmpeg
      ret <- system2("ffmpeg", c(
        "-y", "-ss", sprintf("%.3f", q_start),
        "-t",  sprintf("%.3f", sample_secs),
        "-i",  path,
        "-ac", "1", "-ar", "16000",
        "-loglevel", "error",
        chunk_f), stdout=FALSE, stderr=FALSE)

      if (!file.exists(chunk_f) || file.size(chunk_f) < 1000) {
        log(sprintf("  Warning: chunk could not be extracted Q%d", q))
        next
      }

      # To diary chunk
      chunk_segs <- tryCatch(
        diarize(chunk_f, n_speakers,
                clustering_threshold   = clustering_threshold,
                segmentation_threshold = segmentation_threshold,
                log = log),
        error = \(e) tibble()
      )

      if (nrow(chunk_segs) > 0) {
        # Shift timestamps to the actual value in the original audio
        chunk_segs <- chunk_segs |>
          mutate(start = round(start + q_start, 3),
                 end   = round(end   + q_start, 3))
        segs_all[[q]] <- chunk_segs
      }
    }

    # clean temporary files
    for (f in tmp_chunks) if (file.exists(f)) unlink(f)

    segs <- bind_rows(compact(segs_all))
    if (nrow(segs) == 0) { log("  No segments in sample mode."); return(NULL) }

  } else {
    # Full mode: diar the entire audio
    segs <- diarize(path, n_speakers,
                  clustering_threshold   = clustering_threshold,
                  segmentation_threshold = segmentation_threshold,
                  log = log)
    if (nrow(segs) == 0) { log("  No segments."); return(NULL) }
  }

  roles <- assign_roles(path, segs, ref_emb, log)
  segs  <- segs |> mutate(speaker_id = speaker,
                           speaker    = roles[speaker] %||% "unknown")

  ovlps <- if (do_overlap) detect_overlaps(segs, log)
           else tibble(ov_start=double(), ov_end=double(), ov_dur=double())

# Preserve the original pyannote turns for later metrics
# (build_transcript_df, ov_sum, etc.). The edges used to extract
# features may differ if there is a transcription.
  segs_pyannote <- segs

# ── EARLY Transcription ─────────────────────────────────

# If the researcher activates Whisper, we transcribe BEFORE the computation

# acoustic to derive the segment boundaries of the wordsv(cutting at pauses ≥ pause_thr). 
# The acoustic data is then calculated on the actual spoken segment, 
# not on the pyannote boundaries.
  all_words <- tibble(); all_wsegs <- tibble()
  if (do_transcribe) {
    if (sample_secs > 0) {
      log("  Transcribing sample chunks (pre-acoustic)...")
      for (q in 1:4) {
        q_start  <- quantile_bounds[q]
        chunk_f  <- tempfile(fileext=".wav")
        system2("ffmpeg", c(
          "-y","-ss", sprintf("%.3f", q_start),
          "-t",  sprintf("%.3f", sample_secs),
          "-i",  path, "-ac","1","-ar","16000",
          "-loglevel","error", chunk_f),
          stdout=FALSE, stderr=FALSE)
        if (!file.exists(chunk_f) || file.size(chunk_f) < 1000) next
        res_q <- transcribe_verbatim(chunk_f, language=language, log=log)
        unlink(chunk_f)
        w_q <- if (is.list(res_q)) res_q$words    else res_q
        s_q <- if (is.list(res_q)) res_q$segments else tibble()
        if (nrow(w_q) > 0)
          w_q <- w_q |> mutate(w_start=w_start+q_start, w_end=w_end+q_start)
        if (nrow(s_q) > 0)
          s_q <- s_q |> mutate(w_start=w_start+q_start, w_end=w_end+q_start)
        all_words <- bind_rows(all_words, w_q)
        all_wsegs <- bind_rows(all_wsegs, s_q)
      }
    } else {
      result    <- transcribe_verbatim(path, language=language,
                                       orig_name=orig_name, log=log)
      all_words <- if (is.list(result)) result$words    else result
      all_wsegs <- if (is.list(result)) result$segments else tibble()
    }

    if (nrow(all_words) > 0) {
      pause_thr <- as.numeric(getOption("acustera.pause_split_s", 0.3))
      if (!is.finite(pause_thr)) pause_thr <- 0.3
      n_before <- nrow(segs)
      segs <- build_word_based_segments(segs, all_words, pause_thr = pause_thr)
      log(sprintf(
        "  ✓ Segments reconstructed from the transcript (pause ≥%.2fs): %d → %d.",
        pause_thr, n_before, nrow(segs)))
    } else {
      log("  Note: Whisper did not return any words; I used Pyannote borders.")
    }
  }

  # is_in_sample: always TRUE in sample mode (we already filtered in diary)
  is_in_sample <- function(start, end) TRUE

  log("  Extracting acoustic parameters...")

  rows <- vector("list", nrow(segs))

  for (i in seq_len(nrow(segs))) {
    s <- segs[i,]
    if (s$duration < min_seg) next
    if (!is_in_sample(s$start, s$end)) next
    feat <- py_features(path, s$start, s$end, as.integer(n_mfcc), log)
    if (is.null(feat)) next

    mid    <- (s$start + s$end) / 2
    quantile <- if (do_quantiles)
      as.character(cut(mid/total_dur, c(0,.25,.5,.75,1),
                       c("Q1","Q2","Q3","Q4"), include.lowest=TRUE))
    else NA_character_

    is_ov <- nrow(ovlps) > 0 &&
      any(s$start < ovlps$ov_end & s$end > ovlps$ov_start)

# Text and probability are inherited from the sub-segment if they came from
# the transcript (build_word_based_segments adds them).
# NOTE: must always be a 1-row tibble — bind_cols() with a 0×0 tibble()
# collapses the whole row to 0 rows, silently discarding the segment.
    tx_row <- if ("text" %in% names(s))
                tibble(text       = s$text,
                       n_words  = s$n_words,
                       avg_logprob = s$avg_logprob)
              else
                tibble(text=NA_character_, n_words=NA_integer_, avg_logprob=NA_real_)

    rows[[i]] <- bind_cols(
      tibble(couple=couple, therapy=therapy, therapy_num=therapy_num,
             speaker=s$speaker, speaker_id=s$speaker_id %||% s$speaker,
             seg_start=round(s$start,3), seg_end=round(s$end,3),
             is_overlap=is_ov, quantile=quantile),
      feat,
      tx_row)
  }

  n_long <- sum(segs$duration >= min_seg, na.rm = TRUE)
  features <- bind_rows(compact(rows))
  if (nrow(features) == 0)
    log(sprintf(
      "  WARNING: 0 segments with features (total segs=%d, with duration>=%.1f: %d). Check Python errors above.",
      nrow(segs), min_seg, n_long))
  else
    log(sprintf("  ✓ %d segments with features.", nrow(features)))

  # ── Pauses between turns of the same speaker ────────────────
  if (do_pauses && nrow(features) > 0) {
    features <- features |>
      arrange(seg_start) |>
      group_by(speaker) |>
      mutate(
# Full pause: time since the end of the previous speaker's turn
# of the same speaker, regardless of what happens in between
        pause_pre_total = round(
          seg_start - lag(seg_end, default=NA_real_), 3),

# pause clean: total pause only if no other speaker has an active segment in that interval
# has an active segment in that interval
        pause_pre_clean = {
          prev_end <- lag(seg_end, default=NA_real_)
          pmap_dbl(list(seg_start, prev_end, speaker),
            function(cur_start, prev_e, hab) {
              if (is.na(prev_e)) return(NA_real_)
              gap_start <- prev_e
              gap_end   <- cur_start
              if (gap_end <= gap_start) return(0)
              # Is there a segment from ANOTHER speaker in this interval?
              otros <- features |>
                filter(as.character(speaker) != hab,
                       seg_start < gap_end,
                       seg_end   > gap_start)
              if (nrow(otros) > 0) NA_real_
              else round(gap_end - gap_start, 3)
            })
        }
      ) |>
      ungroup()
    log("  ✓ pauses calculated.")
  } else if (!do_pauses) {
    features <- features |>
      mutate(pause_pre_total=NA_real_, pause_pre_clean=NA_real_)
  }

# Transcription: already done above (pre-acoustic). Here we only build
# the Whisper segment transcription CSV and fill in columns
# with text if there was no transcription.
  transcript_df <- tibble()
  if (do_transcribe) {
    if (!"text" %in% names(features))
      features <- features |> mutate(text=NA_character_, n_words=0L,
                                      avg_logprob=NA_real_)
    if (nrow(all_wsegs) > 0) {
      transcript_df <- build_transcript_df(all_wsegs, segs_pyannote, ovlps,
                                           couple, therapy, therapy_num)
      log(sprintf("  ✓ %d transcribed segments.", nrow(transcript_df)))

      trans_dir <- file.path(getwd(), "transcriptions")
      if (!dir.exists(trans_dir)) dir.create(trans_dir, recursive=TRUE)
      file_stem <- tools::file_path_sans_ext(orig_name %||% basename(path))
      csv_trans <- file.path(trans_dir, paste0(file_stem, "_transcript.csv"))
      write_csv(transcript_df, csv_trans)
      log(paste("  ✓ CSV transcript:", basename(csv_trans)))
    }
  } else {
    features <- features |> mutate(text=NA_character_, n_words=NA_integer_,
                                    avg_logprob=NA_real_)
  }

  # Save CSV of acoustic features
  out_dir <- file.path(getwd(), "transcriptions")
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive=TRUE)
  file_stem <- tools::file_path_sans_ext(orig_name %||% basename(path))
  csv_feat  <- file.path(out_dir, paste0(file_stem, "_features.csv"))
  write_csv(features, csv_feat)
  log(paste("  ✓ CSV features:", basename(csv_feat)))
# Save CSV of overlappings (ov_sum, with therapy_num and shift columns)
# NOTE: We save AFTER constructing ov_sum below

  ov_sum <- tibble(
    couple=couple, therapy=therapy, therapy_num=therapy_num,
    dur_total_s    = round(total_dur, 1),
    n_overlappings= nrow(ovlps),
    dur_overlap_s  = round(sum(ovlps$ov_dur %||% 0), 2),
    ratio_overlap  = round(sum(ovlps$ov_dur %||% 0)/total_dur, 4),
    turns_man    = sum(segs_pyannote$speaker=="man"),
    turns_woman     = sum(segs_pyannote$speaker=="woman"),
    turns_therapist = sum(segs_pyannote$speaker=="therapist"),
    dur_man_s   = round(sum(segs_pyannote$duration[segs_pyannote$speaker=="man"]),1),
    dur_woman_s    = round(sum(segs_pyannote$duration[segs_pyannote$speaker=="woman"]),1),
    dur_terap_s    = round(sum(segs_pyannote$duration[segs_pyannote$speaker=="therapist"]),1)
  )

  # Save CSV of ov_sum (includes therapy_num, turns, ratio_overlap)
  csv_ov <- file.path(out_dir, paste0(file_stem, "_overlaps.csv"))
  write_csv(ov_sum, csv_ov)

  # ── Export TextGrid for review in Praat ─────────────────
  tryCatch({
    tg_dir <- file.path(getwd(), "textgrids")
    if (!dir.exists(tg_dir)) dir.create(tg_dir, recursive = TRUE)
    file_stem <- tools::file_path_sans_ext(orig_name %||% basename(path))
    tg_path   <- file.path(tg_dir, paste0(file_stem, ".TextGrid"))
    # Use segments with text (refined if there was a transcription, otherwise use pyannote)
    segs_for_tg <- if ("text" %in% names(segs)) segs else segs_pyannote
    segs_to_textgrid(segs_for_tg, tg_path, duracion_total = total_dur)
    log(sprintf("  ✓ TextGrid stored: %s", basename(tg_path)))
  }, error = function(e)
    log(paste("  Warning: TextGrid could not be saved:", conditionMessage(e))))

  list(features=features, overlaps=ov_sum, transcript=transcript_df,
       segs=segs_pyannote)
}


# ══════════════════════════════════════════════════════════════
# SECTION 4b — TRANSCRIPTION (faster-whisper, full audio + mapping)
#
# Flow:
# 1. Whisper transcribes the entire audio ONCE with
# word_timestamps=True → each word has a start/end
# 2. Each word is assigned to the pyannote segment that covers
# its midpoint. Overlaps are resolved by coverage.

# 3. Words are regrouped by segment → text by segment
#
# Verbatim configuration: no normalization, no padding token filtering
# to capture lengthening, hesitations, and restarts.

# # ══════════════════════════════════════════════════════════════

.whisper_backend <- function() {
  py_run_string("import sys as _sys; _wplatform = _sys.platform")
  if (py$`_wplatform` == "darwin") "mlx" else "faster"
}

ensure_whisper <- function(log = message) {
  backend <- .whisper_backend()
  if (backend == "mlx") {
    ok <- tryCatch({ py_run_string("import mlx_whisper"); TRUE }, error=\(e) FALSE)
    if (!ok) {
      log("Installing mlx-whisper (Mac / Apple Silicon)...")
      virtualenv_install("acustera-env", "mlx-whisper", ignore_installed=FALSE)
      log("✓ mlx-whisper installed")
    }
  } else {
    ok <- tryCatch({ py_run_string("from faster_whisper import WhisperModel"); TRUE },
                   error=\(e) FALSE)
    if (!ok) {
      log("Installing faster-whisper...")
      virtualenv_install("acustera-env", "faster-whisper", ignore_installed=FALSE)
      log("✓ faster-whisper installed.")
    }
  }
}

# Map of short names to HuggingFace mlx-community repositories
.mlx_model_repo <- function(model_size) {
  repos <- c(
    "large-v3-turbo" = "mlx-community/whisper-large-v3-turbo",
    "large-v3"       = "mlx-community/whisper-large-v3-mlx",
    "medium"         = "mlx-community/whisper-medium-mlx"
  )
  repos[[model_size]] %||% paste0("mlx-community/whisper-", model_size)
}

# Find the mlx_whisper binary in the virtualenv
.mlx_whisper_bin <- function() {
  venv_bin <- "~/.virtualenvs/acustera-env/bin/mlx_whisper"
  venv_bin <- path.expand(venv_bin)
  if (file.exists(venv_bin)) return(venv_bin)
  # Search in PATH
  w <- Sys.which("mlx_whisper")
  if (nchar(w) > 0) return(w)
  NULL
}

init_whisper <- function(model_size = "large-v3-turbo", log = message) {
  backend <- .whisper_backend()
  log(paste0("  Configuring Whisper ", model_size,
             if (backend=="mlx") " [mlx-whisper CLI / Metal]..."
             else " [faster-whisper CPU int8]..."))
  if (backend == "mlx") {
# For mlx we use the CLI as a thread — avoid segfault 
# caused by Metal/MPS within the R process
    bin <- .mlx_whisper_bin()
    if (is.null(bin)) {
      log("  Installing mlx-whisper...")
      virtualenv_install("acustera-env", "mlx-whisper", ignore_installed=FALSE)
      bin <- .mlx_whisper_bin()
    }
    # Save path and model in R variables (not Python)
    options(acustera.mlx_bin   = bin)
    options(acustera.mlx_model = .mlx_model_repo(model_size))
    log(paste0("  ✓ mlx_whisper ready: ", bin))
  } else {
    py_run_string(sprintf('
import multiprocessing as _mpc
from faster_whisper import WhisperModel as _WM
if "_whisper" not in dir():
    _ncores = max(1, _mpc.cpu_count() - 1)
    _whisper         = _WM("%s", device="cpu", compute_type="int8",
                           cpu_threads=_ncores, num_workers=1)
    _whisper_backend = "faster"
', model_size))
    log("  ✓ Whisper ready")
  }
}

# Transcribe the entire audio file using word timestamps and verbatim formatting.
# Mac: mlx_whisper CLI as a thread — avoids Metal/R segfault
# Windows/Linux: faster-whisper via reticulate
# Returns a tibble of words: word, w_start, w_end, probability.
transcribe_verbatim <- function(path, language = "es", orig_name = NULL, log = message) {
  log(paste("  Transcribing (verbatim):", basename(path)))
  backend <- .whisper_backend()

  if (backend == "mlx") {
# ── mlx-whisper as an external thread ─────────────────
# Run the CLI as in the terminal, capture JSON
    bin   <- getOption("acustera.mlx_bin",   .mlx_whisper_bin())
    model <- getOption("acustera.mlx_model", .mlx_model_repo("large-v3-turbo"))

    if (is.null(bin) || !file.exists(bin)) {
      log("  Error: mlx_whisper not found. Enable Whisper and reprocess.")
      return(tibble())
    }

    # Temporary directory for output JSON
    tmp_dir <- tempdir()
    lang_arg <- if (!is.null(language) && nchar(language) > 0) language else "auto"

    args <- c(
      path,
      "--model",          model,
      "--language",       lang_arg,
      "--word-timestamps","True",
      "--output-format",  "json",
      "--output-dir",     tmp_dir
    )

    log(paste("  Executing mlx_whisper in subprocess..."))
    ret <- system2(bin, args=args, stdout=TRUE, stderr=TRUE)

    # The JSON is saved as <name_without_extension>.json
    json_file <- file.path(tmp_dir,
                   paste0(tools::file_path_sans_ext(basename(path)), ".json"))

    if (!file.exists(json_file)) {
      log(paste("  Error mlx_whisper:", paste(tail(ret, 5), collapse=" | ")))
      return(tibble())
    }

    tryCatch({
      json_txt <- readLines(json_file, warn=FALSE) |> paste(collapse="\n")
      json_txt <- gsub("\\bNaN\\b", "null", json_txt)
      result <- jsonlite::fromJSON(json_txt, simplifyVector=FALSE)

      # Full segments (one row per Whisper segment)
      segs  <- tibble()
      words <- tibble()
      for (seg in result$segments) {
        segs <- bind_rows(segs, tibble(
          w_start = as.numeric(seg$start %||% NA),
          w_end   = as.numeric(seg$end   %||% NA),
          text    = as.character(seg$text %||% "")
        ))
        # words with timestamps (for map_words_to_segments)
        if (!is.null(seg$words) && length(seg$words) > 0) {
          for (w in seg$words) {
            words <- bind_rows(words, tibble(
              word        = as.character(w$word        %||% ""),
              w_start     = as.numeric(w$start         %||% NA),
              w_end       = as.numeric(w$end           %||% NA),
              probability = as.numeric(w$probability   %||% NA)
            ))
          }
        }
      }

      # Save original JSON in transcriptions folder/
      trans_dir <- file.path(getwd(), "transcriptions")
      if (!dir.exists(trans_dir)) dir.create(trans_dir, recursive=TRUE)
      file_stem <- tools::file_path_sans_ext(orig_name %||% basename(path))
      dest_json <- file.path(trans_dir, paste0(file_stem, ".json"))
      file.copy(json_file, dest_json, overwrite=TRUE)
      unlink(json_file)

      log(sprintf("  ✓ JSON stored: %s", basename(dest_json)))
      if (nrow(segs) == 0) log("  Notice: Whisper did not return any segments.")
      list(segments=segs, words=words)
    }, error = function(e) {
      log(paste("  Error parsing Whisper JSON", conditionMessage(e)))
      list(segments=tibble(), words=tibble())
    })

  } else {
    # ── faster-whisper via reticulate ────────────────────────
    code <- sprintf('
_path     = "%s"
_language = "%s" if "%s" else None
_words    = []
_iter, _info = _whisper.transcribe(
    _path,
    language                    = _language,
    word_timestamps             = True,
    beam_size                   = 5,
    suppress_tokens             = [],
    no_speech_threshold         = 0.9,
    condition_on_previous_text  = True,
    compression_ratio_threshold = 2.4,
    temperature                 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    log_prob_threshold          = -1.0,
)
for _seg in _iter:
    if _seg.words:
        for _w in _seg.words:
            _words.append({
                "word"       : _w.word,
                "w_start"    : _w.start,
                "w_end"      : _w.end,
                "probability": _w.probability
            })
', path, language %||% "", language %||% "")
    tryCatch({
      py_run_string(code)
      words <- py$`_words`
      if (length(words) == 0) { log("  Notice: Whisper did not return words."); return(list(segments=tibble(), words=tibble())) }
      list(segments=tibble(), words=bind_rows(lapply(words, as.data.frame)) |> mutate(word=as.character(word)))
    }, error = function(e) {
      log(paste("  Transcription error:", conditionMessage(e)))
      list(segments=tibble(), words=tibble())
    })
  }  # end else faster-whisper
}  # end transcribe_verbatim

# Map each word to the PyanNote segment that covers its midpoint.
# `features_df` must have the following columns: `seg_start`, `seg_end`, `speaker`, `is_overlap`
# Return `features_df` with the `text` column added.
# Assign words to diarization segments (for acoustic features)
map_words_to_segments <- function(features_df, words_df) {
  if (nrow(words_df) == 0)
    return(mutate(features_df, text=NA_character_, n_words=NA_integer_,
                  avg_logprob=NA_real_))
  n_segs  <- nrow(features_df)
  buckets <- vector("list", n_segs)
  probs   <- vector("list", n_segs)
  for (i in seq_len(nrow(words_df))) {
    w     <- words_df[i, ]
    w_mid <- (w$w_start + w$w_end) / 2
    hit   <- which(features_df$seg_start <= w_mid & features_df$seg_end > w_mid)
    if (length(hit) > 0) {
      buckets[[hit[1]]] <- c(buckets[[hit[1]]], w$word)
      probs[[hit[1]]]   <- c(probs[[hit[1]]],   w$probability)
    }
  }
  features_df |>
    mutate(
      text       = sapply(buckets, \(ws) if(is.null(ws)||!length(ws)) NA_character_
                           else trimws(paste0(ws, collapse=""))),
      n_words  = sapply(buckets, \(ws) if(is.null(ws)) 0L else length(ws)),
      avg_logprob = sapply(probs,   \(ps) if(is.null(ps)||!length(ps)) NA_real_
                           else mean(log(pmax(ps, 1e-10)), na.rm=TRUE))
    )
}
# Build acoustic segments based on the TRANSCRIPTION: for each turn
# pyannote, group the words that fall within and split them into sub-segments
# when there is a silence ≥ pause_thr between consecutive words. This allows
# acoustic features to be calculated exactly on the spoken snippet
# actual, not on the (wider) borders of pyannote.
build_word_based_segments <- function(pyannote_segs, words_df,
                                      pause_thr = 0.3) {
  if (nrow(pyannote_segs) == 0) return(pyannote_segs)
  if (nrow(words_df) == 0) return(pyannote_segs)
  out <- list()
  for (i in seq_len(nrow(pyannote_segs))) {
    s <- pyannote_segs[i, , drop = FALSE]
    w <- words_df[
      (words_df$w_start + words_df$w_end) / 2 >= s$start &
      (words_df$w_start + words_df$w_end) / 2 <= s$end, , drop = FALSE]
    if (nrow(w) == 0) { out[[length(out)+1]] <- s; next }
    w <- w[order(w$w_start), , drop = FALSE]
    gaps <- c(0, w$w_start[-1] - w$w_end[-nrow(w)])
    grp  <- cumsum(gaps >= pause_thr)
    for (g in unique(grp)) {
      ix <- which(grp == g)
      sub <- s
      sub$start    <- w$w_start[min(ix)]
      sub$end      <- w$w_end[max(ix)]
      sub$duration <- sub$end - sub$start
      sub$text       <- trimws(paste0(w$word[ix], collapse = ""))
      sub$n_words  <- length(ix)
      sub$avg_logprob <- mean(log(pmax(w$probability[ix], 1e-10)), na.rm = TRUE)
      out[[length(out)+1]] <- sub
    }
  }
  dplyr::bind_rows(out)
}

# Re-segmentation by pauses: takes already mapped features + words_df and, for
# each row of features, if there is an internal pause (gap between consecutive words)
# ≥ threshold_s, divides the row into sub-segments.
# Acoustic columns are inherited from the original segment (they are not recomputed);
# only seg_start, seg_end, duration, text, and n_words are adjusted.
split_features_on_pauses <- function(features_df, words_df,
                                      threshold_s = 0.3) {
  if (nrow(features_df) == 0 || nrow(words_df) == 0) return(features_df)
  out <- list()
  for (i in seq_len(nrow(features_df))) {
    r <- features_df[i, , drop = FALSE]
    s <- r$seg_start; e <- r$seg_end
    w <- words_df[words_df$w_start >= s - 1e-6 &
                  words_df$w_end   <= e + 1e-6, , drop = FALSE]
    if (nrow(w) < 2) { out[[length(out)+1]] <- r; next }
    w <- w[order(w$w_start), , drop = FALSE]
    gaps <- c(0, w$w_start[-1] - w$w_end[-nrow(w)])
    cut_at <- which(gaps >= threshold_s)
    if (length(cut_at) == 0) { out[[length(out)+1]] <- r; next }
    grp <- cumsum(seq_len(nrow(w)) %in% cut_at)
    for (g in unique(grp)) {
      ix <- which(grp == g)
      sub <- r
      sub$seg_start  <- w$w_start[min(ix)]
      sub$seg_end    <- w$w_end[max(ix)]
      sub$duration   <- sub$seg_end - sub$seg_start
      sub$text      <- trimws(paste0(w$word[ix], collapse = ""))
      sub$n_words <- length(ix)
      if ("avg_logprob" %in% names(sub))
        sub$avg_logprob <- mean(log(pmax(w$probability[ix], 1e-10)),
                                 na.rm = TRUE)
      out[[length(out)+1]] <- sub
    }
  }
  dplyr::bind_rows(out)
}

# Build the Whisper segment transcription tibble:
# one row per Whisper segment, with the speaker assigned by
# the majority of temporal coverage with the PyanNote segments.
# speaker: lists ALL active speakers in that interval ("man + therapist").
# is_overlap: TRUE if there is >1 active speaker OR it falls within the overlap detection zone.
build_transcript_df <- function(whisper_segs, diar_segs, ovlps,
                                 couple, therapy, therapy_num) {
  if (nrow(whisper_segs) == 0) return(tibble())

  n <- nrow(whisper_segs)
  speaker  <- character(n)
  is_ov     <- logical(n)

  for (i in seq_len(n)) {
    ws <- whisper_segs$w_start[i]
    we <- whisper_segs$w_end[i]

    if (nrow(diar_segs) > 0) {
      overlap <- pmax(0,
        pmin(we, diar_segs$end) - pmax(ws, diar_segs$start))

      # All speakers with any temporary overlap
      activos <- unique(diar_segs$speaker[overlap > 0])
      if (length(activos) == 0) activos <- "unknown"

# The speaker column lists ALL assets (e.g., "man + therapist")
      speaker[i]            <- paste(activos, collapse = " + ")

      # overlapping if there is more than one active speaker
      is_ov[i] <- length(activos) > 1
    } else {
      speaker[i]            <- "unknown"
    }

 # Also mark if it falls within an area detected by Pyannote overlap detection
    if (nrow(ovlps) > 0 && any(ws < ovlps$ov_end & we > ovlps$ov_start))
      is_ov[i] <- TRUE
  }

  tibble(
    couple      = couple,
    therapy     = therapy,
    therapy_num = therapy_num,
    seg_start   = round(whisper_segs$w_start, 3),
    seg_end     = round(whisper_segs$w_end,   3),
    speaker    = speaker,     # all active speakers, separated by " + "
    is_overlap  = is_ov,        # TRUE if more than one active speaker
    text       = whisper_segs$text,
    n_words  = sapply(strsplit(trimws(whisper_segs$text), "\\s+"),
                         \(w) sum(nchar(w) > 0))
  )
}


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — STATISTICS
# ══════════════════════════════════════════════════════════════

PARAM_LABELS <- c( 
f0_mean="F0 mean (Hz)", f0_sd="F0 SD (Hz)", 
f0_range="F0 range (Hz)", f0_st="F0 range (semitones)", 
intensity_mean="Intensity (dB)", 
intensity_sd="SD intensity", hnr="HNR (dB)", 
jitter_local="Jitter local", shimmer_local="Shimmer local", 
cpp="CPP (dB)", 
speech_rate="Speech rate", zcr="ZCR", 
rms="RMS", spectral_centroid="Special centroid (Hz)", 
mfcc1="MFCC-1", mfcc2="MFCC-2", mfcc3="MFCC-3"
)

run_lmm <- function(df, incl_quantile=TRUE, incl_overlap=FALSE,
                    optimizer="bobyqa", p_method="BH") {
  df <- df |>
    filter(speaker %in% c("man","woman","therapist")) |>
    mutate(
      speaker    = factor(speaker, c("man","woman","therapist")),
      couple      = factor(couple),
      quantile     = factor(quantile, c("Q1","Q2","Q3","Q4")),
      therapy_num = as.numeric(therapy_num),
      is_overlap  = as.integer(is_overlap)
    )

  fixed <- "therapy_num * speaker"
  if (incl_quantile && "quantile" %in% names(df) && !all(is.na(df$quantile)))
    fixed <- paste(fixed, "+ quantile")
  if (incl_overlap)
    fixed <- paste(fixed, "+ is_overlap")


  params    <- intersect(names(PARAM_LABELS), names(df))
  lmm_rows  <- list()
  ph_rows   <- list()

  for (p in params) {
    sub <- df |> drop_na(all_of(p))
    if (nrow(sub) < 10) next
    n_par <- n_distinct(sub$couple)
    formula_str <- if (n_par >= 2) paste(p, "~", fixed, "+ (1|couple)")
                   else            paste(p, "~", fixed)
    m <- tryCatch(suppressWarnings(
      if (n_par >= 2)
        lmer(as.formula(formula_str), data=sub, REML=FALSE,
             control=lmerControl(optimizer=optimizer))
      else
        lm(as.formula(formula_str), data=sub)
    ), error=\(e) NULL)
    if (is.null(m)) next
    if (inherits(m, "lmerMod") && isSingular(m)) next
    a <- tryCatch(anova(m), error=\(e) NULL)
    if (is.null(a)) next
    eta <- tryCatch(eta_squared(m, partial=TRUE),
                    error=\(e) data.frame(Parameter=rownames(a),
                                          Eta2_partial=NA_real_))
    for (eff in rownames(a)) {
      ev <- if (eff %in% (eta$Parameter %||% ""))
              eta$Eta2_partial[eta$Parameter==eff] else NA_real_
      lmm_rows[[length(lmm_rows)+1]] <- list(
        parameter=p, label=PARAM_LABELS[p]%||%p, efecto=eff,
        F_value=a[eff,"F value"], df_num=a[eff,"NumDF"],
        df_den=a[eff,"DenDF"],   p_value=a[eff,"Pr(>F)"], eta2_p=ev)
    }
    tryCatch({
      em <- emmeans(m, ~therapy_num|speaker,
                    at=list(therapy_num=sort(unique(df$therapy_num))))
      ph_rows[[length(ph_rows)+1]] <- as_tibble(
        summary(contrast(em, method="pairwise", adjust="tukey"))) |>
        mutate(parameter=p, label=PARAM_LABELS[p]%||%p)
    }, error=\(e) NULL)
  }

  if (length(lmm_rows)==0) return(list(lmm=tibble(), posthoc=tibble()))

  lmm_df <- bind_rows(lmm_rows) |>
    mutate(p_adj=p.adjust(p_value,method=p_method),
           sig=case_when(p_adj<.001~"***",p_adj<.01~"**",
                         p_adj<.05~"*",TRUE~"ns"))
  list(lmm=lmm_df,
       posthoc=if(length(ph_rows)>0) bind_rows(ph_rows) else tibble())
}

build_text_block <- function(features, overlaps, lmm, posthoc,
                             active_params = NULL) {
  if (is.null(active_params)) active_params <- names(PARAM_LABELS)
  L <- c("════════════════════════════════════════════",
         "ACOUSTIC RESULTS — AcusTera",
         paste("Generated:", format(Sys.time(),"%Y-%m-%d %H:%M")),
         "════════════════════════════════════════════","")
  add <- function(...) L <<- c(L, paste0(...))

  if (!is.null(features) && nrow(features)>0) {
    add("── SAMPLE ──────────────────────────────────")
    add("couples:    ", n_distinct(features$couple))
    add("Sessions:   ", n_distinct(features$therapy_num))
    add("segments:  ", nrow(features))
    add("overlappings: ", sum(features$is_overlap,na.rm=TRUE),
        sprintf(" (%.1f%%)", mean(features$is_overlap,na.rm=TRUE)*100))
    add("")
  }

  if (!is.null(lmm) && nrow(lmm)>0) {
    add("── LMM: param ~ Session × speaker + quantile + (1|couple) ──")
    add("   Correction: Benjamini-Hochberg","")
    for (p in unique(lmm$parameter)) {
      sub <- filter(lmm, parameter==p)
      add("  ", sub$label[1])
      for (i in seq_len(nrow(sub))) {
        r  <- sub[i,]
        ps <- if(r$p_adj<.001)"< .001" else sprintf(".%03.0f",r$p_adj*1000)
        es <- if(is.na(r$eta2_p))"NA" else sprintf("%.2f",r$eta2_p)
        add(sprintf("    %-28s F(%2.0f,%4.0f)=%6.2f  p=%s  η²p=%s  %s",
                    r$efecto,r$df_num,r$df_den,r$F_value,ps,es,r$sig))
      }
      add("")
    }
  }

  if (!is.null(posthoc) && nrow(posthoc)>0) {
    sig_ph <- filter(posthoc, p.value<.05)
    if (nrow(sig_ph)>0) {
      add("── POST-HOC Significant (Tukey) ──────────")
      for (p in unique(sig_ph$parameter)) {
        sub <- filter(sig_ph, parameter==p)
        add("  ",sub$label[1])
        for (i in seq_len(nrow(sub))) {
          r <- sub[i,]
          add(sprintf("    %-10s  %s  Δ=%.2f  t=%.2f  p=%.4f",
                      r$speaker%||%"", r$contrast,
                      r$estimate, r$t.ratio, r$p.value))
        }
        add("")
      }
    }
  }

  if (!is.null(overlaps) && nrow(overlaps)>0) {
    add("── overlappings ────────────────────────────")
    if ("therapy_num" %in% names(overlaps) && "ratio_overlap" %in% names(overlaps)) {
      ov <- overlaps |> group_by(therapy_num) |>
        summarise(r=mean(ratio_overlap,na.rm=TRUE)*100,
                  n=mean(n_overlappings,na.rm=TRUE),
                  .groups="drop")
      for (i in seq_len(nrow(ov)))
        add(sprintf("  S%2s: %.1f%% overlapping  n=%.0f",
                    ov$therapy_num[i], ov$r[i], ov$n[i]))
    } else {
      add("  (overlapping data without column therapy_num)")
    }
    add("")
  }

  add("════════════════════════════════════════════",
      "ENDING — sticks Claude for interpretation",
      "════════════════════════════════════════════")
  paste(L, collapse="\n")
}

# ═════════════════════════════════════════════════════════════
# SECTION 5b — PROSODIC CONVERGENCE
#
# Implements two hypotheses:
# H1: Patients reduce prosody throughout therapy
# → LMM on relative values (intra-speaker z-score)
# H2: Patient prosody converges towards therapist prosody
# → Normalized Euclidean distance between patient and therapist per Session
# ═══════════════════════════════════════════════════════════════

# Parameters used for convergence
CONV_PARAMS <- c("f0_mean","f0_sd","f0_range",

"intensity_mean","intensity_sd",

"speech_rate","hnr",

"jitter_local","shimmer_local","cpp")

# ── Prepare the convergence dataframe ─────────────────────
# Aggregate by (couple, therapy_num, speaker) calculating:
# - mean of each parameter per Session/speaker
# - intra-Session range (max - min) as a measure of variability
# - intra-speaker z-score (normalizes anatomical differences)
# - Euclidean distance to the therapist in z-score space

build_convergence_df <- function(df) {
  params <- intersect(CONV_PARAMS, names(df))
  df     <- df |>
    filter(speaker %in% c("man","woman","therapist")) |>
    mutate(therapy_num = as.numeric(therapy_num),
           speaker    = as.character(speaker))

  # ── Step 1: Averages and ranges per couple × Session × speaker ──
  agg <- df |>
    group_by(couple, therapy_num, speaker) |>
    summarise(
      across(all_of(params),
             list(m   = \(x) mean(x, na.rm=TRUE),
                  rng = \(x) { v <- x[!is.na(x)]
                               if (length(v) < 2) NA_real_
                               else max(v) - min(v) }),
             .names = "{.col}__{.fn}"),
      n_segs = n(),
      .groups = "drop"
    )

# ── Step 2: Intra-speaker z-score (normalizes anatomy) ─────
# For each parameter, standardize within each
# couple × speaker combination across the Sessions.
# Thus, a man with F0=145Hz and a woman with F0=210Hz have
  # comparable values ​​when both are at their personal average.
  mean_cols <- paste0(params, "__m")
  agg <- agg |>
    group_by(couple, speaker) |>
    mutate(across(all_of(mean_cols),
                  \(x) as.numeric(scale(x)),
                  .names = "z_{.col}")) |>
    ungroup()

# ── Step 3: Euclidean distance between patient and therapist ─────────

# In the z-score space of the selected parameters.

# Calculated per couple and session.
  z_cols <- paste0("z_", mean_cols)
  z_cols <- intersect(z_cols, names(agg))

  ter <- agg |>
    filter(speaker == "therapist") |>
    select(couple, therapy_num, all_of(z_cols)) |>
    rename_with(\(x) paste0("ter_", x), all_of(z_cols))

  agg <- agg |> left_join(ter, by = c("couple" = "couple", "therapy_num" = "therapy_num"))

  if (length(z_cols) > 0) {
    sq_diffs <- sapply(z_cols, function(zc) {
      d <- agg[[zc]] - agg[[paste0("ter_", zc)]]
      ifelse(is.na(d), 0, d^2)
    })
    agg$dist_therapist <- sqrt(rowSums(sq_diffs, na.rm=TRUE))
  } else {
    agg$dist_therapist <- NA_real_
  }

  agg
}

# ── LMM on distance to therapist ─────────────────────────
# Hypothesis H2: dist_therapist ~ therapy_num + (1|couple)
# Only for patients (man and woman).
run_convergence_lmm <- function(conv_df, optimizer = "bobyqa") {
  results <- list()

# H1: Do patients reduce their prosodic variability?
  rng_cols <- grep("__rng$", names(conv_df), value=TRUE)
  for (rc in rng_cols) {
    param_name <- sub("__rng$","",rc)
    sub <- conv_df |> filter(speaker %in% c("man","woman")) |> drop_na(all_of(rc))
    if (nrow(sub) < 6) next
    n_par <- n_distinct(sub$couple)
    m <- tryCatch(suppressWarnings(
      if (n_par >= 2)
        lmer(as.formula(paste(rc,"~ therapy_num * speaker + (1|couple)")),
             data=sub, REML=FALSE, control=lmerControl(optimizer=optimizer))
      else
        lm(as.formula(paste(rc,"~ therapy_num * speaker")), data=sub)
    ), error=\(e) NULL)
    if (is.null(m)) next
    if (inherits(m, "lmerMod") && isSingular(m)) next
    a <- tryCatch(anova(m), error=\(e) NULL)
    if (is.null(a)) next
    for (eff in rownames(a))
      results[[length(results)+1]] <- list(
        hipotesis  = "H1: variability reduction",
        parameter  = paste0(PARAM_LABELS[param_name] %||% param_name,
                            " (range)"),
        efecto     = eff,
        F_value    = a[eff,"F value"],
        df_num     = a[eff,"NumDF"],
        df_den     = a[eff,"DenDF"],
        p_value    = a[eff,"Pr(>F)"]
      )
  }

# H2: Do patients converge towards the therapist?
  sub2 <- conv_df |> filter(speaker %in% c("man","woman")) |> drop_na(dist_therapist)
  if (nrow(sub2) >= 6) {
    n_par2 <- n_distinct(sub2$couple)
    m2 <- tryCatch(suppressWarnings(
      if (n_par2 >= 2)
        lmer(dist_therapist ~ therapy_num * speaker + (1|couple),
             data=sub2, REML=FALSE, control=lmerControl(optimizer=optimizer))
      else
        lm(dist_therapist ~ therapy_num * speaker, data=sub2)
    ), error=\(e) NULL)
    if (!is.null(m2) && !(inherits(m2,"lmerMod") && isSingular(m2))) {
      a2 <- tryCatch(anova(m2), error=\(e) NULL)
      if (!is.null(a2))
        for (eff in rownames(a2))
          results[[length(results)+1]] <- list(
hypothesis = "H2: convergence to the therapist",
parameter = "Euclidean distance (z-score)",
            efecto     = eff,
            F_value    = a2[eff,"F value"],
            df_num     = a2[eff,"NumDF"],
            df_den     = a2[eff,"DenDF"],
            p_value    = a2[eff,"Pr(>F)"]
          )
    }
  }

  if (length(results) == 0) return(tibble())
  bind_rows(results) |>
    mutate(p_adj = p.adjust(p_value, method="BH"),
           sig   = case_when(p_adj<.001~"***",p_adj<.01~"**",
                             p_adj<.05~"*",TRUE~"ns"))
}

# ── Graph: Euclidean distance per Session ───────────────────
plot_convergence_dist <- function(conv_df, param = "all") {
  # If a specific param is chosen, recompute dist using only that param
  if (!is.null(param) && param != "all" && nchar(param) > 0) {
    z_col  <- paste0("z_", param, "__m")
    ter_col <- paste0("ter_z_", param, "__m")
    if (z_col %in% names(conv_df) && ter_col %in% names(conv_df)) {
      conv_df <- conv_df |>
        mutate(dist_therapist = abs(.data[[z_col]] - .data[[ter_col]]))
    }
  }
  df <- conv_df |>
    filter(speaker %in% c("man","woman")) |>
    drop_na(dist_therapist)
  if (nrow(df) == 0) return(plot_ly() |> plotly_white())

  # Global media coverage by Session and speaker
  global <- df |>
    group_by(therapy_num, speaker) |>
    summarise(m  = mean(dist_therapist, na.rm=TRUE),
              se = sd(dist_therapist, na.rm=TRUE) / sqrt(n()),
              .groups="drop")

  p <- plot_ly()
  # Lines per couple — more visible and dotted
  for (hab in c("man","woman")) {
    for (par in unique(df$couple)) {
      d <- df |> filter(speaker==hab, couple==par) |>
        arrange(therapy_num)
      if (nrow(d) < 2) next
      p <- add_trace(p, x=d$therapy_num, y=d$dist_therapist,
                     type="scatter", mode="lines+markers",
                     name=paste(par, hab),
                     line=list(color=SP[hab], width=1.8, dash="dot"),
                     marker=list(color=SP[hab], size=5),
                     opacity=0.55, showlegend=TRUE,
                     text=sprintf("%s · %s · S%s: %.2f",
                                  par, hab, d$therapy_num, d$dist_therapist),
                     hoverinfo="text")
    }
  }
  # Global line by speaker
  for (hab in c("man","woman")) {
    d <- filter(global, speaker==hab) |> arrange(therapy_num)
    if (nrow(d) == 0) next
    p <- add_trace(p, x=d$therapy_num, y=d$m, name=hab,
                   type="scatter", mode="lines+markers",
                   line=list(color=SP[hab], width=3),
                   marker=list(color=SP[hab], size=8),
                   error_y=list(type="data", array=d$se,
                                visible=TRUE, color=SP[hab],
                                thickness=1.5),
                   text=sprintf("%s S%s: %.2f±%.2f",
                                hab, d$therapy_num, d$m, d$se),
                   hoverinfo="text")
  }
  # Reference line at 0 (therapist)
p |>
add_segments(x=min(global$therapy_num), xend=max(global$therapy_num),
y=0, yend=0,
line=list(color=SP["therapist"], dash="dot", width=2),
name="therapist (ref.)", showlegend=TRUE) |>
dark_layout(
xlab = "Session",
ylab = "Euclidean distance (z-score)",
title = "H2: Prosodic convergence towards the therapist<br><sup>The green line at 0 is correct: the therapist is the reference. Shorter distance = greater similarity.</sup>"
)
}

# ── Chart: intra-Session ranges by parameter ───────────────
plot_ranges <- function(conv_df, param = "f0_mean") {
  rng_col <- paste0(param, "__rng")
  if (!rng_col %in% names(conv_df))
    return(plot_ly() |> plotly_white())

  df <- conv_df |>
    filter(speaker %in% c("man","woman","therapist")) |>
    drop_na(all_of(rng_col))

  global <- df |>
    group_by(therapy_num, speaker) |>
    summarise(m  = mean(.data[[rng_col]], na.rm=TRUE),
              se = sd(.data[[rng_col]],   na.rm=TRUE) / sqrt(n()),
              .groups="drop")

  p <- plot_ly()
  for (hab in c("man","woman","therapist")) {
    d <- filter(global, speaker==hab) |> arrange(therapy_num)
    if (nrow(d) == 0) next
    p <- add_trace(p, x=d$therapy_num, y=d$m, name=hab,
                   type="scatter", mode="lines+markers",
                   line=list(color=SP[hab], width=2.5),
                   marker=list(color=SP[hab], size=7),
                   error_y=list(type="data", array=d$se,
                                visible=TRUE, color=SP[hab],
                                thickness=1.5),
                   hoverinfo="text",
                   text=sprintf("%s S%s: %.1f±%.1f",
                                hab, d$therapy_num, d$m, d$se))
  }
  lbl <- paste0(PARAM_LABELS[param] %||% param, " — range intra-Session")
  p |> dark_layout(xlab="Session", ylab="range (max-min)", title=lbl)
}

# ── Chart: z-scores per Session (H1 visual) ─────────────────
plot_zscores <- function(conv_df, param = "f0_mean") {
  z_col <- paste0("z_", param, "__m")
  if (!z_col %in% names(conv_df))
    return(plot_ly() |> plotly_white())

  df <- conv_df |>
    filter(speaker %in% c("man","woman","therapist")) |>
    drop_na(all_of(z_col))

  global <- df |>
    group_by(therapy_num, speaker) |>
    summarise(m=mean(.data[[z_col]], na.rm=TRUE),
              se=sd(.data[[z_col]], na.rm=TRUE)/sqrt(n()),
              .groups="drop")

  p <- plot_ly()
  for (hab in c("man","woman","therapist")) {
    d <- filter(global, speaker==hab) |> arrange(therapy_num)
    if (nrow(d) == 0) next
    p <- add_trace(p, x=d$therapy_num, y=d$m, name=hab,
                   type="scatter", mode="lines+markers",
                   line=list(color=SP[hab], width=2.5),
                   marker=list(color=SP[hab], size=7),
                   error_y=list(type="data", array=d$se,
                                visible=TRUE, color=SP[hab],
                                thickness=1.5),
                   hoverinfo="text",
                   text=sprintf("%s S%s: z=%.2f±%.2f",
                                hab, d$therapy_num, d$m, d$se))
  }
  p |>
    add_segments(x=min(global$therapy_num), xend=max(global$therapy_num),
                 y=0, yend=0,
                 line=list(color="#525b72", dash="dot", width=1),
                 showlegend=FALSE) |>
    dark_layout(
      xlab  = "Session",
      ylab  = "z-score intra-speaker",
      title = paste0("H1: Relative evolution — ",
                     PARAM_LABELS[param] %||% param)
    )
}

# ═══════════════════════════════════════════════════════════════
# SECTION 5c — ANNOTTED PROSODIC EXAMPLES
#
# Extract segments at extreme percentiles of F0 and intensity,
# annotate them with mixed notation (Simplified Jefferson + values)
# and organizes them by Session tertile for evolutionary comparison.

#
# Notation:
# ↑[210Hz] very high pitch (top percentile)
# ↓[143Hz] very low pitch (bottom percentile)
# SHIFT very high intensity
# °text° very low intensity
# values ​​in [] = acoustic measurement of the segment
# ═════════════════════════════════════════════════════════════

# Assign Session time:
# "Session 1" → the session with the lowest therapy_num
# "intermediatete Session" → the session closest to the median
# "Final Session" → The session marked as is_last=TRUE in metadata;

# if none are marked, the one with therapy_num at most
#
# sess_meta: dataframe with columns therapy_num and is_last (can be NULL)
assign_momento <- function(therapy_num, sess_meta = NULL) {
  Sessions <- sort(unique(therapy_num))
  n        <- length(Sessions)

  s1     <- Sessions[1]
  s_last <- if (!is.null(sess_meta) && "is_last" %in% names(sess_meta)) {
    ultima_num <- sess_meta |>
      filter(as.logical(is_last)) |>
      pull(therapy_num) |>
      as.integer()
    if (length(ultima_num) > 0 && any(ultima_num %in% Sessions))
      max(ultima_num[ultima_num %in% Sessions])
    else
      Sessions[n]
  } else {
    Sessions[n]
  }
  # intermediatete: Session closest to the median, excluding s1 and s_last
  candidates <- Sessions[Sessions != s1 & Sessions != s_last]
  s_mid <- if (length(candidates) == 0)
    NA_integer_
  else
    candidates[which.min(abs(candidates - median(Sessions)))]

  dplyr::case_when(
    therapy_num == s1     ~ "Session 1",
    !is.na(s_mid) & therapy_num == s_mid ~ "Session intermediate",
    therapy_num == s_last ~ "Session final",
    TRUE                  ~ NA_character_    # exclude other Sessions
  )
}

# Annotate a segment with mixed prosodic notation
# Return a text string with symbols and values
annotate_segment <- function(text, f0, intensity,
                              f0_hi, f0_lo,
                              int_hi, int_lo) {
  if (is.na(text) || nchar(trimws(text)) == 0)
    return("[no transcription]")

  txt <- trimws(text)

  # Intensity: CAPITAL if high, °text° if low
  if (!is.na(intensity)) {
    if (intensity >= int_hi)
      txt <- toupper(txt)
    else if (intensity <= int_lo)
      txt <- paste0("\u00b0", txt, "\u00b0")   # °text°
  }

  # Pitch: prefix ↑[Hz] or ↓[Hz]
  prefix <- ""
  if (!is.na(f0)) {
    if (f0 >= f0_hi)
      prefix <- sprintf("\u2191(%.0fHz) ", f0)   # ↑
    else if (f0 <= f0_lo)
      prefix <- sprintf("\u2193(%.0fHz) ", f0)   # ↓
  }

  paste0(prefix, txt)
}

# Construct the table of prosodic examples
# percentile: extreme percentile (e.g., 0.15 = top/bottom 15%)
build_prosodic_examples <- function(df, percentil = 0.15,
                                    sess_meta = NULL) {
  req_cols <- c("text","f0_mean","intensity_mean",
                "speaker","therapy_num","couple","seg_start","seg_end")
  if (!all(req_cols %in% names(df)))
    return(tibble())

  df <- df |>
    filter(speaker %in% c("man","woman","therapist"),
           !is.na(text), nchar(text) > 2) |>
    mutate(momento = assign_momento(therapy_num, sess_meta))  |>
    filter(!is.na(momento))   # only the three Sessions of interest

  if (nrow(df) == 0) return(tibble())

# Global percentiles for F0 and intensity (per speaker)
# Normalize per speaker so that the "height" is relative
# to the speaker itself, not to anatomical differences
  thresholds <- df |>
    group_by(speaker) |>
    summarise(
      f0_hi  = quantile(f0_mean,      1 - percentil, na.rm=TRUE),
      f0_lo  = quantile(f0_mean,          percentil, na.rm=TRUE),
      int_hi = quantile(intensity_mean, 1 - percentil, na.rm=TRUE),
      int_lo = quantile(intensity_mean,   percentil, na.rm=TRUE),
      .groups = "drop"
    )

  df <- df |> left_join(thresholds, by="speaker")

# Mark extreme segments
df <- df |> 
mutate( 
is_extreme = (f0_mean >= f0_hi | f0_mean <= f0_lo | 
intensity_mean >= int_hi | intensity_mean <= int_lo), 
end_type = case_when( 
f0_mean >= f0_hi & intensity_mean >= int_hi ~ "↑pitch + INTENSITY", 
f0_mean <= f0_lo & intensity_mean <= int_lo ~ "↓pitch + °soft°", 
f0_mean >= f0_hi ~ "↑ high pitch", 
f0_mean <= f0_lo ~ "↓ low tone", 
intensity_mean >= int_hi ~ "HIGH INTENSITY", 
intensity_mean <= int_lo ~ "°low intensity°", 
TRUE ~ "" 
), 
annotated_text = pmap_chr( 
list(text, f0_mean, intensity_mean, f0_hi, f0_lo, int_hi, int_lo), 
annotate_segment 
) 
) |> 
filter(is_extreme)

  # Add prev/next speaker context
  df_sorted <- df |> arrange(couple, therapy_num, seg_start)

  df_sorted |>
    select(couple, momento, therapy_num, speaker,
           tipo_extremo, text_anotado,
           f0_mean, intensity_mean,
           seg_start, seg_end) |>
    mutate(
      f0_mean        = round(f0_mean, 1),
      intensity_mean = round(intensity_mean, 1),
      seg_start      = round(seg_start, 1),
      seg_end        = round(seg_end, 1)
    ) |>
    arrange(couple, therapy_num, seg_start)
}

# Render the examples as HTML for the viewer and the report
render_examples_html <- function(examples_df, max_per_cell = 5, 
ctx_before = 0L, ctx_after = 0L, 
ctx_same_only = FALSE, full_df = NULL) { 
if (nrow(df_examples) == 0) 
return("<p style='color:#8890a8'>No examples available.</p>") 

moments <- c("Session 1","Intermediate Session","Final Session") 
speakers <- c("man","woman","therapist") 
h_colors <- c(man="#4a9eff", woman="#e05c8a", therapist="#52c97a") 

# Build HTML table: rows = speaker, columns = moment 
header_cells <- paste0( 
sprintf('<th style="background:#f5f6fa;color:#2C7BB6;padding:10px 14px; 
font-family:IBM Plex Mono;font-size:11px; 
text-transform:uppercase;letter-spacing:.07em; 
border-bottom:2px solid #2C7BB6">%s</th>', 
c("speaker", moments)), 
collapse="")

  rows_html <- sapply(speakers, function(hab) {
    col_color <- h_colors[hab]
    cells <- sapply(momentos, function(mom) {
      ex <- examples_df |>
        filter(speaker == hab, momento == mom) |>
        arrange(therapy_num, seg_start) |>
        head(max_per_cell)

      if (nrow(ex) == 0)
        return('<td style="padding:10px 14px;vertical-align:top;
                           border-bottom:1px solid #e0e0e0;
                           color:#aaa;font-style:italic">—</td>')

      items <- paste0(sapply(seq_len(nrow(ex)), function(ei) {
        row <- ex[ei, ]

        # Context lines: N rows before/after from all speakers, like transcription table
        ctx_before_html <- ""
        ctx_after_html  <- ""
        if (!is.null(full_df) && (ctx_before > 0 || ctx_after > 0)) {
          seg_df <- full_df |>
            filter(couple     == row$couple,
                   therapy_num == row$therapy_num,
                   !is.na(text), nchar(text) > 0) |>
            arrange(seg_start)
          ci <- which(abs(seg_df$seg_start - row$seg_start) < 0.5)
          ci <- if (length(ci) > 0) ci[1] else NA_integer_
          if (!is.na(ci)) {
            make_ctx_line <- function(r) {
              hcol <- switch(as.character(r$speaker),
                man="#2C7BB6", woman="#c0392b",
                therapist="#27ae60", "#888")
              sprintf(
                '<div style="display:flex;gap:8px;padding:3px 6px;
                             background:#f0f0f0;border-radius:3px;
                             font-size:11px;margin:2px 0">
                   <span style="color:%s;font-weight:bold;
                                min-width:80px;font-family:IBM Plex Mono">%s</span>
                   <span style="color:#444">%s</span>
                 </div>',
                hcol, toupper(as.character(r$speaker)),
                htmltools::htmlEscape(r$text))
            }
            if (ctx_before > 0) {
              rows_b <- seg_df[seq(max(1, ci-ctx_before), max(1, ci-1)), ]
              if (!ctx_same_only) rows_b <- rows_b
              else rows_b <- rows_b[as.character(rows_b$speaker)==as.character(row$speaker), ]
              if (nrow(rows_b) > 0)
                ctx_before_html <- paste0(
                  apply(rows_b, 1, function(r) make_ctx_line(as.list(r))),
                  collapse="")
            }
            if (ctx_after > 0) {
              rows_a <- seg_df[seq(min(nrow(seg_df), ci+1), min(nrow(seg_df), ci+ctx_after)), ]
              if (!ctx_same_only) rows_a <- rows_a
              else rows_a <- rows_a[as.character(rows_a$speaker)==as.character(row$speaker), ]
              if (nrow(rows_a) > 0)
                ctx_after_html <- paste0(
                  apply(rows_a, 1, function(r) make_ctx_line(as.list(r))),
                  collapse="")
            }
          }
        }

        sprintf(
          '<div style="margin-bottom:10px">
             %s
             <div style="padding:8px 10px;background:#f9f9f9;border-radius:4px;
                          border-left:3px solid %s">
               <div style="font-family:IBM Plex Mono;font-size:10px;
                            color:#888;margin-bottom:3px">
                 <b style="color:%s">%s</b> · S%s · %s · [%.1fs–%.1fs] · F0: %.1fHz · Int: %.1fdB
               </div>
               <div style="font-family:IBM Plex Sans;font-size:13px;
                            color:#222;line-height:1.5">%s</div>
               <div style="font-family:IBM Plex Mono;font-size:10px;
                            color:%s;margin-top:3px;font-weight:bold">%s</div>
             </div>
             %s
           </div>',
          ctx_before_html,
          col_color, col_color, toupper(hab),
          row$therapy_num, row$couple,
          row$seg_start, row$seg_end,
          row$f0_mean, row$intensity_mean,
          htmltools::htmlEscape(row$text_anotado),
          col_color, row$tipo_extremo,
          ctx_after_html)
      }), collapse="")
      sprintf('<td style="padding:10px 14px;vertical-align:top;
                          border-bottom:1px solid #e0e0e0;
                          min-width:280px">%s</td>', items)
    })

    sprintf(
      '<tr>
        <td style="padding:10px 14px;vertical-align:top;
                   border-bottom:1px solid #e0e0e0;
                   font-family:IBM Plex Mono;font-size:11px;
                   color:%s;font-weight:bold;white-space:nowrap">%s</td>
        %s
      </tr>',
      col_color, toupper(hab), paste(cells, collapse=""))
  })

  sprintf(
    '<div style="overflow-x:auto">
      <table style="width:100%%;border-collapse:collapse;
                    background:#fff;border:1px solid #e0e0e0;border-radius:4px">
        <thead><tr>%s</tr></thead>
        <tbody>%s</tbody>
      </table>
    </div>
    <p style="font-family:IBM Plex Mono;font-size:10px;
              color:#888;margin-top:10px">
      Notation: ↑[Hz] high pitch · ↓[Hz] low pitch ·
      CAPS high intensity · °text° low intensity
    </p>',
    header_cells,
    paste(rows_html, collapse=""))
}

# ═════════════════════════════════════════════════════════════
# SECTION 6 — GRAPHICS (Plotly)
# ══════════════════════════════════════════════════════════════

SP <- c(man="#4a9eff", woman="#e05c8a",
        therapist="#52c97a", unknown="#8890a8")

dark_layout <- function(p, title=NULL, xlab=NULL, ylab=NULL) {
  bw <- isTRUE(getOption("acustera.bw", FALSE))
  colorway <- if (bw) c("#111111","#555555","#999999","#cccccc","#333333","#777777")
              else   c("#2C7BB6","#c0392b","#27ae60","#f39c12","#8e44ad","#16a085")
  p <- layout(p, template="plotly_white",
         font=list(family=if (bw) "Times New Roman" else "IBM Plex Sans",
                   color="#222", size=13),
         colorway = colorway,
         xaxis=list(title=xlab),
         yaxis=list(title=ylab),
         title=list(text=title, font=list(color="#222", size=15)),
         legend=list(font=list(size=12)),
         margin=list(t=40))
  if (bw) {
    # Apply patterns to bars and dashes to lines to distinguish series in B/W
    shapes <- c("","/",".","x","-","\\","|","+")
    dashes <- c("solid","dash","dot","dashdot","longdash","longdashdot")
    p <- plotly::plotly_build(p)
    for (i in seq_along(p$x$data)) {
      tr <- p$x$data[[i]]
      ty <- tr$type %||% ""
      if (ty == "bar") {
        p$x$data[[i]]$marker$pattern <-
          list(shape = shapes[((i-1) %% length(shapes)) + 1],
               fgcolor = "#000", bgcolor = "#fff",
               fillmode = "overlay", size = 6, solidity = 0.35)
        p$x$data[[i]]$marker$line <- list(color = "#000", width = 0.8)
      } else if (ty %in% c("scatter","scattergl")) {
        if (is.null(tr$mode) || grepl("lines", tr$mode)) {
          p$x$data[[i]]$line$dash <- dashes[((i-1) %% length(dashes)) + 1]
          p$x$data[[i]]$line$color <- colorway[((i-1) %% length(colorway)) + 1]
          p$x$data[[i]]$line$width <- 1.4
        }
        if (!is.null(tr$mode) && grepl("markers", tr$mode)) {
          syms <- c("circle","square","triangle-up","diamond","cross","x","star")
          p$x$data[[i]]$marker$symbol <- syms[((i-1) %% length(syms)) + 1]
          p$x$data[[i]]$marker$line <- list(color = "#000", width = 0.6)
        }
      } else if (ty == "box") {
        p$x$data[[i]]$marker$color <- colorway[((i-1) %% length(colorway)) + 1]
        p$x$data[[i]]$line <- list(color = "#000", width = 1)
      }
    }
  }
  plotly::config(p,
    toImageButtonOptions = list(
      format = "png", filename = "acustera_plot",
      width = 1400, height = 600, scale = 3),
    modeBarButtonsToAdd = list(
      list(name = "SVG (vectorial)",
           icon = list(width = 24, height = 24,
             path = "M12 2L2 7v10l10 5 10-5V7L12 2z"),
           click = htmlwidgets::JS(
             "function(gd){Plotly.downloadImage(gd,{format:'svg',filename:'acustera_plot',width:2400,height:1500});}"))
    ),
    displaylogo = FALSE)
}

# Alias ​​— some functions use plotly_white() directly
plotly_white <- function(p, title=NULL, xlab=NULL, ylab=NULL)
  dark_layout(p, title=title, xlab=xlab, ylab=ylab)

plot_descriptive <- function(df, param, by_var="speaker") {
  df <- df |> filter(speaker %in% c("man","woman","therapist")) |>
    drop_na(all_of(param))
  if (nrow(df)==0) return(plot_ly() |> plotly_white())
  stats <- df |>
    group_by(across(all_of(c(by_var,"therapy_num")))) |>
    summarise(m=mean(.data[[param]],na.rm=TRUE),
              sd=sd(.data[[param]],na.rm=TRUE), n=n(), .groups="drop")
  plot_ly(stats, x=~therapy_num, y=~m,
          color=~.data[[by_var]],
          colors=if(by_var=="speaker") SP else "Set2",
          type="scatter", mode="lines+markers",
          error_y=list(type="data",array=~sd,visible=TRUE,thickness=1.5),
          text=~sprintf("%s=%s<br>S%s: %.2f±%.2f (n=%d)",
                        by_var,.data[[by_var]],therapy_num,m,sd,n),
          hoverinfo="text",
          line=list(width=2.5), marker=list(size=7)) |>
    dark_layout(xlab="Session", ylab=PARAM_LABELS[param]%||%param,
                title=paste(PARAM_LABELS[param]%||%param,"by Session"))
}

plot_heatmap <- function(df, params) {
  df <- df |> filter(speaker %in% c("man","woman","therapist"))
  if (nrow(df)==0 || length(params)==0) return(plot_ly() |> plotly_white())
  spk_order <- c("man","woman","therapist")
  panels <- lapply(seq_along(spk_order), function(i) {
    hab <- spk_order[i]
    sub <- df |> filter(speaker==hab) |>
      group_by(therapy_num) |>
      summarise(across(all_of(intersect(params,names(df))),
                       ~mean(.x,na.rm=TRUE)), .groups="drop")
    if (nrow(sub)==0) return(NULL)
    mat   <- sub |> select(-therapy_num) |> as.matrix()
    mat_z <- tryCatch({
      z <- scale(mat); z[is.nan(z)|is.infinite(z)] <- 0; z
    }, error=\(e) matrix(0,nrow=nrow(mat),ncol=ncol(mat),dimnames=dimnames(mat)))
    ylabs <- sapply(colnames(mat), \(p) PARAM_LABELS[p]%||%p)
    cell_txt <- matrix(
      paste0(round(t(mat), 1), "<br>z=", sprintf("%.2f", t(mat_z))),
      nrow=nrow(t(mat_z)), ncol=ncol(t(mat_z)),
      dimnames=dimnames(t(mat_z))
    )
    xref_i <- if (i == 1) "x domain" else paste0("x", i, " domain")
    plot_ly(z=t(mat_z), x=as.character(sub$therapy_num), y=ylabs,
            text=cell_txt,
            texttemplate="%{text}",
            textfont=list(size=13, family="Inter", color="#222222"),
            type="heatmap",
            showscale=(i == 1),
            colorscale=list(c(0,"#3b4cc0"),c(.5,"#f7f7f7"),c(1,"#b40426")),
            colorbar=list(thickness=10, len=0.8, x=1.01,
                          title=list(text="z", side="right"),
                          tickfont=list(size=9)),
            zmin=-2, zmax=2, zmid=0,
            hovertemplate=paste0("<b>",hab,"</b><br>Session: %{x}<br>",
              "%{y}<br>%{text}<extra></extra>")) |>
      layout(annotations=list(list(
        text=toupper(hab), x=0.5, y=1.08,
        xref=xref_i, yref="y domain",
        showarrow=FALSE, font=list(color=SP[hab],size=13,family="Inter"))))
  })
  panels <- compact(panels)
  if (length(panels)==0) return(plot_ly() |> plotly_white())
  do.call(subplot, c(panels, list(nrows=1,shareY=TRUE,
                                   titleX=TRUE,margin=.04))) |>
    dark_layout(title="Acoustic heatmap (z-score per session)") |>
    layout(margin=list(t=80, r=55))
}

plot_trajectories <- function(df, param, show_pairs=TRUE) {
  df <- df |> filter(speaker %in% c("man","woman","therapist")) |>
    drop_na(all_of(param))
  if (nrow(df)==0) return(plot_ly() |> plotly_white())

  couples <- sort(unique(df$couple))
  # Distinguishable grey palette per couple (faint lines)
  n_par    <- length(couples)
  grays    <- grDevices::colorRampPalette(c("#bbbbbb","#555555"))(max(n_par,1))
  par_col  <- setNames(grays, couples)

  p <- plot_ly()

  if (show_pairs) {
    ps <- df |> group_by(couple, therapy_num, speaker) |>
      summarise(m=mean(.data[[param]], na.rm=TRUE), .groups="drop") |>
      filter(is.finite(m))
    for (par in couples) {
      for (hab in c("man","woman","therapist")) {
        d <- filter(ps, speaker==hab, couple==par) |> arrange(therapy_num)
        if (nrow(d) == 0) next
        # With a single Session, draw the point (not the line)
        mode_i <- if (nrow(d) >= 2) "lines+markers" else "markers"
        p <- add_trace(p, x=d$therapy_num, y=d$m,
                       type="scatter", mode=mode_i,
                       opacity=0.7, showlegend=TRUE,
                       name=paste(par, hab),
                       legendgroup=par,
                       line=list(color=SP[hab], width=2, dash="dot"),
                       marker=list(color=SP[hab], size=6, opacity=0.8,
                                   line=list(color=par_col[[par]], width=1.5)),
                       text=sprintf("couple %s · %s<br>S%s: %.2f",
                                    par, hab, d$therapy_num, d$m),
                       hoverinfo="text")
      }
    }
  }

  # Global mean per speaker (thick line)
  gs <- df |> group_by(therapy_num, speaker) |>
    summarise(m  = mean(.data[[param]], na.rm=TRUE),
              se = sd(.data[[param]], na.rm=TRUE) / sqrt(n()),
              .groups="drop")
  for (hab in c("man","woman","therapist")) {
    d <- filter(gs, speaker==hab) |> arrange(therapy_num)
    if (nrow(d)==0) next
    p <- add_trace(p, x=d$therapy_num, y=d$m,
                   name=paste("Mean", hab),
                   legendgroup=paste("mean", hab),
                   type="scatter", mode="lines+markers",
                   line=list(color=SP[hab], width=3),
                   marker=list(color=SP[hab], size=9),
                   error_y=list(type="data", array=d$se,
                                visible=TRUE, color=SP[hab], thickness=1.5),
                   text=sprintf("<b>Mean %s</b><br>S%s: %.2f±%.2f",
                                hab, d$therapy_num, d$m, d$se),
                   hoverinfo="text")
  }
  p |> dark_layout(
    xlab  = "Session",
    ylab  = PARAM_LABELS[param] %||% param,
    title = paste("Trajectories:", PARAM_LABELS[param] %||% param,
                  "<br><sup>Thin lines = individual couples · Thick lines = overall average</sup>")
  )
}

plot_lmm_heatmap <- function(lmm_df) {
  if (is.null(lmm_df)||nrow(lmm_df)==0) return(plot_ly()|>plotly_white())
  elab <- c("therapy_num"="Session","speaker"="speaker",
            "therapy_num:speaker"="Session×speaker",
            "quantile"="quantile","is_overlap"="overlapping")
  base <- lmm_df |>
    filter(efecto %in% names(elab)) |>
    mutate(eff=elab[efecto], nlp=pmin(-log10(p_adj+1e-10),6))
  mat_nlp <- base |> select(label,eff,nlp) |>
    pivot_wider(names_from=eff, values_from=nlp, values_fill=0)
  mat_sig <- base |> select(label,eff,sig) |>
    pivot_wider(names_from=eff, values_from=sig, values_fill="")
  ord   <- lmm_df |> filter(efecto=="therapy_num") |> arrange(p_adj) |> pull(label)
  ecols <- intersect(unique(elab), colnames(mat_nlp)[colnames(mat_nlp)!="label"])
  mat_nlp <- mat_nlp[match(intersect(ord,mat_nlp$label), mat_nlp$label), ]
  mat_sig <- mat_sig[match(mat_nlp$label, mat_sig$label), ]
  zm <- as.matrix(mat_nlp[, ecols, drop=FALSE])
  sm <- as.matrix(mat_sig[, ecols, drop=FALSE])
  plot_ly(z=zm, x=ecols, y=mat_nlp$label, text=sm, type="heatmap",
          colorscale=list(c(0,"#f0f4ff"),c(.4,"#1e4a7a"),
                          c(.7,"#4a9eff"),c(1,"#e8a045")),
          zmin=0, zmax=6,
          hovertemplate="<b>%{y}</b><br>%{x}: -log10(p)=%{z:.2f} %{text}<extra></extra>") |>
    add_annotations(x=rep(ecols,each=nrow(zm)), y=rep(mat_nlp$label,length(ecols)),
                    text=as.vector(sm), showarrow=FALSE,
                    font=list(color="#444",size=11)) |>
    dark_layout(title="Significance by parameter and effect<br><sup>*** p<.001  ** p<.01  * p<.05 (BH)</sup>")
}

plot_overlap_evo <- function(ov) {
  if(is.null(ov)||nrow(ov)==0) return(plot_ly()|>plotly_white())
  if (!"therapy_num" %in% names(ov) || !"ratio_overlap" %in% names(ov))
    return(plot_ly()|>plotly_white(title="No columns therapy_num/ratio_overlap"))
  plot_ly(ov, x=~therapy_num, y=~ratio_overlap, color=~couple,
          colors="Set2", type="scatter", mode="lines+markers",
          text=~sprintf("couple:%s S%s %.1f%%",couple,therapy_num,ratio_overlap*100),
          hoverinfo="text",line=list(width=2),marker=list(size=7)) |>
    dark_layout(xlab="Session",ylab="% overlapped",
                title="Ratio of overlapping") |>
    layout(yaxis=list(tickformat=".1%"))
}

plot_turns <- function(ov) {
  if(is.null(ov)||nrow(ov)==0) return(plot_ly()|>plotly_white())
  if (!"therapy_num" %in% names(ov)) return(plot_ly()|>plotly_white(title="No column therapy_num"))
  dl <- ov |>
    select(therapy_num,man=turns_man,
           woman=turns_woman,therapist=turns_therapist) |>
    pivot_longer(-therapy_num,names_to="rol",values_to="n") |>
    group_by(therapy_num,rol) |>
    summarise(m=mean(n,na.rm=TRUE),.groups="drop")
  plot_ly(dl, x=~therapy_num, y=~m, color=~rol,
          colors=SP[c("man","woman","therapist")],
          type="bar", barmode="group",
          text=~round(m,1), textposition="outside") |>
    dark_layout(xlab="Session",ylab="turns (average)",
                title="turns by role and Session")
}

# ══════════════════════════════════════════════════════════════
#  SECTION 7 — INFORME HTML
# ══════════════════════════════════════════════════════════════

build_report <- function(txt, features, overlaps, lmm, posthoc, notes, sess_meta=NULL) {
  tbl <- function(df) {
    if (is.null(df)||nrow(df)==0) return("<p>Sin datos.</p>")
    ths  <- paste0("<th>",names(df),"</th>",collapse="")
    rows <- apply(df,1,\(r)
      paste0("<tr>",paste0("<td>",htmlEscape(as.character(r)),"</td>",
             collapse=""),"</tr>"))
    sprintf('<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>',
            ths,paste(rows,collapse=""))
  }
  secs <- ""

  # Prosodic examples (only if there is a transcript)
  if (!is.null(features) && nrow(features) > 0 &&
      "text" %in% names(features) &&
      any(!is.na(features$text))) {
    ej <- tryCatch(build_prosodic_examples(features, percentil=0.15,
                                           sess_meta=sess_meta),
                   error=\(e) NULL)
    if (!is.null(ej) && nrow(ej) > 0) {
      secs <- paste0(secs,
 "<section><h2>Annotated Prosodic Examples</h2>",

"<p class='note'>segments at the extreme 15% percentile of F0 ",

"and intensity, normalized by speaker. ",
"Columns = Session 1 / Session intermediate / Session final.</p>",
render_examples_html(ej),
"</section>")
    }
  }

  if (!is.null(features)&&nrow(features)>0) {
    d <- features |>
      filter(speaker %in% c("man","woman","therapist")) |>
      group_by(couple,therapy_num,speaker) |>
      summarise(M_F0=round(mean(f0_mean,na.rm=TRUE),1),
                SD_F0=round(sd(f0_mean,na.rm=TRUE),1),
                n=n(),.groups="drop")
secs <- paste0(secs,"<section><h2>Descriptives — Medium F0</h2>", 
tbl(d),"</section>")
  }
  if (!is.null(lmm)&&nrow(lmm)>0)
   secs <- paste0(secs,"<section><h2>LMM Models</h2>", 
"<p class='note'>Model: <code>parameter ~ Session × speaker + quantile + (1|couple)</code></p>",
      tbl(lmm|>mutate(across(where(is.double),~round(.,3)))|>
            select(Parameter=label,Effect=efecto,F=F_value,
                   p=p_value,`p adj.`=p_adj,`η²p`=eta2_p,Sig=sig)),
      "</section>")
  if (!is.null(posthoc)&&nrow(posthoc)>0) {
    ph <- posthoc|>filter(p.value<.05)|>mutate(across(where(is.double),~round(.,4)))
    if (nrow(ph)>0)
      secs <- paste0(secs,"<section><h2>Post-hoc significant</h2>",
                     tbl(ph),"</section>")
  }
  secs <- paste0(secs,"<section><h2>Block for Claude</h2>",
    "<pre class='cpre'>",htmlEscape(txt),"</pre></section>")

  sprintf('<!DOCTYPE html><html lang="es"><head>
<meta charset="UTF-8"><title>AcusTera</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#ffffff;color:#222;font-family:"IBM Plex Sans",sans-serif;
     font-size:14px;line-height:1.7;max-width:960px;margin:0 auto;padding:48px 32px}
header{border-bottom:2px solid #2C7BB6;padding-bottom:20px;margin-bottom:36px}
.brand{font-family:"IBM Plex Mono";font-size:12px;color:#2C7BB6;letter-spacing:.1em;margin-bottom:10px}
h1{font-family:"Syne";font-size:26px;font-weight:800;color:#222}
.sub{color:#666;margin-top:4px;font-size:13px}
section{margin-bottom:40px}
h2{font-family:"Syne";font-size:17px;color:#2C7BB6;padding-bottom:8px;
   border-bottom:1px solid #e0e0e0;margin-bottom:14px}
.note{font-size:12px;color:#888;margin-bottom:10px}
.note code{background:#f5f6fa;padding:1px 5px;border-radius:3px;color:#2C7BB6}
table{width:100%%;border-collapse:collapse;font-family:"IBM Plex Mono";font-size:11px}
th{background:#f5f6fa;color:#555;text-transform:uppercase;font-size:10px;
   letter-spacing:.07em;padding:8px 10px;border-bottom:1px solid #e0e0e0;text-align:left}
td{padding:6px 10px;border-bottom:1px solid #f0f0f0;color:#333}
tr:hover td{background:#f9f9f9}
.cpre{background:#f5f6fa;border:1px solid #e0e0e0;border-radius:6px;padding:18px;
      font-family:"IBM Plex Mono";font-size:11px;color:#2C7BB6;
      white-space:pre-wrap;overflow:auto;max-height:500px}
.nbox{background:#f0f6ff;border-left:3px solid #2C7BB6;padding:12px 16px;
      border-radius:4px;font-size:13px;color:#444;white-space:pre-wrap}
</style></head><body>
<header>
  <div class="brand">◈ AcusTera</div>
  <h1>Acoustic analysis report</h1>
  <div class="sub">Couple therapy — longitudinal analysis · %s</div>
</header>
%s%s
</body></html>',
    format(Sys.time(),"%d/%m/%Y %H:%M"),
    if(nchar(trimws(notes))>0)
      sprintf('<section><h2>Notes</h2><div class="nbox">%s</div></section>',
              htmlEscape(notes)) else "",
    secs)
}

# ══════════════════════════════════════════════════════════════
# SECTION 8 — UI
# ═══════════════════════════════ ═══════════════════════════════

# Mini UI helpers
# UI helpers moved to bslib components

# ═════════════════════════════════════════════════════════════
# UI — shinydashboard + shinydashboardPlus
# Theme: clear/academic, stepped side navigation,
# compact subpanels, accessible forms
# ══════════════════════════════════════════════════════════════

# ── Helpers ──────────────────────────────────────────────────
hint_txt <- function(...) tags$p(class="text-muted", style="font-size:12px;margin-top:4px", ...)
sec_lbl  <- function(txt) tags$p(style="font-size:11px;font-weight:600;text-transform:uppercase;
                                         letter-spacing:.06em;color:#6c757d;margin:12px 0 4px", txt)
info_alert <- function(...) div(class="callout callout-info", style="margin:8px 0", ...)

ui <- shinydashboardPlus::dashboardPage(
  skin  = "black",
  scrollToTop = TRUE,
  options = list(sidebarExpandOnHover = FALSE),

  # ── Header ────────────────────────────────────────────────
  shinydashboardPlus::dashboardHeader(
    title = "AcusTera"
  ),

  # ── Sidebar ───────────────────────────────────────────────
shinydashboard::dashboardSidebar( 
width = 220, 
shinydashboard::sidebarMenu( 
id = "sidebar_menu", 
shinydashboard::menuItem("01 · Upload audios", tabName="tab_upload", 
icon=shiny::icon("upload")), 
shinydashboard::menuItem("02 · Metadata", tabName="tab_meta", 
icon=shiny::icon("table")), 
shinydashboard::menuItem("03 · Process", tabName="tab_process", 
icon=shiny::icon("play")), 
shinydashboard::menuItem("04 · Results", tabName="tab_results", 
icon=shiny::icon("chart-bar"), 
shinydashboard::menuSubItem("Information", tabName="tab_info"),
shinydashboard::menuSubItem("Descriptive", tabName="tab_desc"),
shinydashboard::menuSubItem("Acoustic Heatmap", tabName="tab_heat"),
shinydashboard::menuSubItem("Trajectories", tabName="tab_traj"),
shinydashboard::menuSubItem("LMM Models", tabName="tab_lmm"),
shinydashboard::menuSubItem("Overlappings", tabName="tab_ov"),
shinydashboard::menuSubItem("Convergence", tabName="tab_conv"),
shinydashboard::menuSubItem("Prosodic Examples", tabName="tab_ej"),
shinydashboard::menuSubItem("Transcription", tabName="tab_trans"),
shinydashboard::menuSubItem("Segmentation", tabName="tab_seg"), 
shinydashboard::menuSubItem("Diagnostics", tabName="tab_diag"), 
shinydashboard::menuSubItem("Prosodic profile", tabName="tab_profile"), 
shinydashboard::menuSubItem("Linguistics (POS)", tabName="tab_ling"), 
shinydashboard::menuSubItem("IA speakers review", tabName="tab_ai_rev") 
), 
shinydashboard::menuItem("Methods", tabName="tab_methods", 
icon=shiny::icon("book")), 
shinydashboard::menuItem("05 · Report", tabName="tab_report", 
icon=shiny::icon("file-alt")), 
shinydashboard::menuItem("Configuration", tabName="tab_config", 
icon=shiny::icon("cog")) 
), 
hr(), 
div(style="padding:0 15px;font-size:11px;color:#aaa", 
uiOutput("sidebar_status")) 
),

  # ── Body ──────────────────────────────────────────────────
  shinydashboard::dashboardBody(

    # Inject shinyjs + custom compact styles
    shinyjs::useShinyjs(),
    shinyjs::extendShinyjs(
      text = "
shinyjs.copyText = function(params) {
  navigator.clipboard.writeText(params[0]).then(function() {
    var btn = document.getElementById('btn_copy');
    if (btn) { var o=btn.innerText; btn.innerText='✓ Copiado';
      setTimeout(function(){ btn.innerText=o; }, 2000); }
  });
};
shinyjs.scrollLog = function() {
  var el = document.getElementById('log_out');
  if (el) { var p=el.querySelector('pre')||el; p.scrollTop=p.scrollHeight; }
};
",
      functions = c("copyText","scrollLog")
    ),

    tags$head(tags$style(HTML("
      /* Academic / paper look */
      body, .content-wrapper, .right-side { background:#f5f6fa !important; }
      .skin-light-blue .main-header .navbar { background:#2C7BB6 !important; }
      .skin-light-blue .main-sidebar { background:#fff !important;
        border-right:1px solid #e0e0e0; }
      .skin-light-blue .sidebar-menu > li.active > a,
      .skin-light-blue .sidebar-menu > li > a:hover {
        background:#e8f0fb !important; color:#2C7BB6 !important;
        border-left:3px solid #2C7BB6 !important; }
      .skin-light-blue .sidebar-menu > li > a { color:#444 !important; }
      .skin-light-blue .treeview-menu > li > a { color:#555 !important; font-size:12px; }
      .skin-light-blue .treeview-menu > li.active > a { color:#2C7BB6 !important; }

      /* Box overrides for academic feel */
      .box { border-top:3px solid #2C7BB6 !important;
             box-shadow:0 1px 4px rgba(0,0,0,.08) !important; }
      .box.box-success { border-top-color:#27ae60 !important; }
      .box.box-warning { border-top-color:#f39c12 !important; }
      .box.box-danger  { border-top-color:#c0392b !important; }
      .box-header { background:#fff !important; padding:8px 12px !important; }
      .box-header .box-title { font-size:13px !important; font-weight:600;
        text-transform:uppercase; letter-spacing:.04em; color:#333; }
      .box-body { padding:12px !important; }

      /* Compact inputs */
      .form-control { font-size:13px !important; }
      label { font-size:12px !important; font-weight:600; color:#555; }
      .shiny-input-container { margin-bottom:8px !important; }

      /* Log console */
      #log_out { background:#1e1e2e; color:#a6e3a1; font-size:11px;
        font-family:'Courier New',monospace; border-radius:4px;
        padding:10px; min-height:180px; max-height:280px;
        overflow-y:auto; border:none; width:100%; }

      /* Results block */
      #txt_block { background:#1e1e2e; color:#a6e3a1; font-size:11px;
        font-family:'Courier New',monospace; border-radius:4px;
        padding:10px; min-height:300px; max-height:500px;
        overflow-y:auto; border:none; width:100%; }

      /* Speaker cards */
      .spk-card { background:#fff; border:1px solid #e0e0e0;
        border-radius:6px; padding:12px; margin-bottom:10px; }
      .spk-card .spk-hdr { display:flex; align-items:center; gap:10px;
        margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #f0f0f0; }
      .spk-card .spk-fields { display:flex; flex-wrap:wrap; gap:8px; }
      .spk-card .spk-fields > div { flex:1; min-width:130px; }
      .badge-man    { background:#d6eaf8; color:#1a5276; padding:3px 8px;
        border-radius:99px; font-size:11px; font-weight:600; }
      .badge-woman     { background:#fdedec; color:#922b21; padding:3px 8px;
        border-radius:99px; font-size:11px; font-weight:600; }
      .badge-therapist { background:#eafaf1; color:#1e8449; padding:3px 8px;
        border-radius:99px; font-size:11px; font-weight:600; }
      .badge-unknown { background:#f2f3f4; color:#666; padding:3px 8px;
        border-radius:99px; font-size:11px; }

      /* Value boxes compact */
      .small-box { border-radius:6px !important; }
      .small-box h3 { font-size:28px !important; }
      .small-box p  { font-size:12px !important; }

      /* Callout */
      .callout { padding:8px 12px; border-left:4px solid #2C7BB6;
        background:#f0f6ff; border-radius:0 4px 4px 0;
        font-size:12px; color:#333; margin-bottom:10px; }
      .callout-warning { border-left-color:#f39c12; background:#fffbf0; }
      .callout-success { border-left-color:#27ae60; background:#f0faf4; }

      /* Upload areas */
      .upload-hint { font-size:11px; color:#999; font-style:italic;
        margin-top:4px; }

      /* Tabs inside box */
      .nav-tabs-custom { margin:0 !important; }
      .nav-tabs-custom .tab-content { padding:10px 0 0 !important; }
    "))),

    shinydashboard::tabItems(

      # ── 01 SUBIR AUDIOS ────────────────────────────────────
      shinydashboard::tabItem("tab_upload",
        shiny::fluidRow(
shinydashboard::box( 
title="Therapeutic sessions", width=6, status="primary", 
shiny::icon("music"), hint_txt("couple{N}_therapy{N}.wav"), 
fileInput("files_sessions", NULL, multiple=TRUE, 
accept=c(".wav",".mp3",".flac"), 
buttonLabel="Select", 
placeholder="No files") 
), 
shinydashboard::box( 
title="Speaker references", width=6, status="info", 
shiny::icon("user"), 
hint_txt("Optional · couple{N}_therapist.wav · minimum 10 s per speaker"), 
fileInput("files_refs", NULL, multiple=TRUE, 
accept=c(".wav",".mp3"), 
buttonLabel="Select", 
placeholder="No files"), 
DTOutput("dt_refs") 
) 
),
shiny::fluidRow(
shinydashboard::box(
title="Reviewed TextGrids (optional)", width=12, status="warning",
collapsible=TRUE, collapsed=TRUE,
shiny::div(class="callout callout-info",
shiny::tags$b("When to use this?"),
" If you have already reviewed the TextGrids generated by AcusTera in Praat ",
"(or you have your own), upload them here. ",
shiny::tags$b("Automatic diarization will be skipped"),
" and the acoustic features will be calculated over the intervals of the TextGrid. ",
"Named: ", shiny::tags$code("couple{N}_therapy{N}.TextGrid"),
" (must match the corresponding audio).",
shiny::tags$br(),
shiny::tags$b("Expected Tiers:"),
" one per speaker with names containing ", 
shiny::tags$i("man"), ", ", shiny::tags$i("woman"), 
" or ", shiny::tags$i("therapist"), 
". The text of the interval is used as a transcription."), 
fileInput("files_textgrids", NULL, multiple=TRUE, 
accept=c(".TextGrid",".textgrid"), 
buttonLabel="Select TextGrids", 
placeholder="No files"), 
shiny::uiOutput("ui_tg_status") 
) 
),
shiny::fluidRow(
shinydashboard::box(
title="Files Detected", width=12, status="primary",
DTOutput("dt_files"),
br(),
actionButton("go_metadata", "Continue to Metadata →",
class="btn btn-primary btn-sm")
)
),
shiny::fluidRow(
shinydashboardPlus::box(
title="Load Previous Analysis", width=12, status="warning",
collapsible=TRUE, collapsed=FALSE,
info_alert(tags$b("New:"),
" Loads saved analyses with one click. You can select multiple ",
" at once to merge couples in the same session."),
shiny::fluidRow(
shiny::column(9,
shiny::uiOutput("ui_sessions_picker") 

), 
shiny::column(3, 
actionButton("btn_sessions_load", 
tagList(shiny::icon("folder-open"), " Load selected"), 
class="btn btn-warning w-100"), 
br(), br(), 
actionButton("btn_sessions_refresh", 
tagList(shiny::icon("rotate"), " Refresh"), 
class="btn btn-outline-secondary btn-sm w-100"), 
br(), br(), 
actionButton("btn_sessions_delete", 
tagList(shiny::icon("trash"), " Delete"), 
class="btn btn-outline-danger btn-sm w-100") 
) 
),
tags$hr(),

tags$details(

tags$summary(tags$b("Load from loose CSVs (legacy mode)")),

info_alert("Alternative: Upload the CSVs (_features, _transcript, _overlaps, metadata). ",
"Use this only if you are upgrading from a previous version."),

shiny::fluidRow(

shiny::column(6,

fileInput("file_prev_features","Acoustic Features (_features.csv)",

accept=".csv", multiple=TRUE,

buttonLabel="Select",

placeholder="No files"),

hint_txt("File *_features.csv generated by AcusTera")

),

shiny::column(6,

fileInput("file_prev_transcript","Transcript (_transcript.csv)",

accept=".csv", multiple=TRUE,
buttonLabel="Select",
placeholder="No files (optional)"),
hint_txt("File *_transcript.csv generated by AcusTera"),
fileInput("file_prev_overlaps","overlappings (_overlaps.csv)",
accept=".csv", multiple=TRUE,
buttonLabel="Select",
placeholder="No files (optional)"),
hint_txt("File *_overlaps.csv generated by AcusTera")
)
),
shiny::fluidRow(
shiny::column(12,
fileInput("file_prev_meta","Metadata (metadata.csv, optional)",
accept=".csv", multiple=FALSE,
buttonLabel="Select",
placeholder="No file"), 

hint_txt("Exported from Metadata tab") 
) 
), 
actionButton("btn_load_prev", 
tagList(shiny::icon("folder-open"), " Load CSVs"), 
class="btn btn-warning btn-sm") 
) # details 
) 
) 
),

# ── 02 METADATA ─────────────────────────────────────── 
shinydashboard::tabItem("tab_meta", 
shiny::fluidRow( 
shinydashboard::box( 
title="Sessions metadata", width=12, status="primary", 
tags$div(class="callout", 
tags$b("N°★"), " = Session number (covariate in LMM) · ", 
tags$b("Language★"), " = code ISO: es, en, ca… empty = auto · ",
tags$b("Last?"), " = Session end of process"
),
info_alert("Edit cells directly. Double-click to edit."),
DTOutput("dt_sess_meta"),
br(),
actionButton("save_sess", shiny::icon("save"), label=" Save",

class="btn btn-success btn-sm me-2"),
downloadButton("export_meta", shiny::icon("download"),

label=" Export CSV",

class="btn btn-outline-secondary btn-sm")
)
),
shiny::fluidRow(
shinydashboard::box(
title="Mass reassignment of roles by file",
width=12, status="warning",
info_alert(
"Here you can correct the assigned role to each speaker_id in each Session. ",
  "Directly edit the column ", tags$b("Correct role"),
". Press ", tags$b("Apply to All"), " to propagate the change ",
"to all segments of that speaker_id in that Session."

),
DTOutput("dt_bulk_roles"),

br(),
actionButton("btn_bulk_roles",
tagList(shiny::icon("check"), " Apply to all segments"),

class="btn btn-warning btn-sm")


),

shiny::fluidRow(
shinydashboard::box(
title="Speaker Profile", width=12, status="info",
info_alert("Available after processing. Assigns gender, age, clinical role, and psychological profile."),
uiOutput("ui_spk_cards"),

br(), 

actionButton("save_spk", shiny::icon("save"), label=" Save", 
class="btn btn-success btn-sm") 
) 
), 
shiny::fluidRow( 
tags$div(style="padding:0 15px", 
actionButton("go_process", "Continue to Process →", 
class="btn btn-primary btn-sm")) 
) 
)
),

      # ── 03 PROCESAR ────────────────────────────────────────
 shinydashboard::tabItem("tab_process",
shiny::fluidRow( 
shinydashboard::box( 
title="Token HuggingFace", width=3, status="warning", 
passwordInput("hf_token", NULL, 
placeholder="hf_xxxxxxxxxxxxxxxxxxxxx"), 
hint_txt("Required for pyannote/speaker-diarization-3.1"), 
hr(), 
selectInput("embedding_model", "Embeddings model", 
choices = c( 
"WavLM Base+ (recommended)" = "microsoft/wavlm-base-plus-sv", 
"ECAPA-TDNN (fast)" = "speechbrain/spkrec-ecapa-voxceleb", 
"WeSpeaker ResNet34 (default pyannote)" = "pyannote/wespeaker-voxceleb-resnet34-LM"),


selected = "microsoft/wavlm-base-plus-sv"),
hint_txt("WavLM offers better separation of similar speakers. ",

"ECAPA is faster. WeSpeaker is the default in pyannote.")
),
shinydashboard::box(
title="Diarization", width=3, status="primary",
numericInput("n_speakers","Number of speakers",3,2,8),
numericInput("min_seg","Minimum sec (s)",2,.5,10,.5),
checkboxInput("do_overlap","Detect overlappings",TRUE),
hr(),
sliderInput("clustering_threshold",
tagList("Clustering threshold",
tags$small(class="text-muted"," (lower = stricter)")),

min=0.3, max=0.9, value=0.7, step=0.05),
hint_txt("Controls when two segments are considered to be from the same speaker. ",
"Lower it (0.5–0.6) if it confuses different speakers."),
sliderInput("segmentation_threshold",
tagList("Segmentation threshold",
tags$small(class="text-muted"," (turn changes)")),
min=0.3, max=0.7, value=0.5, step=0.05),
hint_txt("Sensitivity to detect speaker changes. ",
"Lower it to detect more changes, raise it for less fragmentation.")
),
shinydashboard::box( 
title="Automatic variables", width=3, status="primary", 
checkboxInput("do_quantiles", 
tags$span("temporal quantile Q1–Q4", 
tags$small(class="text-muted", " ★")), TRUE), 
hint_txt("Segment position in the audio (Q1=first quarter...)"), 
checkboxInput("do_mfcc","MFCCs 1–13",TRUE), 
hr(), 
sliderInput("sample_mins", 
tags$span("Sample by quantile (min)", 
tags$small(class="text-muted", " 0=full")), 
min=0, max=15, value=0, step=1, post=" min"), 
hint_txt("0 = analyze the full quantile. ", 
"N > 0 = analyze only the first N minutes of each quantile. ",
"Reduces processing time by ~4x."),
hr(),
checkboxInput("do_pauses","Calculate pauses between turns",TRUE),
hint_txt("pause_pre_total: time since the previous turn of the same speaker. ",
"pause_pre_clean: same if the interval does not contain another speaker.")
),
shinydashboard::box(
title="Whisper Transcription", width=3, status="info",
checkboxInput("do_transcribe","Activate Whisper",FALSE),
selectInput("whisper_model","Model",

c("large-v3-turbo","large-v3","medium"),"large-v3-turbo"),
hint_txt("Model HuggingFace: mlx-community/whisper-{model}"),
hint_txt("Mac: mlx-whisper (MPS). Win/Linux: faster-whisper.",

"Language in col. Language★ of metadata."),

hr(),

numericInput("pause_split_thr",

"Pause threshold for segmenting (s)",

value = 0.3, min = 0.1, max = 2, step = 0.05),

hint_txt(

"If Whisper is active, the audio segments are derived from ",
"the transcript: they are split when there is an internal silence ≥ threshold ",
"between words. Thus, the features (F0, intensity, jitter…) are ",
"calculated on the actual spoken segment. ",

"If Whisper is disabled, pyannote borders are used.")

)
),
shiny::fluidRow( 
shinydashboard::box( 
title="Launch pipeline", width=12, status="primary", 
actionButton("btn_process", 
label=tagList(shiny::icon("play")," Launch pipeline"), 
class="btn btn-primary btn-lg me-3"), 
tags$span(class="text-muted small", 
"It may take several minutes per file."), 
br(), br(), 
uiOutput("prog_ui"), 
tags$hr(), 
tags$div(style="display:flex;gap:10px;align-items:center;flex-wrap:wrap", 
tags$label(style="margin:0;font-weight:500", "Label (optional):"), 
tags$input(id="session_save_label", type="text", 
class="form-control form-control-sm", 
style="max-width:260px", 
placeholder="e.g. P03_pilot"), 
actionButton("btn_session_save", 
tagList(shiny::icon("save"), " Save analysis"), 
class="btn btn-success btn-sm"), 
downloadButton("dl_textgrids_zip", 
"⬇ TextGrids (Praat)", 
class="btn btn-outline-primary btn-sm"), 
tags$span(class="text-muted small", 
"Save features+transcript+metadata in a single .rds.", 
"TextGrids: a .TextGrid per Session to review in Praat.") 
), 
sec_lbl("Processing log"), 
verbatimTextOutput("log_out") 
) 
) 
),

# ── 04a DESCRIPTIVES ───────────────────────────────────
# ── 04a INFORMATION ────────────────────────────────────
shinydashboard::tabItem("tab_info",
shiny::fluidRow(
shinydashboard::box(
title="Collected variables and selection for analysis",

width=12, status="primary",
info_alert(
"Select the variables you want to include in heatmap, LMM, trajectories, and profile. ",
"Changes are applied instantly to all sections."

),
DTOutput("dt_var_info"),
br(),
checkboxGroupInput(
"active_params", "Active variables:", 
              choiceNames  = as.list(unname(PARAM_LABELS)),
              choiceValues = as.list(names(PARAM_LABELS)),
              selected     = names(PARAM_LABELS),
inline = TRUE 
), 
br(), 
actionButton("btn_select_all", "All", class="btn btn-sm btn-default me-1"), 
actionButton("btn_select_none", "None", class="btn btn-sm btn-default me-1"), 
actionButton("btn_select_core", "Core (F0+Int+HNR+SR)", 
class="btn btn-sm btn-primary") 
) 
) 
),

shinydashboard::tabItem("tab_desc",
shiny::fluidRow(
shinydashboard::box(title=NULL, width=12,
div(class="callout callout-info", tags$b("Descriptive Statistics: "),
"Average of each parameter per speaker, Session, and couple. ",
"Identifies baseline differences between roles and changes throughout the process. ",
"Each point = average of all segments for that speaker in that Session.")
)
),
shiny::fluidRow(
shinydashboardPlus::box(
title="Controls", width=4, status="primary", collapsible=TRUE,
selectInput("desc_p","Parameter",
setNames(names(PARAM_LABELS),PARAM_LABELS),"f0_mean"),
selectInput("desc_by","Group by", 
c("speaker","couple","therapy_num"),"speaker") 
), 
shinydashboard::box(title="Chart", width=8, status="primary", 
plotlyOutput("p_desc",height="340px")) 
), 
shiny::fluidRow( 
shinydashboard::box(title="Summary table", width=12,
DTOutput("dt_desc")) 
) 
),

      # ── 04b HEATMAP ────────────────────────────────────────
      shinydashboard::tabItem("tab_heat",
        shiny::fluidRow(
          shinydashboard::box(title=NULL, width=12,
div(class="callout callout-info", tags$b("Acoustic Heatmap: "),

"Z-score per parameter and Session, normalized by speaker/couple. ",

"Red = above the average · Blue = below. ",

"Allows you to see patterns of joint change between parameters throughout the process.")

)
        ),
        shiny::fluidRow(
          shinydashboardPlus::box(
            title="Parameters", width=3, status="primary", collapsible=TRUE,
            info_alert("Z-scores by Session. ",
              tags$span(style="color:#c0392b","■")," above average · ",
              tags$span(style="color:#2c7bb6","■")," below."),
            checkboxGroupInput("heat_p","",
              setNames(names(PARAM_LABELS),PARAM_LABELS),
              selected=c("f0_mean","f0_sd","intensity_mean",
                         "speech_rate","spectral_centroid","mfcc1"))
          ),
          shinydashboard::box(title="Heatmap acoustic", width=9, status="primary",
            plotlyOutput("p_heat",height="620px"))
        )
      ),

      # ── 04c TRAJECTORIES ───────────────────────────────────
      shinydashboard::tabItem("tab_traj",
        shiny::fluidRow(
          shinydashboardPlus::box(
            title="Controls", width=3, status="primary", collapsible=TRUE,
            selectInput("traj_p","Parameter",
                        setNames(names(PARAM_LABELS),PARAM_LABELS),"f0_mean"),
            checkboxInput("traj_pairs","Lines by couple",TRUE)
          ),
          shinydashboard::box(title="Longitudinal trajectories",
            width=9, status="primary",
            plotlyOutput("p_traj",height="480px"))
        )
      ),

      # ── 04d MODELOS LMM ────────────────────────────────────
      shinydashboard::tabItem("tab_lmm",
        shiny::fluidRow(
          shinydashboard::box(title=NULL, width=12,
            div(class="callout callout-info", tags$b("Mixed effects models (LMM): "),
"One model per parameter: parameter ~ Session × speaker + quantile + (1|couple).",

"Session indicates longitudinal change. The Session×speaker interaction indicates whether the change differs by role.",

"(1|couple) controls dependence between measures of the same couple.",

"p-values ​​corrected with Benjamini-Hochberg.")
          )
        ),
        shiny::fluidRow(
          shinydashboard::box(title="What is this and how do you read it?",
            width=12, status="info", collapsible=TRUE, collapsed=TRUE,
            tags$div(style="font-size:13px;line-height:1.55;color:#333",
              tags$h5(tags$b("Linear mixed effects model (LMM)")),
              tags$p("Un ",tags$b("LMM")," It is a regression that, in addition to the effects ",
                "main (",tags$i("fixed"),": Session, speaker, quantile…), model ",
                "la ",tags$b("dependence between measures of the same couple")," by ",
                "un ",tags$i("random intercept")," ",tags$code("(1|couple)"),
                ". This avoids treating 40 sessions of 5 couples as if they were 40 independent observations."),
              tags$h5(tags$b("How to read the table")),
              tags$ul(
                tags$li(tags$b("therapy_num"),": Session effect (does the parameter change throughout the process?). ",
                  "β positive = increases Session by Session; negative = decreases."),
                tags$li(tags$b("speaker"),": fundamental differences between roles (man vs. woman vs. therapist)."),
                tags$li(tags$b("therapy_num × speaker"),": la ",tags$i("pending")," The change rate differs by speaker.. ",
                  "If it turns out to be significant, e.g., only the woman reduces her F0 over time."),
                tags$li(tags$b("quantile"),": effect of being in the 1st/2nd/3rd/4th quarter of the audio (warm-up, emotional intensity…)."),
                tags$li(tags$b("(1|couple)"),": variance attributable to the couple as a unit.")
              ),
              tags$h5(tags$b("Benjamini-Hochberg (BH) Correction")),
              tags$p("They are estimated ",tags$b("a model by acoustic parameter"),
                " (F0, intensity, HNR, jitter…). That's a lot of simultaneous tests., ",
                "So if we used p < 0.05 without correction, we would get ",tags$i("false positives")," ",
                "by pure chance. BH controls the ",tags$b("false discovery rate (FDR)"),
                "Of the results marked 'significant', at most 5% are due to chance.",
                "It is less conservative than Bonferroni and is recommended in psychology and quantitative linguistics."),
tags$p("The table shows ",tags$code("p_adj"),
" (adjusted p-value) and the column ",tags$b("sig"),
" with the usual convention: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant."),
              tags$h5(tags$b("Significance Heatmap")),
tags$p("Rows are parameters and columns are effects. The color is ",
tags$code("-log10(p_adj)"),
": the darker the color, the more significant it is. It allows you to see at a glance which parameters change, what differs by speaker, and what interactions emerge.")
            )
          )
        ),
shiny::fluidRow( 
shinydashboard::box( 
title="Model specification", width=12, status="primary", 
info_alert(tags$code( 
"parameter ~ Session × speaker + quantile + (1|couple)"), 
" Benjamini-Hochberg correction."),
shiny::fluidRow( 
shiny::column(3, checkboxInput("lmm_q","temporal quantile",TRUE)), 
shiny::column(3, checkboxInput("lmm_ov","overlapping",FALSE)), 
shiny::column(3, actionButton("btn_lmm", 
tagList(shiny::icon("calculator"),"Calculate LMMs"), 
class="btn btn-primary btn-sm")) 
) 
) 
),
shiny::fluidRow( 
shinydashboard::box(title="Significance heatmap", width=12, 
plotlyOutput("p_lmm_heat",height="400px")) 
), 
shiny::fluidRow( 
shinydashboardPlus::box( 
title="Full results", width=6, 
collapsible=TRUE, collapsed=FALSE, 
DTOutput("dt_lmm") 
), 
shinydashboardPlus::box( 
title="Post-hoc contrasts (emmeans, Tukey)", width=6, 
collapsible=TRUE, collapsed=FALSE, 
DTOutput("dt_ph") 
) 
) 
),

      # ── 04e overlappings ──────────────────────────────────
shinydashboard::tabItem("tab_ov", 
shiny::fluidRow( 
shinydashboard::box(title="Overlap ratio", width=6, 
plotlyOutput("p_ov_evo",height="280px")), 
shinydashboard::box(title="turns by role and Session", width=6, 
plotlyOutput("p_ov_turns",height="280px")) 
), 
shiny::fluidRow( 
shinydashboard::box(title="Detailed table", width=12, 
DTOutput("dt_ov")) 
) 
),

      # ── 04f CONVERGENCE ───────────────────────────────────
      shinydashboard::tabItem("tab_conv",
        shiny::fluidRow(
          shinydashboard::box(
            title="What is prosodic convergence and what is calculated here?",
            width=12, status="info",
            # ── Full description for non-experts ─────
            tags$p(style="margin-bottom:10px;font-size:13px",
              "When two people speak, they tend to synchronize their way of speaking: ",
              "The tone, the rhythm, the volume. This is called",
              tags$b("prosodic convergence"),
              " And it's a sign of emotional connection and empathy. In couples therapy, ",
              "if patients become more vocally similar to the therapist over time, ",
              "It may indicate that they are adopting their most regulated and calm way of communicating."),
            tags$hr(style="margin:6px 0"),
            tags$table(style="width:100%;font-size:12px;border-collapse:collapse",
              tags$tr(
                tags$th(style="padding:6px 8px;background:#f5f6fa;border:1px solid #e0e0e0;width:30%",
                        "Chart / Statistical"),
                tags$th(style="padding:6px 8px;background:#f5f6fa;border:1px solid #e0e0e0;width:35%",
                        "What it measures"),
                tags$th(style="padding:6px 8px;background:#f5f6fa;border:1px solid #e0e0e0;width:35%",
                        "How to interpret it")
              ),
              tags$tr(
                tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
                        tags$b("H1 — Relative evolution (z-score)")),
                tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
                        "How each vocal parameter varies from session to session, ",
                        "normalized with respect to the average of that same speaker. ",
                        "z=0 means that the Session is in its own mean."),
                tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
                        "If the lines systematically rise/fall, that parameter ",
                        "It is changing throughout the process. LMM H1 models this change.")
              ),
              tags$tr(
                tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
                        tags$b("H1 — range intra-Session")),
                tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
                        "The difference between the maximum and minimum values ​​of the parameter ",
                        "within each session. It measures how much the voice varies WITHIN a session."),
                tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
                        "high range = very variable voice in that session (high activation). ",
                        "decreasing range with sessions = greater emotional regulation.")
              ),
tags$tr(
tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
tags$b("H2 — Euclidean distance")),
tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
"How much the patient's vocal values ​​differ from ",
"the therapist's in that session (in z-score space). ",
"The green line at 0 is correct: the therapist is the reference point."),

tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
"Lines that slope downwards towards 0 = convergence. ",
"The selector allows you to see the distance for a specific parameter ",
"or for all at once (multidimensional Euclidean distance).")
),
tags$tr(
tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
tags$b("LMMs convergence")),
tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
"Statistical models that test whether distance or range ",
"change significantly with the number of Sessions."),

tags$td(style="padding:6px 8px;border:1px solid #e0e0e0",
"p < .05 (with BH correction) = the change is not random. ",
"η²p indicates the effect size (> .14 = large effect).")
)
),
tags$hr(style="margin:8px 0"), 
shiny::fluidRow( 
shiny::column(4, selectInput("conv_param_z","Parameter (z-score)", 
setNames(CONV_PARAMS,sapply(CONV_PARAMS,\(p) PARAM_LABELS[p]%||%p)), 
"f0_mean")), 
shiny::column(4, selectInput("conv_param_rng","Parameter (range)", 
setNames(CONV_PARAMS,sapply(CONV_PARAMS,\(p) PARAM_LABELS[p]%||%p)), 
"f0_mean")), 
shiny::column(4) 
) 
) 
),
shiny::fluidRow( 
shinydashboard::box(title="H1 — Z-score per Session", 
width=6, plotlyOutput("p_conv_z",height="280px")), 
shinydashboard::box(title="H1 — range intra-Session", 
width=6, plotlyOutput("p_conv_rng",height="280px")) 
), 
shiny::fluidRow( 
shinydashboard::box(title="H2 — Euclidean distance to therapist", 
width=12, status="primary", 
shiny::fluidRow( 
shiny::column(4, 
selectInput("conv_param_dist", "Parameter for distance", 
c("All (multiparameter Euclidean)"="all", 
setNames(CONV_PARAMS, 
sapply(CONV_PARAMS, \(p) PARAM_LABELS[p]%||%p))), 
selected="all") 
) 
), 
plotlyOutput("p_conv_dist",height="320px")) 
), 
shiny::fluidRow( 
shinydashboard::box( 
title="Convergence LMMs", width=12, 
actionButton("btn_conv_lmm", 
tagList(shiny::icon("calculator"),"Calculate"), 
class="btn btn-primary btn-sm"), 
br(), br(), 
DTOutput("dt_conv_lmm"), 
br(), 
sec_lbl("Relative values ​​per Session"), 
DTOutput("dt_conv_table") 
) 
) 
),

# ── 04g PROSODIC EXAMPLES ─────────────────────────────
ui_tab_examples_v2(),

# ── 04h TRANSCRIPTION ────────────────────────────────── 

shinydashboard::tabItem("tab_trans", 
shiny::fluidRow( 
shinydashboard::box( 
title="Filters", width=3, status="primary", 
selectInput("trans_couple","couple",choices=NULL), 
selectInput("trans_therapy","Session",choices=NULL), 
selectInput("trans_speaker","speaker", 
c("All","man","woman","therapist","unknown"), 
"All"), 
checkboxInput("trans_show_ov","Overlaps only",FALSE), 
hr(), 
div(class="callout callout-info", 
style="font-size:11px;padding:6px 10px", 
shiny::icon("volume-up"), " ", 
"To play segments, copy the WAV audios to ", 
tags$code("www/audios/"), " inside the app directory. ", 
"Expected name: ", tags$code("couple_therapyN.wav") 
), 
br(), 
uiOutput("dl_trans_csv_ui"), 
br(), 
downloadButton("dl_trans_features","⬇ Features CSV", 
class="btn btn-outline-secondary btn-sm w-100") 
),
shinydashboard::box(
title="Transcription by segment Whisper",
width=9, status="primary",
info_alert(
"Click a row to play that audio segment (if the file is available). ",
"Edit the columns ", tags$b("speaker"), " and ", tags$b("overlapping"),
" to correct mappings."

),
uiOutput("ui_audio_player"),

uiOutput("ui_trans_row_editor"),

DTOutput("dt_trans")

)

)),
# ── 04i SEGMENTATION ─────────────────────────────────── 
shinydashboard::tabItem("tab_seg", 
shiny::fluidRow( 
shinydashboard::box( 
title="Filters", width=3, status="primary", 
selectInput("seg_couple","couple",choices=NULL), 
selectInput("seg_therapy","Session",choices=NULL), 
selectInput("seg_speaker","speaker",
c("All","man","woman","therapist","unknown"),

"All"),

hr(),

hint_txt("Click a row to play the segment. ",
"Edit the speaker above. Changes propagate to Transcription.")

),
shinydashboard::box(

title="Segmentation by Speaker (PyanNote)",

width=9, status="primary",

info_alert(

"One row per segment for diary (PyanNote). ",
"Changing the speaker here also updates the Transcription table."


),

uiOutput("ui_seg_audio_player"),

uiOutput("ui_seg_row_editor"),

DTOutput("dt_seg")





))),

 # ── 04j DIAGNOSIS speakers ──────────────────────────
shinydashboard::tabItem("tab_diag",
shiny::fluidRow(
shinydashboard::box(
title="F0 by Session and Role — Assignment Validation",

width=12, status="primary",

info_alert("If roles are swapped, the distributions of ",
"man and woman F0 will not match as expected. ",
"Use the table below to correct this."),
plotlyOutput("p_diag", height="320px")
)
),
shiny::fluidRow(
shinydashboard::box(
title="Role Fix by File", width=12, status="warning",
info_alert(
"For each combination couple+Session+speaker_id, select the correct role. ",
tags$b("Apply Fixes"), " updates the data in memory and rewrites the CSVs."

),
DTOutput("dt_role_fix"),

br(),
actionButton("btn_apply_roles",
tagList(shiny::icon("check"), " Apply Fixes"),

class="btn btn-warning btn-sm")


)

)),

# ── 04j PROSODIC PROFILE ─────────────────────────────── 
shinydashboard::tabItem("tab_profile", 
shiny::fluidRow( 
shinydashboard::box( 
title="Controls", width=2, status="primary",
checkboxGroupInput("group_profile", "Group by",
  choices  = c("Role"="speaker", "Couple"="couple", "Session"="therapy_num"),
  selected = c("speaker","couple","therapy_num")),
checkboxGroupInput("roles_profile", "Roles",
  choices  = c("Man"="man", "Woman"="woman", "Therapist"="therapist"),
  selected = c("man","woman","therapist")),
selectInput("couple_profile", "Couple (filter)",
  choices=c("All"), selected="All"),
checkboxGroupInput("profile_params", "Parameters", 
setNames(names(PARAM_LABELS), PARAM_LABELS), 
selected=c("f0_mean","f0_sd","f0_range", 
"intensity_mean","speech_rate","hnr")), 
hr(), 
checkboxInput("zscore_profile", 
"Show z-scores (normalized by parameter)", TRUE) 
), 
shinydashboard::box( 
title="Prosodic profile — Heatmap", width=10, status="primary", 
plotlyOutput("p_profile_heat", height="720px") 
) 
), 
shiny::fluidRow( 
shinydashboard::box( 
title="Table of means +/- SD", width=12, 
DTOutput("dt_profile") 
) 
) 
),

# ── METHODS / LIBRARIES ──────────────────────────────────
shinydashboard::tabItem("tab_methods",
shiny::fluidRow(
shinydashboard::box(
title = "Python Libraries and What You Extract with Each One",
width = 12, status = "primary", collapsible = TRUE,
shiny::HTML(paste0(
"<h4>Diarization and Transcription</h4>",
"<ul>",
"<li><b>pyannote.audio (3.4)</b> — diarization (who speaks when) and overlapping detection. Model: <code>pyannote/speaker-diarization-3.1</code> over MPS/CUDA. Also produces the speaker embedding (by default <i>wespeaker</i>) which is used to identify the therapist by cosine similarity to the therapist's voice sample.</li>",

"<li><b>mlx_whisper</b> (CLI) — automatic transcription (<i>verbatim</i> mode). Returns segments with <code>start</code>, <code>end</code>, <code>text</code>, <code>avg_logprob</code>. Segment→speaker alignment is done by crossing over with pyannote turns.</li>",

"</ul>",

"<h4>Prosody and Voice Quality</h4>",

"<ul>",

"<li><b>parselmouth / Praat</b> — <b>F0</b> (mean, SD, range), <b>jitter</b>, <b>shimmer</b>, <b>HNR</b>, <b>CPP</b>. Primary source of tonal and voice quality data. <code>speech_rate</code> actual = number of frames with F0 detected / duration; it is a <i>proxy</i> for speed, <b>not</b> real syllables/second.</li>",
"<li><b>librosa</b>—<b>MFCC</b> (13 coefficients), <b>spectral centroid</b>, <b>ZCR</b>, <b>RMS</b>. Timbre and energy features.</li>",

"<li><b>numpy / scipy / soundfile</b>—audio I/O, resampling, auxiliary statistics.</li>",

"</ul>",

"<h4>Linguistics</h4>",
"<ul>",
"<li><b>udpipe</b> (via R) with Spanish model — tokenization, lemma, and POS tag per token. Basis for TTR, lexical density, and Yule's K metrics.</li>",
"</ul>",
"<h4>AI (optional)</h4>",
"<ul>",
"<li><b>Anthropic (Claude)</b> and <b>OpenAI (GPT-4o-mini)</b> via REST API (<code>httr</code>) — speaker attribution review of the transcript.</li>",
"</ul>",
"<h4>Origin of each variable (summary)</h4>",
"<table class='table table-sm' style='font-size:12px'>",
"<tr><th>Variable</th><th>Library</th><th>How to calculate</th></tr>", 
"<tr><td>f0_mean, f0_sd, f0_range</td><td>parselmouth</td><td>Praat pitch (ac), statistics on frames pitched</td></tr>", 
"<tr><td>jitter_local, shimmer_local</td><td>parselmouth</td><td>PointProcess + Praat</td></tr>", 
"<tr><td>hnr</td><td>parselmouth</td><td>Harmonicity (ac)</td></tr>", 
"<tr><td>cpp</td><td>numpy/scipy</td><td>Cepstral Peak Prominence over real cepstrum</td></tr>", 
"<tr><td>speech_rate</td><td>parselmouth</td><td>frames with F0 / duration (proxy)</td></tr>", 
"<tr><td>intensity_mean, intensity_sd</td><td>parselmouth</td><td>Praat Intensity (RMS energy in dB)</td></tr>", 
"<tr><td>mfcc1–mfcc13</td><td>librosa</td><td>librosa.feature.mfcc (average per segment)</td></tr>", 
"<tr><td>spectral_centroid, zcr, rms</td><td>librosa</td><td>librosa.feature.*</td></tr>", 
"<tr><td>is_overlap</td><td>pyannote</td><td>pipeline overlapped-speech-detection</td></tr>", 
"</table>" 
)) 
) 
) 
),

# ── 05 REPORT ──────────────────── ───────────────────── 
shinydashboard::tabItem("tab_report", 
shiny::fluidRow( 
shinydashboard::box( 
title="Options", width=4, status="primary", 
checkboxInput("rep_desc","Descriptives", TRUE), 
checkboxInput("rep_lmm", "LMM Models", TRUE), 
checkboxInput("rep_ph", "Post-hoc", TRUE), 
checkboxInput("rep_ov", "overlappings", TRUE), 
hr(), 
textAreaInput("rep_notes","Notes / context", rows=5, 
placeholder="Hypothesis, sample, observations..."), 
br(), 
actionButton("btn_report", 
tagList(shiny::icon("file-export")," Generate HTML report"), 
class="btn btn-primary w-100 mb-2"), 
actionButton("btn_report_word", 
tagList(shiny::icon("file-word")," Generate Word report"), 
class="btn btn-outline-primary w-100"), 
br(), br(), 
uiOutput("dl_ui"), 
br(), 
uiOutput("dl_word_ui"), 
br(), hr(), 
shiny::tags$h5("Report v2 (with hints for AI)"), 
shiny::uiOutput("ui_dl_v2"), 
br(), 
shiny::selectInput("ai_report_provider","AI Provider", 
c("Anthropic" = "anthropic", "OpenAI" = "openai"), "anthropic"), 
shiny::actionButton("btn_ai_interpret", 
shiny::tagList(shiny::icon("robot")," Ask AI for interpretation"), 
class="btn btn-warning w-100") 
), 
shinydashboard::box( 
title="Result block for Claude", width=8, 
div(style="text-align:right;margin-bottom:6px", 
actionButton("btn_copy","Copy to clipboard", 
class="btn btn-outline-secondary btn-sm", 
id="btn_copy")), 
verbatimTextOutput("txt_block"), 
shiny::hr(), 
shiny::tags$h5("AI interpretation of the report"), 
shiny::uiOutput("ui_ai_interp") 
) 
) 
),

      # ── LINGÜÍSTICA (POS) ──────────────────────────────────
      ui_tab_linguistica(),

      # ── REVISIÓN IA DE speakerS ───────────────────────────
      ui_tab_ai_review(),

  # ── SETTINGS ────────────────────────────────────── 
shinydashboard::tabItem("tab_config", 
shiny::fluidRow( 
shinydashboard::box( 
title="LMM Model", width=4, status="primary", 
selectInput("lmm_opt","Optimizer", 
c("bobyqa","Nelder_Mead","nlminbwrap"),"bobyqa"), 
selectInput("lmm_pc","P-value correction",
c("BH","bonferroni","holm","none"),"BH"),
numericInput("lmm_a","Level α",.05,.001,.1,.005),
tags$hr(),
checkboxInput("bw_mode",
"B/W mode with patterns (for colorless publishing)", FALSE),
helpText(style="font-size:11px;color:#777",
"Applies grayscale, patterns to bars, dashes to lines",
" and different symbols to markers. Affects all",
" new graphics (reopens tabs to refresh).")
),
shinydashboard::box(
title="Python Environment", width=4, status="info",
verbatimTextOutput("py_info"),

br(),
actionButton("btn_reinstall",
tagList(shiny::icon("refresh")," Reinstall dependencies Python"), 
class="btn btn-warning btn-sm") 
), 
shinydashboard::box( 
title="System status", width=4, 
uiOutput("ui_status_card") 
) 
), 
shiny::fluidRow( 
shinydashboard::box( 
title = "API Keys (IA)", width = 6, status = "warning", 
shiny::uiOutput("ui_api_keys") 
), 
shinydashboard::box(
  title = "Cache and performance", width = 6, status = "info",
  shiny::tags$p(
    style = "font-size:12px;color:#666",
    "Intermediate results are cached in ",
    shiny::tags$code(".acustera_cache/"),
    "to speed up reanalysis."
  ),                                          # ← comma added
  shiny::actionButton(
    "btn_clear_cache", "Clear cache",
    class = "btn btn-outline-danger btn-sm"
  )
)
) 
) 
), # tabItems 
examples_v2_js 
) # dashboardBody
) # dashboardPage

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 9 — SERVER
# ══════════════════════════════════════════════════════════════

server <- function(input, output, session) {

  rv <- reactiveValues(
    sessions=NULL, refs=NULL, sess_meta=NULL, spk_meta=NULL,
    features=NULL, overlaps=NULL, transcript=NULL, segs=NULL,
    lmm=NULL, posthoc=NULL,
    conv_lmm=NULL, word_tmp=NULL,
    py_ok=FALSE, py_tried=FALSE, log=character(0), busy=FALSE,
    tokens=NULL, ai_review=NULL, ai_interp=NULL,
    ej2_selected=character(0)
  )

  log_add <- function(...) {
    msg <- paste0(format(Sys.time(),"[%H:%M:%S] "), paste0(...))
    rv$log <- c(rv$log, msg)
    message(msg)   # visible in the Positron console in real time
  }

  # State — sidebar + notification
  setstatus <- function(state, txt) {
    output$sidebar_status <- renderUI(
      tags$span(style=paste0("color:",
        switch(state, running="#f39c12", error="#e74c3c",
               active="#27ae60", "#aaa")),
        switch(state, running="⟳ ", error="✗ ", active="✓ ", ""), txt)
    )
    if (state == "error")
      showNotification(txt, duration=5, type="error")
  }

  # ── Python bootstrap ──────────────────────────────────────
  observe({
    if (rv$py_ok || isTRUE(rv$py_tried)) return()
    rv$py_tried <- TRUE
    tryCatch({
      setup_python(log_add)
      rv$py_ok <- TRUE
      setstatus("active","Python listo")
    }, error=function(e){
      log_add("ERROR Python: ",conditionMessage(e))
      setstatus("error", paste0("Python error: ", conditionMessage(e)))
    })
  })

  output$py_info <- renderText({
    if (!rv$py_ok) return("Not initialized.")
    cfg <- py_config()
    paste0("Python: ",cfg$python,"\nVersion: ",cfg$version)
  })

  observeEvent(input$btn_reinstall, {
    rv$py_ok    <- FALSE
    rv$py_tried <- FALSE
    tryCatch({
      virtualenv_remove("acustera-env",confirm=FALSE)
      setup_python(log_add); rv$py_ok <- TRUE
    }, error=\(e) log_add("ERROR reinstall: ",conditionMessage(e)))
  })

  # ── Upload ────────────────────────────────────────────────
  observeEvent(input$files_sessions, {
    req(input$files_sessions)
    df <- as_tibble(input$files_sessions) |>
      mutate(
        # Extract everything BEFORE _therapy as couple identifier
        # Works with any prefix: couple1, Madrid1, BCN_02, etc.
        # Remove extension first
        name_base   = tools::file_path_sans_ext(name),
        couple      = str_extract(name_base, "^.+?(?=_[Tt]herapy)"),
        therapy     = tolower(str_extract(name_base, "(?i)therapy\\w+")),
        therapy_num = as.integer(str_extract(therapy, "\\d+"))
      ) |>
      # If no couple detected (name doesn't match pattern), use name without extension
      mutate(couple = ifelse(is.na(couple), name_base, couple))
    rv$sessions  <- df
    rv$sess_meta <- df |>
      select(name, couple, therapy, therapy_num) |>
      mutate(lugar="", fecha="", notas="", is_last=FALSE, idioma="es")
    setstatus("",sprintf("%d files loaded",nrow(df)))
  })

  observeEvent(input$files_refs,{
    req(input$files_refs)
    rv$refs <- as_tibble(input$files_refs) |>
      mutate(
        name_base = tools::file_path_sans_ext(name),
        couple    = str_extract(name_base, "^.+?(?=_[Tt]herapy|_ref|_ref\\.|$)"),
        couple    = ifelse(is.na(couple), name_base, couple)
      )
  })

  output$dt_files <- renderDT({
    req(rv$sessions)
    rv$sessions |>
      select(Archivo=name,couple=couple,Session=therapy,
             `N°`=therapy_num,`MB`=size) |>
      mutate(MB=round(MB/1e6,2)) |>
      datatable(options=list(pageLength=10,dom="tip"),
                rownames=FALSE,selection="none")
  })

  output$dt_refs <- renderDT({
    req(rv$refs)
    rv$refs |>
      select(Archivo=name, couple=couple, `MB`=size) |>
      mutate(MB=round(MB/1e6,2)) |>
      datatable(options=list(pageLength=5, dom="tip"),
                rownames=FALSE, selection="none")
  })

  # ── Metadata ──────────────────────────────────────────────
  output$dt_sess_meta <- renderDT({
    req(rv$sess_meta)
    datatable(rv$sess_meta,
      editable=list(target="cell",
        columns=which(names(rv$sess_meta) %in%
          c("couple","therapy","lugar","fecha",
            "notas","therapy_num","is_last","idioma"))-1),
      options=list(pageLength=15,dom="tip"),
      rownames=FALSE, selection="none",
      colnames=c("Archivo","couple","Session","N°★",
                 "Place","Date","Notes","Last?","Language★"))
  },server=TRUE)

  observeEvent(input$dt_sess_meta_cell_edit,
    rv$sess_meta <- editData(rv$sess_meta,
                             input$dt_sess_meta_cell_edit,
                             "dt_sess_meta"))

  output$export_meta <- downloadHandler(
    filename="metadata.csv", content=\(f) write_csv(rv$sess_meta,f))

  output$ui_spk_cards <- renderUI({
    req(rv$spk_meta); m <- rv$spk_meta
    do.call(tagList,lapply(seq_len(nrow(m)),function(i){
      r   <- m[i,]
      rol <- as.character(r$rol_detected%||%"unknown")
      tags$div(class="spk-card",
        tags$div(class="spk-hdr",
          tags$span(class=paste("spk-badge",rol),rol),
          tags$span(class="badge badge-info",
                    style="background:#eef3fb;color:#1a5490;padding:2px 8px;border-radius:3px;font-size:11px;margin-right:6px",
                    shiny::icon("users"), " couple ",
                    tags$b(r$couple %||% "—")),
          tags$strong(r$archivo), tags$code(r$speaker_id)),
        tags$div(class="spk-fields",
          tags$div(selectInput(paste0("sx",i),"Sex",
            c("","man","woman","non-binary"),r$sex%||%"")),
          tags$div(numericInput(paste0("ag",i),"Age",
            r$age%||%NA,18,90)),
          tags$div(selectInput(paste0("rl",i),"Clinical role",
            c("","patient","therapist","observer"),r$rol%||%rol)),
          tags$div(selectInput(paste0("pf",i),"Psych. profile",
            c("","pursuer","avoider","unclassified"),r$perfil%||%"")),
          tags$div(textInput(paste0("nt",i),"Notes",r$notas%||%""))))
    }))
  })

  observeEvent(input$save_spk,{
    req(rv$spk_meta)
    n <- nrow(rv$spk_meta)
    for(i in seq_len(n)){
      rv$spk_meta$sex[i]  <- input[[paste0("sx",i)]]%||%""
      rv$spk_meta$age[i]  <- input[[paste0("ag",i)]]%||%NA
      rv$spk_meta$rol[i]   <- input[[paste0("rl",i)]]%||%""
      rv$spk_meta$perfil[i]<- input[[paste0("pf",i)]]%||%""
      rv$spk_meta$notas[i] <- input[[paste0("nt",i)]]%||%""
    }
    showNotification("Guardado.",type="message",duration=2)
  })

  # ── Process ───────────────────────────────────────────────
  output$log_out <- renderText({
    # Re-render each time rv$log changes
    isolate(paste(rv$log, collapse="\n"))
  })

  # Observe changes to rv$log and update the output
  observe({
    req(length(rv$log) > 0)
    txt <- paste(rv$log, collapse="\n")
    output$log_out   <- renderText(txt)
    output$log_modal <- renderText(txt)
  })
  output$log_modal <- renderText({ paste(rv$log %||% character(0), collapse="\n") })
  output$prog_ui <- renderUI({
    req(rv$busy)
    tags$div(class="prog-wrap",
      tags$div(class="prog-lbl","Procesando..."),
      tags$div(class="prog-track",tags$div(class="prog-fill")))
  })

  observeEvent(input$btn_process,{
req(rv$sessions, rv$py_ok,
nchar(trimws(input$hf_token%||%""))>5)
rv$busy <- TRUE; rv$log <- character(0)
rv$features <- NULL; rv$overlaps <- NULL
setstatus("running","Processing...")

# Progress modal with live log
showModal(modalDialog(
title = tagList(shiny::icon("gear", class="fa-spin"),
" Processing Sessions..."),
tags$div(style="font-size:12px;color:#555;margin-bottom:8px",
"This process may take several minutes. The log is updated in real time. ",
"You can close this window; processing continues in the background."), 
tags$div(id="log_modal_box", 
style="background:#1e1e2e;color:#a6e3a1;font-size:11px;font-family:'JetBrains Mono',monospace;padding:10px;border-radius:4px;max-height:420px;overflow-y:auto", 
verbatimTextOutput("log_modal")), 
tags$script(shiny::HTML(" 
(function(){ var sel='#log_modal_box'; 
var mo = new MutationObserver(function(){ 
var el = document.querySelector(sel); 
if (el) el.scrollTop = el.scrollHeight; 
}); 
setTimeout(function(){ 
var el = document.querySelector(sel); 
if (el) mo.observe(el, {childList:true, subtree:true, characterData:true}); 
}, 200); 
})(); 
")), 
size = "l", easyClose = TRUE, 
footer = tagList( 
tags$span(id="modal_status", class="text-muted small", 
"Processing... you can minimize."), 
modalButton("Close") 
) 
)) 
log_add("Starting: ",nrow(rv$sessions)," Sessions.") 
log_add("sample_mins = ", input$sample_mins %||% "NULL") 
log_add("do_pauses = ", isTRUE(input$do_pauses))

    # Only initialise pyannote when at least one session lacks a matching TextGrid
    .needs_diarization <- is.null(input$files_textgrids) || {
      stems_audio <- tools::file_path_sans_ext(rv$sessions$name)
      stems_tg    <- tools::file_path_sans_ext(input$files_textgrids$name)
      any(!tolower(stems_audio) %in% tolower(stems_tg))
    }
    if (.needs_diarization) {
      ensure_pyannote(log_add)
      rv$device <- init_pyannote_pipelines(input$hf_token,
                      input$embedding_model %||% 'microsoft/wavlm-base-plus-sv',
                      log_add)
    } else {
      log_add("All sessions have a TextGrid — skipping pyannote initialisation.")
      rv$device <- "cpu"
    }
    if (isTRUE(input$do_transcribe)) {
      ensure_whisper(log_add)
      init_whisper(input$whisper_model %||% "large-v3-turbo", log_add)
    }

    # Embeddings de referencia
    ref_embs <- list()
    if (!is.null(rv$refs) && nrow(rv$refs)>0) {
      log_add("Calculating reference embeddings...")
      for(i in seq_len(nrow(rv$refs))){
        r   <- rv$refs[i,]
        par <- r$couple%||%"all"
        dur <- tryCatch({
          h <- readWave(r$datapath,header=TRUE)
          min(h$samples/h$sample.rate,60)
        },error=\(e) 30)
        ref_embs[[par]] <- py_embedding(r$datapath,0,dur)
        log_add("  Ref '",par,"' lista.")
      }
    }

    feats <- list(); ovs <- list(); trans_list <- list(); segs_list <- list()

    for(i in seq_len(nrow(rv$sessions))){
      s  <- rv$sessions[i,]
      mt <- if(!is.null(rv$sess_meta))
              filter(rv$sess_meta,name==s$name) else NULL
      par <- mt$couple %||% s$couple %||% paste0("p",i)
      ter <- mt$therapy%||% s$therapy%||% paste0("t",i)
      num <- as.integer(mt$therapy_num%||%s$therapy_num%||%i)
      ref <- ref_embs[[par]]%||%ref_embs[["all"]]%||%NULL

      lang <- mt$idioma %||% "es"
      # NULL or empty → automatic Whisper detection
      if (is.na(lang) || nchar(trimws(lang)) == 0) lang <- NULL

      # Emcoupler TextGrid subido por nombre base (mismo stem que el audio)
      tg_path <- NULL
      if (!is.null(input$files_textgrids)) {
        stem_audio <- tools::file_path_sans_ext(s$name)
        tgf <- input$files_textgrids
        stems_tg <- tools::file_path_sans_ext(tgf$name)
        ix <- which(tolower(stems_tg) == tolower(stem_audio))
        if (length(ix) == 1) tg_path <- tgf$datapath[ix]
      }

      res <- tryCatch(
        process_session(s$datapath,par,ter,num,
          as.integer(input$n_speakers), ref,
          input$min_seg, if(isTRUE(input$do_mfcc))13L else 0L,
          isTRUE(input$do_overlap), isTRUE(input$do_quantiles),
          isTRUE(input$do_transcribe),
          language               = lang,
          orig_name              = tools::file_path_sans_ext(s$name),
          sample_mins            = as.numeric(input$sample_mins %||% 0),
          do_pauses              = isTRUE(input$do_pauses),
          clustering_threshold   = as.numeric(input$clustering_threshold %||% 0.7),
          segmentation_threshold = as.numeric(input$segmentation_threshold %||% 0.5),
          textgrid_path          = tg_path,
          log                    = log_add),
        error=function(e){
          log_add("ERROR: ",s$name," — ",conditionMessage(e)); NULL})

      if(!is.null(res)){
        feats[[i]] <- res$features
        ovs[[i]]   <- res$overlaps
        if (!is.null(res$transcript) && nrow(res$transcript) > 0)
          trans_list[[i]] <- res$transcript
        if (!is.null(res$segs) && nrow(res$segs) > 0) {
          seg_with_meta <- res$segs |>
            mutate(couple=par, therapy=ter, therapy_num=num)
          segs_list[[i]] <- seg_with_meta
        }
      }
    }

    feats_df <- bind_rows(compact(feats))

    if (nrow(feats_df) == 0) {
      log_add("ERROR: No segments were extracted. Check the processing log.")
      rv$processing <- FALSE
      setstatus("error", "No segments")
      return()
    }

    # Add missing columns with default values if they don't exist
    if (!"speaker"   %in% names(feats_df)) feats_df$speaker   <- "unknown"
    if (!"quantile"    %in% names(feats_df)) feats_df$quantile    <- NA_character_
    if (!"is_overlap" %in% names(feats_df)) feats_df$is_overlap <- FALSE
    if (!"therapy_num"%in% names(feats_df)) feats_df$therapy_num <- 1L

    rv$features <- feats_df |>
      mutate(speaker    = factor(speaker, c("man","woman","therapist","unknown")),
             quantile     = factor(quantile,  c("Q1","Q2","Q3","Q4")),
             is_overlap  = as.logical(is_overlap),
             therapy_num = as.integer(therapy_num)) |>
      group_by(speaker) |>
      mutate(f0_st = if ("f0_range" %in% names(feats_df) &&
                          "f0_mean"  %in% names(feats_df)) {
               # range in semitones: how many semitones the F0 variation covers
               # Always positive because it uses the range (max-min), not the deviation
               12 * log2(1 + pmax(f0_range, 0) / pmax(f0_mean, 1))
             } else NA_real_) |>
      ungroup()

    rv$overlaps   <- bind_rows(compact(ovs))
    rv$segs       <- bind_rows(compact(segs_list))
    rv$transcript <- bind_rows(compact(trans_list))

    rv$spk_meta <- rv$features |>
      distinct(archivo=therapy,couple,speaker_id=speaker,
               rol_detected=speaker) |>
      mutate(sex="",age=NA_integer_,
             rol=as.character(rol_detected),
             perfil="",notas="")

    log_add(sprintf("✓ %d segments processed.",nrow(rv$features)))
    rv$busy <- FALSE
    setstatus("active","Data ready")
    shinyjs::js$scrollLog()
    showNotification(
      sprintf("✓ Process completed: %d segments.", nrow(rv$features)),
      type = "message", duration = 6)
    # Auto-close the modal after a moment so the last message is visible
    shinyjs::delay(2000, removeModal())
  })

  # ── Resultados ────────────────────────────────────────────
  # ── Variable information ──────────────────────────────
  VAR_INFO <- tibble::tribble(
    ~Variable, ~`What it measures`, ~`Purpose`, ~Unit,
"f0_mean",          "Average pitch of the voice (fundamental frequency). This is what we perceive as a higher or lower voice.",                                     "Indicator of emotional state: pitch rises with emotional activation and falls with calmness or sadness. Women have higher F0 than men due to anatomy.",                                           "Hz",

  "f0_sd",            "How much the pitch varies across the segment (SD). A monotonous voice has low SD; an expressive voice, high SD.",                              "Measures prosodic expressiveness. In therapy, greater variability can indicate greater emotional involvement.",                                                                                   "Hz",

  "f0_range",         "Difference between the highest and lowest pitch within the segment.",                                                                           "Indicates intonation range. Low range = flat speech; high range = very expressive speech.",                                                                                                       "Hz",

  "f0_st",            "F0 range expressed in musical semitones, relative to the speaker's own average. Eliminates anatomical differences between speakers.",           "Allows comparing prosodic expressiveness between men and women without interference from natural pitch differences.",                                                                              "semitones",

  "intensity_mean",   "Average volume of the voice during the segment.",                                                                                              "Related to emotional activation and the energy put into speech. Very high volumes may indicate tension; very low volumes, withdrawal.",                                                           "dB",

  "intensity_sd",     "Volume variability (SD). A voice with very uniform volume has low SD.",                                                                         "Measures dynamism in vocal expression.",                                                                                                                                                          "dB",

  "hnr",              "Ratio between the harmonic component (clean sound) and voice noise. A healthy, relaxed voice has high HNR.",                                    "Indicator of vocal strain and signal quality. Low HNR may indicate strain, hoarseness, or stress.",                                                                                               "dB",

  "jitter_local",     "Variation in the duration of each vocal cycle relative to the previous one. In a stable voice, all cycles last practically the same.",         "Indicator of vocal irregularity and laryngeal tension. Values above 1% are usually associated with stress, emotional tension, or vocal fatigue.",                                                "%",

  "shimmer_local",    "Variation in the amplitude of each vocal cycle relative to the previous one. Complements jitter by measuring cycle-by-cycle volume irregularity.", "Indicator of laryngeal tension and vocal quality. Values above 3% are associated with a strained voice, fatigue, or stress. Together with jitter, forms a robust indicator of vocal state.",  "%",

  "cpp",              "Cepstral Peak Prominence. Measures how much the periodic peak stands out in the cepstral spectrum relative to a regression line.",              "One of the most robust indicators of vocal quality in emotional contexts. High CPP = clean, modal voice; low CPP = strained, creaky, or whispered. Highly sensitive to subtle emotional changes.", "dB",

  "speech_rate",      "Speech speed estimated as the proportion of voiced frames per second, normalized to the speaker's own mean (z-score).",                        "Reflects cognitive and emotional activation: speaking faster than usual may indicate anxiety or urgency.",                                                                                        "z-score",

  "zcr",              "Number of times the audio signal crosses zero per second. Related to fricative energy and spectral noise.",                                      "Complements other indicators of vocal quality and strain.",                                                                                                                                        "norm",

  "rms",              "Average energy of the audio signal (root mean square).",                                                                                        "Alternative measure of volume, more sensitive to energy peaks.",                                                                                                                                   "norm",

  "spectral_centroid","Center of gravity of the frequency spectrum. A brighter voice has a higher centroid.",                                                          "Indicator of vocal timbre. Changes in the centroid can reflect shifts in articulatory tension.",                                                                                                   "Hz",

  "mfcc1",            "First Mel-frequency cepstral coefficient. Strongly correlated with overall spectral energy.",                                                   "MFCCs collectively describe vocal timbre. MFCC-1 largely overlaps with RMS energy; most useful in multivariate models.",                                                                         "norm",

  "mfcc2",            "Second MFCC. Captures the broad shape of the spectral envelope.",                                                                               "Complements MFCC-1 to characterize vocal timbre.",                                                                                                                                                "norm",

  "mfcc3",            "Third MFCC. Captures finer spectral detail.",                                                                                                  "Together with MFCC-1 and MFCC-2, describes the complete timbre profile.",                                                                                                                         "norm"
)

  output$dt_var_info <- renderDT({
    df <- VAR_INFO |>
      mutate(
        Disponible = ifelse(Variable %in% names(rv$features %||% list()), "✓", "—"),
        Activa     = ifelse(Variable %in% (input$active_params %||%
                            names(PARAM_LABELS)), "✓", "—")
      )
    datatable(df,
      options=list(pageLength=20, dom="t", scrollX=TRUE),
      rownames=FALSE) |>
      formatStyle("Disponible",
        color=styleEqual(c("✓","—"), c("#27ae60","#aaa"))) |>
      formatStyle("Activa",
        color=styleEqual(c("✓","—"), c("#2C7BB6","#aaa")))
  })

  observeEvent(input$btn_select_all, {
    updateCheckboxGroupInput(session, "active_params",
      selected=names(PARAM_LABELS))
  })
  observeEvent(input$btn_select_none, {
    updateCheckboxGroupInput(session, "active_params",
      selected=character(0))
  })
  observeEvent(input$btn_select_core, {
    updateCheckboxGroupInput(session, "active_params",
      selected=c("f0_mean","f0_sd","f0_range","f0_st",
                 "intensity_mean","hnr","speech_rate"))
  })

  active_params <- reactive({
    req(rv$features)
    sel <- input$active_params %||% names(PARAM_LABELS)
    intersect(sel, names(rv$features))
  })

  output$p_desc <- renderPlotly({
    req(rv$features)
    plot_descriptive(rv$features,input$desc_p,input$desc_by)
  })
  output$dt_desc <- renderDT({
    req(rv$features); p <- input$desc_p
    rv$features |>
      filter(speaker %in% c("man","woman","therapist")) |>
      drop_na(all_of(p)) |>
      group_by(couple,therapy_num,speaker) |>
      summarise(M=round(mean(.data[[p]],na.rm=TRUE),3),
                SD=round(sd(.data[[p]],na.rm=TRUE),3),
                n=n(),.groups="drop") |>
      dt_analysis()
  })
  output$p_heat <- renderPlotly({
    req(rv$features)
    params <- if (length(input$heat_p) > 0) input$heat_p else active_params()
    params <- intersect(params, active_params())
    if (length(params) == 0) params <- active_params()
    plot_heatmap(rv$features, params)
  })
  # Sync heat_p checkboxes when active_params changes
  observe({
    req(rv$features)
    ap <- active_params()
    updateCheckboxGroupInput(session, "heat_p",
      choices  = setNames(names(PARAM_LABELS), PARAM_LABELS),
      selected = intersect(input$heat_p %||% ap, ap))
  })
  output$p_traj <- renderPlotly({
    req(rv$features)
    param <- input$traj_p
    if (is.null(param) || !param %in% active_params())
      param <- active_params()[1]
    plot_trajectories(rv$features, param, input$traj_pairs)
  })
  observe({
    req(rv$features)
    ap <- active_params()
    updateSelectInput(session, "traj_p",
      choices  = setNames(ap, sapply(ap, \(p) PARAM_LABELS[p]%||%p)),
      selected = if (input$traj_p %in% ap) input$traj_p else ap[1])
  })

  # ── Prosodic profile ────────────────────────────────────
  observe({
    req(rv$features)
    couples <- sort(unique(as.character(rv$features$couple)))
    updateSelectInput(session, "couple_profile",
      choices = c("All", couples), selected = "All")
  })

  output$p_profile_heat <- renderPlotly({
    req(rv$features)
    roles <- if (length(input$roles_profile) > 0) input$roles_profile
             else c("man","woman","therapist")
    df <- rv$features |> filter(speaker %in% roles)
    cp <- input$couple_profile
    if (!is.null(cp) && cp != "All") df <- df |> filter(couple == cp)
    params <- input$profile_params
    if (is.null(params) || length(params) == 0)
      params <- c("f0_mean","f0_sd","f0_range","intensity_mean","speech_rate","hnr")
    params <- intersect(params, names(df))
    if (length(params) == 0) return(plot_ly() |> plotly_white())
    grp_vars <- intersect(input$group_profile %||% "speaker",
                          c("speaker","couple","therapy_num"))
    if (length(grp_vars) == 0) grp_vars <- "speaker"
    use_z  <- isTRUE(input$zscore_profile)
    summ <- df |>
      group_by(across(all_of(grp_vars))) |>
      summarise(across(all_of(params), ~mean(.x, na.rm=TRUE)), .groups="drop")
    grp_labels <- apply(summ[, grp_vars, drop=FALSE], 1,
                        \(r) paste(r, collapse=" · "))
    mat <- summ |> select(all_of(params)) |> as.matrix()
    rownames(mat) <- grp_labels
    mat_disp <- if (use_z) {
      z <- scale(mat); z[is.nan(z)|is.infinite(z)] <- 0; z
    } else mat
    ylabs <- sapply(params, \(p) PARAM_LABELS[p] %||% p)
    cell_txt <- if (use_z)
      matrix(paste0(round(t(mat),1), "<br>z=", sprintf("%.2f", t(mat_disp))),
             nrow=length(params), ncol=nrow(summ))
    else
      matrix(as.character(round(t(mat),1)),
             nrow=length(params), ncol=nrow(summ))
    grp_title <- paste(c(speaker="Role",couple="Couple",
                         therapy_num="Session")[grp_vars], collapse=" × ")
    p <- plot_ly(
      z=t(mat_disp), x=grp_labels, y=ylabs,
      text=cell_txt,
      texttemplate="%{text}",
      textfont=list(size=13, family="Inter", color="#222222"),
      type="heatmap",
      colorscale=list(c(0,"#3b4cc0"),c(.5,"#f7f7f7"),c(1,"#b40426")),
      colorbar=list(thickness=10, len=0.8, x=1.01,
                    title=list(text=if(use_z)"z" else "mean", side="right"),
                    tickfont=list(size=9)),
      hovertemplate=paste0("%{x}<br>%{y}<br>%{text}<extra></extra>"))
    if (use_z) p <- p |> layout(zmin=-2, zmax=2)
    p |> dark_layout(title=paste0("Prosodic profile — by ", grp_title))
  })

  output$dt_profile <- renderDT({
    req(rv$features)
    roles <- if (length(input$roles_profile) > 0) input$roles_profile
             else c("man","woman","therapist")
    df <- rv$features |> filter(speaker %in% roles)
    cp <- input$couple_profile
    if (!is.null(cp) && cp != "All") df <- df |> filter(couple == cp)
    params <- input$profile_params
    if (is.null(params) || length(params) == 0)
      params <- c("f0_mean","f0_sd","f0_range","intensity_mean","speech_rate","hnr")
    params <- intersect(params, names(df))
    if (length(params) == 0) return(data.frame())
    grp_vars <- intersect(input$group_profile %||% "speaker",
                          c("speaker","couple","therapy_num"))
    if (length(grp_vars) == 0) grp_vars <- "speaker"
    df |>
      group_by(across(all_of(grp_vars))) |>
      summarise(across(all_of(params),
        list(M=~round(mean(.x,na.rm=TRUE),2),
             SD=~round(sd(.x,na.rm=TRUE),2)),
        .names="{.col}_{.fn}"), .groups="drop") |>
      dt_analysis()
  })

  observeEvent(input$btn_lmm,{
    req(rv$features)
    withProgress(message="Adjusting LMMs…",value=0,{
      ap  <- active_params()
      feat_lmm <- rv$features
      # Keep only active params + required cols
      keep_cols <- c("couple","therapy","therapy_num","speaker","quantile",
                     "is_overlap","seg_start","seg_end","speaker_id",
                     intersect(ap, names(rv$features)))
      feat_lmm <- feat_lmm |> select(any_of(keep_cols))
      res        <- run_lmm(feat_lmm,
                            isTRUE(input$lmm_q),isTRUE(input$lmm_ov),
                            input$lmm_opt%||%"bobyqa",
                            input$lmm_pc%||%"BH")
      rv$lmm     <- res$lmm
      rv$posthoc <- res$posthoc
      incProgress(1)
    })
    showNotification("LMMs calculated.",type="message",duration=2)
  })

  output$p_lmm_heat <- renderPlotly({ req(rv$lmm); plot_lmm_heatmap(rv$lmm) })
  output$dt_lmm <- renderDT({
    req(rv$lmm)
    rv$lmm |> mutate(across(where(is.double),~round(.,3))) |>
      select(Parameter=label,Effect=efecto,F=F_value,
             `gl n`=df_num,`gl d`=df_den,
             p=p_value,`p adj.`=p_adj,`η²p`=eta2_p,Sig=sig) |>
      dt_analysis() |>
      formatStyle("Sig",backgroundColor=styleEqual(
        c("***","**","*","ns"),
        c("rgba(82,201,122,.15)","rgba(82,201,122,.08)",
          "rgba(232,160,69,.1)","transparent")))
  })
  output$dt_ph <- renderDT({
    req(rv$posthoc)
    rv$posthoc |> mutate(across(where(is.double),~round(.,4))) |>
      dt_analysis()
  })
  output$p_ov_evo   <- renderPlotly({req(rv$overlaps);plot_overlap_evo(rv$overlaps)})
  output$p_ov_turns <- renderPlotly({req(rv$overlaps);plot_turns(rv$overlaps)})
  output$dt_ov <- renderDT({
    req(rv$overlaps)
    dt_analysis(rv$overlaps)
  })
  output$p_diag <- renderPlotly({
    req(rv$features)
    df <- rv$features |>
      filter(speaker %in% c("man","woman","therapist")) |>
      drop_na(f0_mean)
    plot_ly(df,x=~paste(couple,therapy),y=~f0_mean,color=~speaker,
            colors=SP[c("man","woman","therapist")],
            type="box",boxpoints="outliers") |>
      dark_layout(xlab="",ylab="F0 (Hz)",
                  title="F0 by Session and role — assignment validation") |>
      layout(xaxis=list(tickangle=-45))
  })
  # ── Role correction ──────────────────────────────────

  output$dt_role_fix <- renderDT({
    req(rv$features)
    df <- rv$features |>
      group_by(couple, therapy, therapy_num, speaker_id, speaker) |>
      summarise(n_segs=n(), f0_media=round(mean(f0_mean,na.rm=TRUE),1),
                .groups="drop") |>
      arrange(couple, therapy_num, speaker_id)

    datatable(df,
      editable = list(target="cell", disable=list(columns=c(0,1,2,3,5,6))),
      selection = "none",
      options   = list(pageLength=20, dom="tip"),
      rownames  = FALSE,
      colnames  = c("couple","Session","N°","Speaker ID",
                    "Role (editable)","N segs","F0 mean (Hz)")
    )
  })

  observeEvent(input$dt_role_fix_cell_edit, {
    info <- input$dt_role_fix_cell_edit
    # col 4 (0-indexed) = speaker
    if (info$col == 4) {
      df_fix <- rv$features |>
        group_by(couple, therapy, therapy_num, speaker_id, speaker) |>
        summarise(.groups="drop") |>
        arrange(couple, therapy_num, speaker_id)
      row      <- info$row
      new_rol  <- trimws(info$value)
      old_row  <- df_fix[row, ]
      # Apply to rv$features
      rv$features <- rv$features |>
        mutate(speaker = if_else(
          couple     == old_row$couple &
          therapy    == old_row$therapy &
          speaker_id == old_row$speaker_id,
          factor(new_rol, levels=levels(speaker)),
          speaker
        ))
    }
  })

  observeEvent(input$btn_apply_roles, {
    req(rv$features)
    # Reescribir CSVs con roles corregidos
    tryCatch({
      out_dir <- file.path(getwd(), "transcriptions")
      if (dir.exists(out_dir)) {
        for (orig in unique(rv$features$couple)) {
          # Find matching features files
          csvs <- list.files(out_dir, pattern="_features\\.csv$", full.names=TRUE)
          for (csv_f in csvs) {
            old_df <- tryCatch(read_csv(csv_f, show_col_types=FALSE), error=\(e) NULL)
            if (is.null(old_df)) next
            # Update speaker where couple matches
            new_df <- old_df |>
              left_join(
                rv$features |>
                  distinct(couple, therapy, speaker_id, speaker_new=as.character(speaker)),
                by=c("couple","therapy","speaker_id")
              ) |>
              mutate(speaker = if_else(!is.na(speaker_new),
                                        speaker_new, as.character(speaker))) |>
              select(-speaker_new)
            write_csv(new_df, csv_f)
          }
        }
      }
      showNotification("✓ Roles corrected and CSVs updated.",
                       type="message", duration=3)
    }, error=\(e)
      showNotification(paste("Error:", conditionMessage(e)), type="error"))
  })

  output$dt_roles <- renderDT({
    req(rv$features)
    rv$features |>
      group_by(couple, therapy, speaker_id, speaker) |>
      summarise(n_segs=n(), f0_mean_hz=round(mean(f0_mean,na.rm=TRUE),1),
                .groups="drop") |>
      arrange(couple, therapy, speaker) |>
      datatable(options=list(pageLength=15,dom="tip"),rownames=FALSE)
  })

  # ── Mass role reassignment ─────────────────────────
  output$dt_bulk_roles <- renderDT({
    req(rv$features)
    # One row per couple × therapy × speaker_id
    df <- rv$features |>
      group_by(couple=couple, Session=therapy, `N°`=therapy_num,
               `Speaker ID`=speaker_id) |>
      summarise(
        `Current role`  = as.character(first(speaker)),
        `N segments` = n(),
        `F0 mean`    = round(mean(f0_mean, na.rm=TRUE), 1),
        .groups = "drop"
      ) |>
      arrange(couple, `N°`, `Speaker ID`) |>
      mutate(`Correct role` = `Current role`)

    locked <- which(!names(df) %in% "Correct role") - 1L
    datatable(df,
      editable  = list(target="cell",
                       disable=list(columns=as.list(locked))),
      selection = "none",
      options   = list(pageLength=30, dom="tip"),
      rownames  = FALSE)
  })

  bulk_role_edits <- reactiveVal(tibble(
    couple=character(), therapy=character(),
    speaker_id=character(), new_rol=character()))

  observeEvent(input$dt_bulk_roles_cell_edit, {
    info <- input$dt_bulk_roles_cell_edit
    req(rv$features)
    df <- rv$features |>
      group_by(couple, therapy, therapy_num, speaker_id) |>
      summarise(`Current role`=as.character(first(speaker)),
                .groups="drop") |>
      arrange(couple, therapy_num, speaker_id)
    if (info$row > nrow(df)) return()
    r <- df[info$row, ]
    new_rol <- trimws(as.character(info$value))
    # Store the edit
    edits <- bulk_role_edits()
    edits <- bind_rows(edits,
      tibble(couple=r$couple, therapy=r$therapy,
             speaker_id=r$speaker_id, new_rol=new_rol))
    bulk_role_edits(edits)
  })

  observeEvent(input$btn_bulk_roles, {
    edits <- bulk_role_edits()
    req(nrow(edits) > 0, rv$features)
    tryCatch({
      feat <- rv$features
      for (i in seq_len(nrow(edits))) {
        e <- edits[i, ]
        mask <- as.character(feat$couple)     == e$couple &
                as.character(feat$therapy)    == e$therapy &
                as.character(feat$speaker_id) == e$speaker_id
        feat$speaker[mask] <- factor(e$new_rol,
          levels=levels(feat$speaker))
      }
      rv$features <- feat
      bulk_role_edits(tibble(
        couple=character(), therapy=character(),
        speaker_id=character(), new_rol=character()))

      # Rewrite affected features CSVs
      out_dir <- file.path(getwd(), "transcriptions")
      if (dir.exists(out_dir)) {
        for (par in unique(edits$couple)) {
          csvs <- list.files(out_dir,
            pattern=paste0("(?i)", par, ".*_features\\.csv$"),
            full.names=TRUE, perl=TRUE)
          for (f in csvs) {
            old_df <- tryCatch(
              read_csv(f, show_col_types=FALSE), error=\(e) NULL)
            if (is.null(old_df)) next
            for (i in seq_len(nrow(edits))) {
              e <- edits[i, ]
              if (!grepl(e$couple, f, fixed=TRUE)) next
              mask <- as.character(old_df$speaker_id) == e$speaker_id &
                      as.character(old_df$therapy)    == e$therapy
              old_df$speaker[mask] <- e$new_rol
            }
            write_csv(old_df, f)
          }
        }
      }
      showNotification(
        sprintf("✓ %d reassignments applied.", nrow(edits)),
        type="message", duration=3)
    }, error=function(e)
      showNotification(paste("Error:", conditionMessage(e)),
                       type="error", duration=5))
  })

  # ── Prosodic profile ──────────────────────────────────────

  perfil_df <- reactive({
    req(rv$features)
    params <- input$perfil_params %||% c("f0_mean","intensity_mean","speech_rate")
    params <- intersect(params, names(rv$features))
    req(length(params) > 0)

    df <- rv$features |>
      filter(speaker %in% c("man","woman","therapist"))

    if (!is.null(input$perfil_couple) && input$perfil_couple != "All")
      df <- filter(df, couple == input$perfil_couple)

    grupo <- input$perfil_grupo %||% "couple"

    df |>
      group_by(across(all_of(c(grupo, "speaker")))) |>
      summarise(
        across(all_of(params),
               list(m  = \(x) round(mean(x,  na.rm=TRUE), 3),
                    sd = \(x) round(sd(x,    na.rm=TRUE), 3)),
               .names="{.col}__{.fn}"),
        n_segs = n(),
        .groups="drop"
      )
  })

  output$p_perfil_heat <- renderPlotly({
    req(perfil_df())
    df      <- perfil_df()
    params  <- input$perfil_params %||% c("f0_mean","intensity_mean","speech_rate")
    params  <- intersect(params, names(rv$features))
    grupo   <- input$perfil_grupo %||% "couple"
    use_z   <- isTRUE(input$perfil_zscore)

    # Build matrix: rows = parameters, cols = group × speaker
    m_cols  <- paste0(params, "__m")
    m_cols  <- intersect(m_cols, names(df))

    mat_df  <- df |>
      unite("grupo_hab", all_of(c(grupo, "speaker")), sep=" · ") |>
      select(grupo_hab, all_of(m_cols)) |>
      column_to_rownames("grupo_hab")

    mat <- t(as.matrix(mat_df))
    rownames(mat) <- sapply(rownames(mat), \(p) {
      p2 <- sub("__m$","",p)
      PARAM_LABELS[p2] %||% p2
    })

    if (use_z) {
      mat <- t(scale(t(mat)))  # z-score per parameter (rows)
      zlab <- " (z-score)"
    } else {
      zlab <- ""
    }

    plot_ly(
      z         = mat,
      x         = colnames(mat),
      y         = rownames(mat),
      type      = "heatmap",
      colorscale= list(c(0,"#1e4a7a"), c(.5,"#f5f6fa"), c(1,"#8a2020")),
      zmid      = 0,
      text      = round(mat, 2),
      hovertemplate = "%{x}<br>%{y}: %{text}<extra></extra>"
    ) |>
      layout(
        title  = list(text=paste0("Prosodic profile", zlab),
                      font=list(size=13)),
        xaxis  = list(title="", tickangle=-35),
        yaxis  = list(title=""),
        margin = list(l=160, b=120),
        paper_bgcolor="#f5f6fa",
        plot_bgcolor ="#f5f6fa",
        font   = list(family="IBM Plex Sans", size=11)
      )
  })

  output$dt_perfil <- renderDT({
    req(perfil_df())
    df     <- perfil_df()
    params <- input$perfil_params %||% c("f0_mean","intensity_mean","speech_rate")
    params <- intersect(params, names(rv$features))

    # Reshape to long: one row per group × speaker × parameter
    df_long <- df |>
      pivot_longer(
        cols      = matches("__m$|__sd$"),
        names_to  = c("param",".value"),
        names_sep = "__"
      ) |>
      mutate(
        param = sapply(param, \(p) PARAM_LABELS[p] %||% p),
        `Media_SD` = sprintf("%.2f +/- %.2f", m, sd)
      )

    grupo <- input$perfil_grupo %||% "couple"
    df_long |>
      select(grupo=.data[[grupo]], speaker, Parameter=param, `Media_SD`, n_segs) |>
      dt_analysis(filter = "top")
  })

  # Update couple selector in profile
  observe({
    req(rv$features)
    couples <- c("All", sort(unique(as.character(rv$features$couple))))
    updateSelectInput(session, "perfil_couple", choices=couples)
  })

  # ── Load previous analysis ─────────────────────────────
  observeEvent(input$btn_load_prev, {
    req(input$file_prev_features)
    tryCatch({
      # Leer y concatenar todos los CSVs de features subidos
      feat <- bind_rows(lapply(
        input$file_prev_features$datapath,
        function(p) read_csv(p, show_col_types=FALSE)
      )) |>
        mutate(
          speaker    = factor(speaker,
                               c("man","woman","therapist","unknown")),
          quantile     = factor(quantile, c("Q1","Q2","Q3","Q4")),
          is_overlap  = as.logical(is_overlap),
          therapy_num = as.integer(therapy_num)
        ) |>
        # Recalcular f0_st si no existe o es todo NA
        (\(df) {
          has_f0st <- "f0_st" %in% names(df) && !all(is.na(df$f0_st))
          has_f0   <- "f0_mean" %in% names(df)
          if (!has_f0st && has_f0)
            df |>
              mutate(f0_st = if ("f0_range" %in% names(df))
                12 * log2(1 + pmax(f0_range, 0) / pmax(f0_mean, 1))
              else NA_real_)
          else df
        })()
      rv$features <- feat
      log_add("✓ Features loaded: ", nrow(rv$features),
              " segments from ", nlevels(factor(feat$couple)), " couples.")

      # Leer y concatenar transcripciones (opcional)
      if (!is.null(input$file_prev_transcript) &&
          nrow(input$file_prev_transcript) > 0) {
        rv$transcript <- bind_rows(lapply(
          input$file_prev_transcript$datapath,
          function(p) read_csv(p, show_col_types=FALSE)
        ))
        log_add("✓ Transcription loaded: ",
                nrow(rv$transcript), " segments.")
      }
      # Load overlappings if uploaded
      if (!is.null(input$file_prev_overlaps) &&
          nrow(input$file_prev_overlaps) > 0) {
        rv$overlaps <- bind_rows(lapply(
          input$file_prev_overlaps$datapath,
          function(p) read_csv(p, show_col_types=FALSE)
        ))
        log_add("✓ overlappings loaded: ", nrow(rv$overlaps), " rows.")
      }
      # Reconstruir sess_meta desde features si no se sube CSV de metadatos
      if (!is.null(input$file_prev_meta) && nrow(input$file_prev_meta) > 0) {
        rv$sess_meta <- read_csv(input$file_prev_meta$datapath,
                                 show_col_types=FALSE)
        log_add("✓ Metadata loaded.")
      } else {
        # Reconstruir desde features: una fila por couple × therapy
        rv$sess_meta <- feat |>
          distinct(couple, therapy, therapy_num) |>
          arrange(couple, therapy_num) |>
          mutate(lugar="", fecha="", notas="",
                 is_last=FALSE, idioma="es",
                 name=paste0(couple,"_",therapy,".wav"))
        log_add("✓ Metadatos reconstruidos desde features.")
      }

      # Reconstruir spk_meta: speaker_id → rol asignado por archivo
      rv$spk_meta <- feat |>
        distinct(couple, therapy, therapy_num, speaker_id, speaker) |>
        arrange(couple, therapy_num, speaker_id) |>
        mutate(rol = as.character(speaker))

      showNotification(
        paste0("Analysis loaded: ", nrow(feat), " segments."),
        type="message", duration=3)
      shinydashboard::updateTabItems(session, "sidebar_menu", "tab_meta")
    }, error = function(e) {
      showNotification(paste("Error loading:", conditionMessage(e)),
                       type="error", duration=5)
    })
  })

  # ── Transcription ─────────────────────────────────────────
  observe({
    req(rv$features)
    req("text" %in% names(rv$features))
    updateSelectInput(session, "trans_couple",
      choices = sort(unique(as.character(rv$features$couple))))
  })

  observe({
    req(rv$features, input$trans_couple)
    therapys <- rv$features |>
      filter(couple == input$trans_couple) |>
      pull(therapy_num) |> unique() |> sort()
    updateSelectInput(session, "trans_therapy", choices = therapys)
  })

  # Selectors update from transcript OR features (whichever has data)
  observe({
    df <- if (!is.null(rv$transcript) && nrow(rv$transcript) > 0)
            rv$transcript
          else if (!is.null(rv$features) && "text" %in% names(rv$features))
            rv$features
          else NULL
    req(df)
    updateSelectInput(session, "trans_couple",
      choices = sort(unique(as.character(df$couple))))
  })

  observe({
    df <- if (!is.null(rv$transcript) && nrow(rv$transcript) > 0)
            rv$transcript
          else if (!is.null(rv$features) && "text" %in% names(rv$features))
            rv$features
          else NULL
    req(df, input$trans_couple)
    nums <- df |> filter(couple==input$trans_couple) |>
      pull(therapy_num) |> unique() |> sort()
    updateSelectInput(session, "trans_therapy", choices=nums)
  })

  trans_filtered <- reactive({
    # Prefer rv$transcript (Whisper segments) over rv$features (diarization segments)
    has_trans <- !is.null(rv$transcript) && nrow(rv$transcript) > 0
    req(input$trans_couple, input$trans_therapy)

    if (has_trans) {
      df <- rv$transcript |>
        filter(couple == input$trans_couple,
               therapy_num == as.integer(input$trans_therapy))
      if (input$trans_speaker != "All")
        df <- filter(df, speaker == input$trans_speaker)
      if (isTRUE(input$trans_show_ov))
        df <- filter(df, is_overlap)
      df |> arrange(seg_start) |>
        transmute(
          Inicio       = format(round(seg_start, 2), nsmall=2),
          Final        = format(round(seg_end,   2), nsmall=2),
          speaker     = speaker,
          overlapping = ifelse(is_overlap, "⚠", ""),
          text        = text
        )
    } else {
      req(rv$features, "text" %in% names(rv$features))
      df <- rv$features |>
        filter(couple == input$trans_couple,
               therapy_num == as.integer(input$trans_therapy),
               !is.na(text), nchar(text) > 0)
      if (input$trans_speaker != "All")
        df <- filter(df, as.character(speaker) == input$trans_speaker)
      df |> arrange(seg_start) |>
        transmute(
          Inicio       = sprintf("%.2f", seg_start),
          Final        = sprintf("%.2f", seg_end),
          speaker     = speaker,
          overlapping = ifelse(is_overlap, "⚠", ""),
          text        = text
        )
    }
  })

  # Timestamps for the selected row — kept separately from display df
  trans_timestamps <- reactive({
    has_trans <- !is.null(rv$transcript) && nrow(rv$transcript) > 0
    req(input$trans_couple, input$trans_therapy)
    if (has_trans) {
      rv$transcript |>
        filter(couple == input$trans_couple,
               therapy_num == as.integer(input$trans_therapy)) |>
        arrange(seg_start) |>
        select(seg_start, seg_end)
    } else if (!is.null(rv$features) && "text" %in% names(rv$features)) {
      rv$features |>
        filter(couple == input$trans_couple,
               therapy_num == as.integer(input$trans_therapy),
               !is.na(text), nchar(text) > 0) |>
        arrange(seg_start) |>
        select(seg_start, seg_end)
    } else tibble(seg_start=numeric(), seg_end=numeric())
  })

  output$dt_trans <- renderDT({
    df <- trans_filtered()
    datatable(df,
      selection = list(mode="single", target="row"),
      options   = list(
        pageLength=30, dom="tip",
        scrollX=TRUE,
        columnDefs=list(
          list(width="70px",  targets=0),   # Inicio
          list(width="70px",  targets=1),   # Final
          list(width="90px",  targets=2),   # speaker
          list(width="40px",  targets=3),   # overlapping
          list(width="auto",  targets=4)    # text — fill remaining
        )
      ),
      rownames = FALSE
    )
  })

  # ── Inline row editor for transcription ──────────────────
  output$ui_trans_row_editor <- renderUI({
    row <- input$dt_trans_rows_selected
    if (is.null(row) || length(row) == 0) return(NULL)
    df <- trans_filtered()
    if (row > nrow(df)) return(NULL)
    cur_hab <- as.character(df$speaker[row])
    cur_ov  <- as.character(df$overlapping[row])
    cur_txt <- as.character(df$text[row])
    tags$div(
      style="background:#f9f9f9;border:1px solid #e0e0e0;border-radius:4px;padding:10px;margin-bottom:8px",
      tags$b(sprintf("Fila %d — editar:", row)),
      tags$br(), tags$br(),
      shiny::fluidRow(
        shiny::column(3,
          selectInput("trans_edit_hab", "speaker",
            choices  = c("man","woman","therapist","unknown"),
            selected = cur_hab)
        ),
        shiny::column(3,
          selectInput("trans_edit_ov", "overlapping",
            choices  = c("no"="", "yes"="⚠ overlapping"),
            selected = cur_ov)
        ),
        shiny::column(4,
          textAreaInput("trans_edit_txt", "text",
            value=cur_txt, rows=2, resize="vertical")
        ),
        shiny::column(2,
          br(),
          actionButton("btn_apply_trans_roles",
            tagList(shiny::icon("check"), " Save"),
            class="btn btn-warning btn-sm w-100")
        )
      )
    )
  })

  trans_role_edits <- reactiveVal(list())


  observeEvent(input$btn_apply_trans_roles, {
    row <- input$dt_trans_rows_selected
    req(row, rv$transcript, input$trans_edit_hab)
    tryCatch({
      mask <- rv$transcript$couple == input$trans_couple &
              rv$transcript$therapy_num == as.integer(input$trans_therapy)
      idx  <- which(mask)
      if (row > length(idx)) return()
      real_idx <- idx[row]
      rv$transcript$speaker[real_idx]   <- input$trans_edit_hab
      rv$transcript$is_overlap[real_idx] <- nchar(input$trans_edit_ov) > 0
      if (!is.null(input$trans_edit_txt) && nchar(trimws(input$trans_edit_txt)) > 0)
        rv$transcript$text[real_idx] <- trimws(input$trans_edit_txt)

      out_dir <- file.path(getwd(), "transcriptions")
      if (dir.exists(out_dir)) {
        csvs <- list.files(out_dir,
          pattern=paste0(input$trans_couple,".*_transcript\\.csv$"),
          full.names=TRUE)
        for (f in csvs)
          write_csv(rv$transcript |>
            filter(couple==input$trans_couple,
                   therapy_num==as.integer(input$trans_therapy)), f)
      }
      showNotification(
        sprintf("Fila %d: %s stored.", row, input$trans_edit_hab),
        type="message", duration=2)
    }, error=function(e)
      showNotification(paste("Error:", conditionMessage(e)),
                       type="error", duration=4))
  })

  # ── Reproductor de audio ─────────────────────────────────
  output$ui_audio_player <- renderUI({
    row <- input$dt_trans_rows_selected
    if (is.null(row) || length(row) == 0) return(NULL)

    df <- trans_filtered()
    if (row > nrow(df)) return(NULL)

    ts <- trans_timestamps()
    if (row > nrow(ts)) return(NULL)
    seg_start <- ts$seg_start[row]
    seg_end   <- ts$seg_end[row]
    if (is.na(seg_start)) return(NULL)

    par <- input$trans_couple  %||% ""
    ter <- input$trans_therapy %||% ""

    # Find source audio in www/audios/
    audio_folder <- file.path(getwd(), "www", "audios")
    all_files    <- list.files(audio_folder, full.names=TRUE)
    candidates   <- all_files[
      grepl(par, basename(all_files), ignore.case=TRUE) &
      grepl(paste0("therapy", ter), basename(all_files), ignore.case=TRUE) &
      grepl("\\.(wav|mp3|flac)$", basename(all_files), ignore.case=TRUE)
    ]
    if (length(candidates) == 0)
      candidates <- all_files[
        grepl(par, basename(all_files), ignore.case=TRUE) &
        grepl(ter,  basename(all_files), ignore.case=TRUE) &
        grepl("\\.(wav|mp3|flac)$", basename(all_files), ignore.case=TRUE)
      ]

    if (length(candidates) == 0) return(tags$div(
      class="callout", style="font-size:11px;padding:6px 10px;margin-bottom:6px",
      sprintf("Audio not found: www/audios/%s_therapy%s.wav", par, ter)))

    src_file <- candidates[1]
    dur <- max(0.1, (seg_end %||% (seg_start + 3)) - seg_start)

    # Cut the segment with av::av_audio_convert into www/audios/clips/
    if (!requireNamespace("av", quietly=TRUE)) return(tags$div(
      class="callout", style="font-size:11px;padding:6px 10px",
      "Instala el paquete av: install.packages('av')"))

    clip_dir <- file.path(getwd(), "www", "audios", "clips")
    if (!dir.exists(clip_dir)) dir.create(clip_dir, recursive=TRUE)
    clip_name <- sprintf("clip_%s_%s_%07.2f.wav", par, ter, seg_start)
    clip_path <- file.path(clip_dir, clip_name)

    if (!file.exists(clip_path)) {
      tryCatch({
        av::av_audio_convert(
          audio   = src_file,
          output  = clip_path,
          format  = "wav",
          start_time = seg_start,
          total_time = dur + 0.2
        )
      }, error = function(e) {
        message("av clip error: ", e$message)
      })
    }

    if (!file.exists(clip_path)) return(tags$div(
      class="callout", style="font-size:11px;padding:6px 10px",
      paste("No se pudo cortar el segment:", clip_name)))

    # Serve from www/ — clip URL is audios/clips/clip_...wav
    clip_url <- paste0("audios/clips/", utils::URLencode(clip_name))

    tags$div(
      class="callout callout-info",
      style="padding:6px 10px;margin-bottom:8px",
      tags$b(sprintf("[%.2fs – %.2fs]  ", seg_start,
                     seg_end %||% (seg_start+dur))),
      as.character(df$speaker[row]),
      tags$br(),
      tags$audio(
        src      = clip_url,
        controls = NA,
        autoplay = NA,
        style    = "width:100%;margin-top:4px"
      )
    )
  })

  # Download CSV only if there is a transcription
  output$dl_trans_csv_ui <- renderUI({
    req(rv$features, "text" %in% names(rv$features))
    downloadButton("dl_trans_csv", "⬇ CSV", class="btn-ghost")
  })
  output$dl_trans_csv <- downloadHandler(
    filename = \() paste0(input$trans_couple, "_",
                          input$trans_therapy, "_transcripcion.csv"),
    content = function(f) write_csv(trans_filtered(), f))

  output$dl_trans_features <- downloadHandler(
    filename = \() paste0(input$trans_couple, "_",
                          input$trans_therapy, "_features.csv"),
    content = function(f) {
      req(rv$features)
      rv$features |>
        filter(couple==input$trans_couple,
               therapy_num==as.integer(input$trans_therapy)) |>
        write_csv(f)
    })

  # ── Segmentation ─────────────────────────────────────────
  observe({
    df <- rv$segs %||% rv$features
    req(df)
    updateSelectInput(session, "seg_couple",
      choices=sort(unique(as.character(df$couple))))
  })
  observe({
    df <- rv$segs %||% rv$features
    req(df, input$seg_couple)
    nums <- df |> filter(couple==input$seg_couple) |>
      pull(therapy_num) |> unique() |> sort()
    updateSelectInput(session, "seg_therapy", choices=nums)
  })

  seg_filtered <- reactive({
    df <- rv$segs %||% rv$features
    req(df, input$seg_couple, input$seg_therapy)
    df <- df |>
      filter(couple==input$seg_couple,
             therapy_num==as.integer(input$seg_therapy))
    if (!is.null(input$seg_speaker) && input$seg_speaker != "All")
      df <- filter(df, as.character(speaker)==input$seg_speaker)
    s_col <- if ("start" %in% names(df)) "start" else "seg_start"
    e_col <- if ("end"   %in% names(df)) "end"   else "seg_end"
    df |> arrange(.data[[s_col]]) |>
      transmute(
        Inicio   = sprintf("%.2f", .data[[s_col]]),
        Final    = sprintf("%.2f", .data[[e_col]]),
        speaker = as.character(speaker),
        Dur      = sprintf("%.1fs", .data[[e_col]] - .data[[s_col]])
      )
  })

  seg_timestamps <- reactive({
    df <- rv$segs %||% rv$features
    req(df, input$seg_couple, input$seg_therapy)
    df <- df |>
      filter(couple==input$seg_couple,
             therapy_num==as.integer(input$seg_therapy))
    if (!is.null(input$seg_speaker) && input$seg_speaker != "All")
      df <- filter(df, as.character(speaker)==input$seg_speaker)
    s_col <- if ("start" %in% names(df)) "start" else "seg_start"
    e_col <- if ("end"   %in% names(df)) "end"   else "seg_end"
    df |> arrange(.data[[s_col]]) |>
      transmute(seg_start=.data[[s_col]], seg_end=.data[[e_col]])
  })

  output$dt_seg <- renderDT({
    df <- seg_filtered()
    datatable(df,
      selection = list(mode="single", target="row"),
      options   = list(pageLength=50, dom="tip",
        columnDefs=list(
          list(width="70px", targets=0),
          list(width="70px", targets=1),
          list(width="90px", targets=2),
          list(width="60px", targets=3))),
      rownames=FALSE)
  })

  output$ui_seg_row_editor <- renderUI({
    row <- input$dt_seg_rows_selected
    if (is.null(row) || length(row)==0) return(NULL)
    df <- seg_filtered()
    if (row > nrow(df)) return(NULL)
    cur_hab <- df$speaker[row]
    tags$div(
      style="background:#f9f9f9;border:1px solid #e0e0e0;border-radius:4px;padding:8px;margin-bottom:8px",
      shiny::fluidRow(
        shiny::column(5,
          selectInput("seg_edit_hab","speaker",
            c("man","woman","therapist","unknown"), selected=cur_hab)
        ),
        shiny::column(4, br(),
          actionButton("btn_apply_seg_role",
            tagList(shiny::icon("check")," Store & extend"),
            class="btn btn-warning btn-sm")
        )
      )
    )
  })

  output$ui_seg_audio_player <- renderUI({
    row <- input$dt_seg_rows_selected
    if (is.null(row) || length(row)==0) return(NULL)
    ts <- seg_timestamps()
    if (row > nrow(ts)) return(NULL)
    seg_start <- ts$seg_start[row]
    seg_end   <- ts$seg_end[row]
    if (is.na(seg_start)) return(NULL)
    par <- input$seg_couple  %||% ""
    ter <- input$seg_therapy %||% ""
    audio_folder <- file.path(getwd(), "www", "audios")
    all_files    <- list.files(audio_folder, full.names=TRUE)
    cands <- all_files[
      grepl(par, basename(all_files), ignore.case=TRUE) &
      grepl(paste0("therapy", ter), basename(all_files), ignore.case=TRUE) &
      grepl("\\.(wav|mp3|flac)$", basename(all_files), ignore.case=TRUE)]
    if (length(cands)==0) return(tags$div(
      class="callout", style="font-size:11px;padding:6px",
      sprintf("Audio not found: www/audios/%s_therapy%s.wav", par, ter)))
    dur <- max(0.1, (seg_end %||% (seg_start+3)) - seg_start)
    clip_dir  <- file.path(getwd(), "www", "audios", "clips")
    if (!dir.exists(clip_dir)) dir.create(clip_dir, recursive=TRUE)
    clip_name <- sprintf("seg_%s_%s_%07.2f.wav", par, ter, seg_start)
    clip_path <- file.path(clip_dir, clip_name)
    if (!file.exists(clip_path) && requireNamespace("av", quietly=TRUE))
      tryCatch(av::av_audio_convert(cands[1], clip_path,
        format="wav", start_time=seg_start, total_time=dur+0.2),
        error=\(e) NULL)
    if (!file.exists(clip_path)) return(NULL)
    clip_url <- paste0("audios/clips/", utils::URLencode(clip_name))
    tags$div(class="callout callout-info",
      style="padding:6px 10px;margin-bottom:6px",
      tags$b(sprintf("[%.2fs – %.2fs]  ", seg_start, seg_end)),
      tags$br(),
      tags$audio(src=clip_url, controls=NA, autoplay=NA,
                 style="width:100%;margin-top:4px"))
  })

  observeEvent(input$btn_apply_seg_role, {
    row <- input$dt_seg_rows_selected
    req(row, input$seg_edit_hab)
    new_hab <- input$seg_edit_hab
    par <- input$seg_couple
    ter <- as.integer(input$seg_therapy)
    ts  <- seg_timestamps()
    if (row > nrow(ts)) return()
    t_start <- ts$seg_start[row]
    t_end   <- ts$seg_end[row]
    # Update rv$segs
    if (!is.null(rv$segs)) {
      m <- rv$segs$couple==par & rv$segs$therapy_num==ter &
           abs(rv$segs$start - t_start) < 0.05
      rv$segs$speaker[m] <- new_hab
    }
    # Update rv$features
    if (!is.null(rv$features)) {
      m <- rv$features$couple==par & rv$features$therapy_num==ter &
           rv$features$seg_start >= (t_start-0.1) &
           rv$features$seg_end   <= (t_end+0.1)
      rv$features$speaker[m] <- factor(new_hab, levels=levels(rv$features$speaker))
    }
    # Update rv$transcript
    if (!is.null(rv$transcript)) {
      m <- rv$transcript$couple==par & rv$transcript$therapy_num==ter &
           rv$transcript$seg_start >= (t_start-0.1) &
           rv$transcript$seg_end   <= (t_end+0.1)
      rv$transcript$speaker[m] <- new_hab
    }
    showNotification(
      sprintf("[%.2f–%.2f] → %s (propagado)", t_start, t_end, new_hab),
      type="message", duration=2)
  })

  # ── Convergencia ──────────────────────────────────────────

  # Reactive: construye el df de convergencia cuando hay features
  conv_df <- reactive({
    req(rv$features)
    tryCatch(build_convergence_df(rv$features),
             error = \(e) { message("conv_df error: ", e); NULL })
  })

  output$p_conv_z <- renderPlotly({
    req(conv_df())
    plot_zscores(conv_df(), input$conv_param_z %||% "f0_mean")
  })

  output$p_conv_rng <- renderPlotly({
    req(conv_df())
    plot_ranges(conv_df(), input$conv_param_rng %||% "f0_mean")
  })

  output$p_conv_dist <- renderPlotly({
    req(conv_df())
    param_dist <- input$conv_param_dist %||% "all"
    plot_convergence_dist(conv_df(), param=param_dist)
  })

  observeEvent(input$btn_conv_lmm, {
    req(conv_df())
    withProgress(message = "Adjusting LMMs convergence...", value=0, {
      rv$conv_lmm <- run_convergence_lmm(conv_df(),
                                          input$lmm_opt %||% "bobyqa")
      incProgress(1)
    })
    showNotification("LMMs of convergence calculated.",
                     type="message", duration=2)
  })

  output$dt_conv_lmm <- renderDT({
    req(rv$conv_lmm)
    rv$conv_lmm |>
      mutate(across(where(is.double), ~round(., 3))) |>
      select(Hypothesis=hipotesis, Parameter=parameter, Effect=efecto,
             F=F_value, `gl n`=df_num, `gl d`=df_den,
             p=p_value, `p adj.`=p_adj, Sig=sig) |>
      dt_analysis() |>
      formatStyle("Sig",
        backgroundColor=styleEqual(
          c("***","**","*","ns"),
          c("rgba(82,201,122,.15)","rgba(82,201,122,.08)",
            "rgba(232,160,69,.1)","transparent")))
  })

  output$dt_conv_table <- renderDT({
    req(conv_df())
    df <- conv_df() |>
      select(couple, therapy_num, speaker, n_segs,
             dist_therapist,
             matches("^z_f0|^z_intensity|^z_speech|^z_hnr"),
             matches("f0_mean__rng|intensity_mean__rng|speech_rate__rng")) |>
      mutate(across(where(is.double), ~round(., 3))) |>
      arrange(couple, therapy_num, speaker)
    dt_analysis(df)
  })

  # ── prosodic examples v2 → logic in attach_new_server() ──
  # Only the guard that shows a warning if there is no transcription is kept.
  output$ui_examples_guard <- renderUI({
    req(rv$features)
    has_trans <- "text" %in% names(rv$features) &&
                 any(!is.na(rv$features$text))
    if (!has_trans)
      ibox("Activate transcription in step 03 and reprocess ",
           "to see the prosodic examples.")
    else NULL
  })

  # ── Informe ───────────────────────────────────────────────
  output$txt_block <- renderText({
    req(rv$features)
    build_text_block(rv$features,rv$overlaps,rv$lmm,rv$posthoc,
                         input$active_params%||%names(PARAM_LABELS))
  })
  observeEvent(input$btn_copy,{
    req(rv$features)
    txt <- build_text_block(rv$features,rv$overlaps,rv$lmm,rv$posthoc,
                            input$active_params%||%names(PARAM_LABELS))
    # shinyjs needs a list with one element for params[0] in JS
    shinyjs::js$copyText(list(txt))
  })
  output$dl_ui <- renderUI({
    req(rv$features)
    downloadButton("btn_dl","⬇  Download HTML",class="btn-ghost")
  })
  output$dl_word_ui <- renderUI({
    req(rv$word_tmp)
    if (!file.exists(rv$word_tmp)) return(NULL)
    downloadButton("dl_report_word","⬇  Download Word",
                   class="btn-ghost mt-1")
  })
  output$btn_dl <- downloadHandler(
    filename=\() paste0("acustera_",format(Sys.time(),"%Y%m%d_%H%M"),".html"),
    content=function(f){
      html <- build_report(
        build_text_block(rv$features,rv$overlaps,rv$lmm,rv$posthoc,
                         input$active_params%||%names(PARAM_LABELS)),
        rv$features,rv$overlaps,rv$lmm,rv$posthoc,
        input$rep_notes%||%"", sess_meta=rv$sess_meta)
      writeLines(html, f)
      # Also save to reports/ folder
      tryCatch({
        out_dir <- file.path(getwd(), "reports")
        if (!dir.exists(out_dir)) dir.create(out_dir, recursive=TRUE)
        writeLines(html, file.path(out_dir,
          paste0("acustera_", format(Sys.time(),"%Y%m%d_%H%M"), ".html")))
        # Save key plots as PNG
        if (!is.null(rv$features)) {
          ap <- active_params()
          p1 <- plot_heatmap(rv$features, ap)
          save_plot_png(p1, paste0("heatmap_",format(Sys.time(),"%Y%m%d_%H%M"),".png"))
          p2 <- plot_trajectories(rv$features, ap[1])
          save_plot_png(p2, paste0("traj_",format(Sys.time(),"%Y%m%d_%H%M"),".png"))
        }
      }, error=\(e) message("Error saving to reports/: ", e$message))
    })

  # ── Word report ─────────────────────────────────────────────
  observeEvent(input$btn_report_word, {
    req(rv$features)
    if (!requireNamespace("officer",  quietly=TRUE) ||
        !requireNamespace("flextable",quietly=TRUE)) {
      showNotification(
        "Instala: install.packages(c('officer','flextable'))",
        type="error", duration=8)
      return()
    }
    withProgress(message="Generating Word document...", value=0, {
      tryCatch({
        doc <- officer::read_docx()
        ft  <- function(df, cap=NULL) {
          if (!is.null(cap))
            doc <<- officer::body_add_par(doc, cap, style="Normal")
          ft <- flextable::flextable(as.data.frame(df)) |>
                flextable::autofit() |>
                flextable::theme_booktabs()
          doc <<- flextable::body_add_flextable(doc, ft)
        }
        h1 <- function(t) doc <<- officer::body_add_par(doc, t, style="heading 1")
        h2 <- function(t) doc <<- officer::body_add_par(doc, t, style="heading 2")
        pp <- function(t) doc <<- officer::body_add_par(doc, t, style="Normal")
        br <- function()  doc <<- officer::body_add_par(doc, "", style="Normal")

        # Cover page
        h1("Acoustic analysis report — AcusTera")
        pp(paste("Date:", format(Sys.time(), "%d/%m/%Y %H:%M")))
        pp(paste("couples:", paste(unique(rv$features$couple), collapse=", ")))
        pp(paste("Sessions:", length(unique(rv$features$therapy_num))))
        pp(paste("segments:", nrow(rv$features)))
        br()

        # 1. Introduction
        h1("1. Introduction")
        pp("This report presents the results of the longitudinal acoustic analysis of couple therapy sessions. Prosodic parameters were extracted from each diaryd speech segment, and their changes were modeled using mixed-effects models (MEMs). The main hypothesis is that emotional regulation and prosodic convergence are indicators of therapeutic change.")
        br()
        incProgress(0.1)

        # 2. Methodology
        h1("2. Methodology")
        h2("2.1 Diarrhization and extraction pipeline")
        pp("Pipeline: (1) diarization with pyannote.audio speaker-diarization-3.1; (2) role assignment using voice embeddings and cosine similarity; (3) parameter extraction with librosa and Praat/parselmouth; (4) verbatim transcription with Whisper.")
        h2("2.2 Statistical analysis")
        pp("LMM per parameter: parameter ~ Session × speaker + quantile + (1|couple). Benjamini-Hochberg correction. Convergence: Euclidean distance in z-score space between patient and therapist per Session.")
        br()
        incProgress(0.2)

        # 3. Variables
        h1("3. Acoustic variables")
        pp("Table 1 describes the extracted acoustic parameters.")
        ft(tibble::tribble(
          ~Variable, ~Description, ~Unit,
          "F0 mean",    "Average fundamental frequency (vocal pitch)", "Hz",
          "F0 SD",      "Pitch variability",                     "Hz",
          "F0 range",   "range of F0 (max − min)",                  "Hz",
          "F0 st",      "F0 range in semitones",                  "st",
          "Intensity", "Average intensity (volume)",                 "dB",
          "HNR",        "Harmonics/noise ratio",                  "dB",
          "Speech rate","Speech rate (normalized)",               "z"
        ), "Table 1. Acoustic variables")
        br()
        incProgress(0.3)

        # 4. Corpus
        h1("4. Corpus description")
        if (!is.null(rv$sess_meta) && nrow(rv$sess_meta) > 0) {
          corpus <- rv$features |>
            group_by(couple=couple, Session=therapy_num) |>
            summarise(segments=n(),
                      `Mean dur. (s)`=round(mean(seg_end-seg_start,na.rm=TRUE),1),
                      .groups="drop")
          ft(corpus, "Table 2. Analyzed corpus.")
        }
        br()
        incProgress(0.4)

        # 5. Descriptive statistics
        h1("5. Descriptive statistics")
        pp("Mean and standard deviation of the main parameters by speaker and session.")
        params_w <- intersect(
          input$active_params %||% names(PARAM_LABELS),
          intersect(names(rv$features), c("f0_mean","f0_sd","intensity_mean","hnr","speech_rate")))
        if (length(params_w) > 0) {
          desc <- rv$features |>
            filter(speaker %in% c("man","woman","therapist")) |>
            group_by(couple=couple, Session=therapy_num, speaker=as.character(speaker)) |>
            summarise(across(all_of(params_w),
              \(x) paste0(round(mean(x,na.rm=TRUE),2),"±",round(sd(x,na.rm=TRUE),2)),
              .names="{.col}"), .groups="drop")
          ft(desc, "Table 3. Descriptive statistics (mean±SD).")
        }
        br()
        incProgress(0.5)

        # 6. Heatmap note
        h1("6. Acoustic Heatmap")
        pp("The heatmap displays the z-score for each parameter per session and speaker, normalized to the mean of all sessions. Positive values ​​(red) indicate above the mean; negative values ​​(blue) indicate below. This allows for the identification of patterns of overall change.")
        br()

        # 7. Prosodic profile
        h1("7. Prosodic profile")
        pp("The prosodic profile summarizes the vocal characteristics of each participant. Z-scores facilitate comparison between couples with different baselines (e.g., F0 differences by sex).")
        br()
        incProgress(0.6)

        # 8. LMM
        if (!is.null(rv$lmm) && nrow(rv$lmm) > 0) {
          h1("8. Mixed effects models")
          pp("Table 4 shows the significant effects (p_adj < .05) after BH correction.")
          sig <- rv$lmm |>
            filter(p_adj < .05) |>
            select(Parameter=label, Effect=efecto, F=F_value,
                   p=p_value, `p_adj`=p_adj, `η²p`=eta2_p, Sig=sig) |>
            mutate(across(where(is.double), \(x) round(x,3)))
          if (nrow(sig)>0) ft(sig, "Table 4. Significant effects.") else
            pp("No significant effects were found.")
          br()
        }
        incProgress(0.7)

        # 9. Convergencia
        if (!is.null(rv$conv_lmm) && nrow(rv$conv_lmm)>0) {
          h1("9. Prosodic convergence")
          pp("Convergence was measured as the reduction in Euclidean distance in z-score space between patients and therapists over the course of the sessions. Table 5 shows the convergence LMMs.")
          ft(rv$conv_lmm |>
            select(Parameter=parameter, Effect=efecto, F=F_value,
                   p=p_value, p_adj) |>
            mutate(across(where(is.double), \(x) round(x,3))),
            "Table 5. Convergence LMMs.")
          br()
        }
        incProgress(0.8)

        # 10. Notas
        notes <- input$rep_notes %||% ""
        if (nchar(trimws(notes)) > 0) {
          h1("10. Researcher's Notes")
          pp(notes)
        }
        incProgress(0.9)

        tmp <- tempfile(fileext=".docx")
        print(doc, target=tmp)
        rv$word_tmp <- tmp
        incProgress(1.0)
        showNotification("Word created Use '⬇ Download Word'.",
                         type="message", duration=4)
      }, error=function(e)
        showNotification(paste("Error Word:", conditionMessage(e)),
                         type="error", duration=6))
    })
  })

  output$dl_report_word <- downloadHandler(
    filename=\() paste0("acustera_",format(Sys.time(),"%Y%m%d_%H%M"),".docx"),
    content=function(f) {
      req(rv$word_tmp, file.exists(rv$word_tmp))
      file.copy(rv$word_tmp, f)
    })

  # ── Navigation: "Continue" buttons switch tab ──────────
  observeEvent(input$go_metadata,
    shinydashboard::updateTabItems(session, "sidebar_menu", "tab_meta"))
  observeEvent(input$go_process,
    shinydashboard::updateTabItems(session, "sidebar_menu", "tab_process"))
  observeEvent(input$save_sess,
    showNotification("Saved.", type="message", duration=2))

  # ── Status card in configuration ─────────────────────────
  output$ui_status_card <- renderUI({
    py_ok <- rv$py_ok
    n_sess <- nrow(rv$sessions %||% data.frame())
    n_feat <- nrow(rv$features %||% data.frame())
    tagList(
      tags$table(class="table table-sm table-borderless",
        style="font-size:12px",
        tags$tbody(
          tags$tr(
            tags$td(tags$b("Python")),
            tags$td(if(py_ok) tags$span(style="color:#27ae60","✓ Ready")
                    else tags$span(style="color:#e74c3c","✗ Not initialized"))
          ),
          tags$tr(
            tags$td(tags$b("Sessions")),
            tags$td(n_sess)
          ),
          tags$tr(
            tags$td(tags$b("segments")),
            tags$td(n_feat)
          )
        )
      )
    )
  })

  # Clear cache on demand
  shiny::observeEvent(input$btn_clear_cache, {
    tryCatch(cache_clear(), error = function(e) NULL)
    shiny::showNotification("✓ Cache emptied.", type = "message")
  })

  # ── Status de TextGrids subidos + emcouplemiento con audios ──
  output$ui_tg_status <- shiny::renderUI({
    tgf <- input$files_textgrids
    if (is.null(tgf) || nrow(tgf) == 0)
      return(tags$p(style="color:#888;font-size:12px",
        "No TextGrid loaded. If you upload one with the same base name ",
        "that an audio recording will be used as a reference point instead of diary entries."))
    audios <- input$files_sessions
    stems_tg <- tools::file_path_sans_ext(tgf$name)
    if (is.null(audios) || nrow(audios) == 0) {
      items <- sprintf("<li>%s (no corresponding audio yet)</li>",
                       htmltools::htmlEscape(tgf$name))
    } else {
      stems_aud <- tools::file_path_sans_ext(audios$name)
      items <- vapply(seq_len(nrow(tgf)), function(i) {
        match <- tolower(stems_tg[i]) %in% tolower(stems_aud)
        sprintf("<li>%s %s</li>",
          htmltools::htmlEscape(tgf$name[i]),
          if (match) "<span style='color:#27ae60'>✓ linked</span>"
          else       "<span style='color:#c0392b'>✗ without audio with that name</span>")
      }, character(1))
    }
    tags$div(
      tags$p(style="margin:6px 0;font-size:12px",
             sprintf("%d TextGrid(s) uploaded:", nrow(tgf))),
      HTML(sprintf("<ul style='font-size:12px;padding-left:18px'>%s</ul>",
                    paste(items, collapse=""))))
  })

  # ── Descarga ZIP de TextGrids ──
  output$dl_textgrids_zip <- shiny::downloadHandler(
    filename = function()
      paste0("acustera_textgrids_", format(Sys.time(), "%Y%m%d_%H%M"), ".zip"),
    content  = function(f) {
      tg_dir <- file.path(getwd(), "textgrids")
      files  <- if (dir.exists(tg_dir))
        list.files(tg_dir, pattern = "\\.TextGrid$", full.names = TRUE)
        else character(0)
      if (length(files) == 0) {
        writeLines("There are no TextGrids yet. Process at least a couple.",
                   file.path(tempdir(), "README.txt"))
        utils::zip(f, file.path(tempdir(), "README.txt"), flags = "-j")
        return(invisible())
      }
      utils::zip(f, files, flags = "-j")
    }, contentType = "application/zip")

  # ── Modo B/N para plots ──
  shiny::observe({
    options(acustera.bw = isTRUE(input$bw_mode))
  })

  # ── Pause threshold for re-segmentation (only applies if transcribe=TRUE) ──
  shiny::observe({
    thr <- as.numeric(input$pause_split_thr %||% 0.3)
    if (!is.finite(thr)) thr <- 0.3
    options(acustera.pause_split_s = thr)
  })

  # ── Extended modules (POS, examples v2, AI, report v2) ──
  tryCatch(attach_new_server(input, output, session, rv),
           error = function(e) message("attach_new_server: ", conditionMessage(e)))
}

# ══════════════════════════════════════════════════════════════
# Create www/audios/ if it doesn't exist
# Shiny serves www/ automatically as static root
# → los audios en www/audios/couple_therapyN.wav
#   son accesibles como URL: audios/couple_therapyN.wav
www_dir   <- file.path(getwd(), "www")
audio_dir <- file.path(www_dir, "audios")
if (!dir.exists(www_dir))   dir.create(www_dir,   recursive=TRUE)
if (!dir.exists(audio_dir)) dir.create(audio_dir, recursive=TRUE)

tryCatch(.enable_lmm_cache(), error = function(e) invisible(NULL))

shinyApp(ui, server)