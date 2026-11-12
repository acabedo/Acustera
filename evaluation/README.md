# Evaluation of AcusTera on complete therapy sessions

Materials for the evaluation reported in the accompanying article: 36 complete sessions (6 couples × sessions 1, 6, 9, 15, 19 and 20; 44.2 h) transcribed with CrisperWhisper 2.0 and diarized with DiariZen.

## Data (`data/`)

The data are anonymised. Couples are coded `C1`–`C6` and therapists `T1`–`T3` (one per site, two couples each). The files contain no audio and no transcripts.

| File | Content |
|---|---|
| `session_durations.csv` | Duration of each session (s), used to locate the quartiles. |
| `speaker_similarity.csv` | Cosine similarity between the mean speaker embeddings (WeSpeaker ResNet34) of the 18 attributed speakers. |

**Segment-level acoustic data are available on request.** The file `segments_acoustic.csv` contains one row per segment ≥ 2 s (17,410 rows) with the attributed role, timing, quartile, overlap, the AcusTera acoustic descriptors, the number of words and the role-attribution diagnostics. Because it derives from clinical recordings, it is not distributed publicly, and access is subject to the conditions of the participants' consent and of the project's data protection agreement. With this file in `data/`, `fig_sampling.R` reproduces the results in `results/`, which are included.

## Scripts (`scripts/`)

Run them from this folder; they need R with `dplyr`, `tidyr`, `ggplot2` and `readr`.

```sh
Rscript scripts/fig_sampling.R     # needs data/segments_acoustic.csv (on request) → results/*.csv, figures/fig_sampling.pdf
Rscript scripts/fig_similarity.R   # speaker similarity figure → figures/fig_similarity.pdf
```

`scripts/pipeline/` documents how the data were produced from the CrisperWhisper and DiariZen outputs. These scripts need the audio and the transcripts, which cannot be shared, so they are included for transparency and are not meant to be run on the published data.

- `01_utterances.py`: words → diarization turns; segments split at pauses ≥ 0.3 s; spurious speaker labels removed.
- `02_extract.py`: speaker embeddings and AcusTera acoustic descriptors for each segment ≥ 2 s.
- `03_roles.py`: cross-session role assignment per couple. Embeddings are centred by session and clustered with k-means (k = 3). Clusters are labelled with a voice-sex classifier, median F0 and lexical cues of the therapist's discourse. Ambiguous segments are flagged.

## Results (`results/`)

The `.csv` files written by `fig_sampling.R`:

- `sampling_error.csv`: median, quartiles and 90th percentile of the absolute error by measure and minutes per quartile.
- `sampling_rank_correlation.csv`: median Spearman correlation between the sample and full-session means of the six sessions.
- `between_session_sd.csv`: median between-session SD per speaker.
