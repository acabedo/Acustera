# Evaluation of AcusTera's sample mode against the full sessions
# (6 couples x 6 sessions, CrisperWhisper + DiariZen, roles validated).
# Sample mode = first m minutes of each session quartile (as in process_session()).
# Run from the evaluation/ folder: Rscript scripts/fig_sampling.R
# For each speaker x session, the sample-mode mean is compared with the
# full-session mean of the same segments. Only numeric columns are read:
# the transcripts are never loaded.
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2); library(readr)})

base <- "data"
num_cols <- cols_only(couple = "c", therapy_num = "i", speaker = "c", seg_start = "d",
                      seg_end = "d", duration = "d", f0_mean = "d", intensity_mean = "d",
                      n_words = "d", role_flag = "c")
f <- read_csv(file.path(base, "segments_acoustic.csv"), col_types = num_cols) |>
  filter(role_flag == "ok")
sdur <- read_csv(file.path(base, "session_durations.csv"),
                 col_types = cols_only(couple = "c", therapy_num = "i", dur_s = "d")) |>
  rename(dur = dur_s)

f <- f |> left_join(sdur, by = c("couple", "therapy_num")) |>
  group_by(couple, speaker) |> mutate(f0_st = 12 * log2(f0_mean / median(f0_mean, na.rm = TRUE))) |>
  group_by(couple, therapy_num) |> mutate(int_rel = intensity_mean - median(intensity_mean, na.rm = TRUE)) |>
  ungroup() |> mutate(tempo = n_words / duration,
                      q0 = dur * (pmin(floor(4 * seg_start / dur), 3)) / 4)

summ <- function(d) d |> group_by(couple, therapy_num, speaker) |>
  summarise(f0_st = mean(f0_st, na.rm = TRUE), int_rel = mean(int_rel, na.rm = TRUE),
            tempo = mean(tempo, na.rm = TRUE), t = sum(duration), n = n(), .groups = "drop") |>
  group_by(couple, therapy_num) |> mutate(share = 100 * t / sum(t)) |> ungroup()

full <- summ(f)
mins <- c(1, 2, 3, 4, 5, 6, 8, 10, 12)
ev <- bind_rows(lapply(mins, function(m) {
  s <- summ(filter(f, seg_start >= q0, seg_end <= q0 + 60 * m))
  inner_join(s, full, by = c("couple", "therapy_num", "speaker"), suffix = c("_s", "_f")) |>
    mutate(m = m)
}))

long <- ev |> transmute(m, couple, therapy_num, speaker, n_s,
  `F0 level (st)`         = abs(f0_st_s - f0_st_f),
  `Intensity (dB)`        = abs(int_rel_s - int_rel_f),
  `Tempo (words/s)`       = abs(tempo_s - tempo_f),
  `Share of speech (pp)`  = abs(share_s - share_f)) |>
  pivot_longer(-(m:n_s), names_to = "measure", values_to = "err")

tab <- long |> group_by(measure, m) |>
  summarise(median = median(err, na.rm = TRUE), q25 = quantile(err, .25, na.rm = TRUE),
            q75 = quantile(err, .75, na.rm = TRUE), p90 = quantile(err, .9, na.rm = TRUE),
            .groups = "drop")

# Does the sample keep each speaker's ordering of the six sessions?
rank_keep <- ev |> group_by(m, couple, speaker) |>
  summarise(F0 = cor(f0_st_s, f0_st_f, method = "spearman"),
            Intensity = cor(int_rel_s, int_rel_f, method = "spearman"),
            Tempo = cor(tempo_s, tempo_f, method = "spearman"),
            Share = cor(share_s, share_f, method = "spearman"), .groups = "drop") |>
  group_by(m) |> summarise(across(F0:Share, \(x) median(x, na.rm = TRUE)), .groups = "drop")

# Between-session spread of the full-session means (scale of the effects to detect)
spread <- full |> group_by(couple, speaker) |>
  summarise(F0 = sd(f0_st), Intensity = sd(int_rel), Tempo = sd(tempo), Share = sd(share),
            .groups = "drop") |> summarise(across(F0:Share, median))

dir.create("results", showWarnings = FALSE)
write_csv(tab, "results/sampling_error.csv")
write_csv(rank_keep, "results/sampling_rank_correlation.csv")
write_csv(spread, "results/between_session_sd.csv")
cat("segments per speaker-session in sample (median) by m:\n")
print(ev |> group_by(m) |> summarise(n_s = median(n_s)))
print(tab |> filter(m %in% c(3, 6, 12)), n = 50)
print(rank_keep); print(spread)

ink <- "#2B4C7E"
spread_l <- tibble(measure = c("F0 level (st)", "Intensity (dB)", "Tempo (words/s)", "Share of speech (pp)"),
                   y = unlist(spread[1, c("F0", "Intensity", "Tempo", "Share")]))
p <- ggplot(tab, aes(m, median)) +
  geom_vline(xintercept = 3, colour = "grey70", linewidth = .4, linetype = "dashed") +
  geom_hline(data = spread_l, aes(yintercept = y), colour = "grey45", linewidth = .4, linetype = "dotted") +
  geom_ribbon(aes(ymin = q25, ymax = q75), fill = ink, alpha = .18) +
  geom_line(colour = ink, linewidth = .7) + geom_point(colour = ink, size = 1.4) +
  facet_wrap(~measure, nrow = 1, scales = "free_y") +
  scale_x_continuous(breaks = c(1, 3, 6, 10), name = "Minutes sampled per quartile") +
  scale_y_continuous(limits = c(0, NA), name = "Absolute error vs. full session") +
  theme_minimal(base_size = 8, base_family = "Times") +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "grey92"),
        strip.text = element_text(face = "bold", hjust = 0))
ggsave("figures/fig_sampling.pdf", p, width = 6.3, height = 1.9, device = cairo_pdf)
cat("figure written\n")
