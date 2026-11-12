# Cosine similarity between the voice profiles (mean WeSpeaker embedding) of the
# 18 speakers attributed by the pipeline: 6 couples x {therapist, man, woman}.
# Couples and sites are anonymised (C1-C6; therapists T1-T3, one per site).
suppressPackageStartupMessages({library(dplyr); library(ggplot2); library(readr)})
d <- read_csv("data/speaker_similarity.csv", show_col_types = FALSE)
cp <- setNames(paste0("C", 1:6), paste0("C", 1:6))
th <- c(C1 = "T1", C2 = "T1", C3 = "T2", C4 = "T2", C5 = "T3", C6 = "T3")
lab <- function(c, r) ifelse(r == "therapist", paste0(th[c], " (", cp[c], ")"),
                             paste0(cp[c], " ", ifelse(r == "man", "man", "woman")))
d <- d |> mutate(a = lab(a_couple, a_role), b = lab(b_couple, b_role))
lev <- unique(d$a)
d <- d |> mutate(a = factor(a, lev), b = factor(b, rev(lev)))
p <- ggplot(d, aes(a, b, fill = cos)) + geom_tile(colour = "white", linewidth = .3) +
  scale_fill_gradient(low = "#F1F4F9", high = "#1E3A66", limits = c(min(0, min(d$cos)), 1),
                      name = "Cosine") +
  labs(x = NULL, y = NULL) + coord_fixed() +
  theme_minimal(base_size = 7, base_family = "Times") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5), panel.grid = element_blank(),
        legend.key.height = unit(.5, "cm"), legend.key.width = unit(.18, "cm"))
ggsave("figures/fig_similarity.pdf", p, width = 3.15, height = 2.9, device = cairo_pdf)
cat("ok\n")
