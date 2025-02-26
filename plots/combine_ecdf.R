library(tidyverse)
library(patchwork)

ecdf_tbl <- read_csv("../compiled_ecdf_data.csv")

# Add a simulation number for pivoting
ecdf_tbl <- group_by(ecdf_tbl, N, signal_type) 
ecdf_tbl <- mutate(ecdf_tbl, idx = row_number())
ecdf_tbl <- ungroup(ecdf_tbl)

# Pivot so that harmonic number is clear
ecdf_tbl <- ecdf_tbl %>%
  pivot_longer(
    -c(signal_type, N, idx),
    names_to = "harmonic_no",
    values_to = "Zk"
  )



ecdf_tbl <- ecdf_tbl %>%
  nest(data = c(idx, Zk)) %>%
  mutate(data = map2(data, N, function(.d, .N) {
    Zk <- pull(.d, Zk)
    efun <- ecdf(Zk)
    x <- c(seq(0, .N, length.out = 1000), seq(.N, 12, by = 1))
    y <- efun(x)
    return(tibble(x = x, y = y))
  })) %>%
  unnest(cols = data)


# Save this ecdf to one file for easier manipulation.
write_csv(ecdf_tbl, "ecdf_tbl.csv")



