library(tidyverse)
folder <- "../analytical_ecdfs/"
lf <- list.files(folder, pattern = ".csv")

analytical_cdf <- map_dfr(lf, function(.lf){
  read_csv(file.path(folder, .lf)) %>% 
    mutate(N = str_match(.lf, "_N([0-9]+)")[,2] %>% parse_integer())
}) %>%
  rename(x = r, y = ecdf) %>%
  arrange(N, x)


write_csv(analytical_cdf, "analytical_cdf_tbl.csv")
