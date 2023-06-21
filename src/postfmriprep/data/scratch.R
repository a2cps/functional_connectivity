library(tidyverse)

Networks_RH_ContB_PFCld_1 = c(41, 33, 37)
Networks_RH_ContB_PFCld_2 = c(42, 14, 49)
Networks_RH_ContB_PFCld_3 = c(23, 24, 53)


read_tsv("power2011_atlas.tsv") |>
  mutate(
    a = pmap_dbl(list(x,y,z), function(x, y, z) sqrt(sum((c(x, y, z) - Networks_RH_ContB_PFCld_1)^2))),
    b = pmap_dbl(list(x,y,z), function(x, y, z) sqrt(sum((c(x, y, z) - Networks_RH_ContB_PFCld_2)^2))),
    c = pmap_dbl(list(x,y,z), function(x, y, z) sqrt(sum((c(x, y, z) - Networks_RH_ContB_PFCld_3)^2))),
    m = pmap_dbl(list(a,b,c), min)) |>
  slice_min(m, n=1)


