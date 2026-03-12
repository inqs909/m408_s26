## R Packages

# install.packages("torchdatasets")
# install.packages("feasts")

library(tidyverse)
library(torch)
library(luz) # high-level interface for torch

torch_manual_seed(909)
library(tsibble)
library(tsibbledata)
vic_elec
