# .........................................................
# Run all models for Bonhomme et al. paper GMM vs CNN
#
# Author(s): 
#   - Vincent Bonhomme <bonhomme.vincent@gmail.com>
#  Date:
# - 2019 initial version
# - 2024 submission to PCI
# - 2025 resubmission to PCA
# .........................................................


# dependencies
library(tidyverse) # general data handling
library(Momocs)    # for everything morphometrics
library(keras)     # deep learning models

# Python environment
# this is machine-dependent, 
# change it with your own virtual environment
use_virtualenv("myenv")

# load tibbles
# each one of these four rdas contain data.frame(s) structured as follows:
# id          the name of the seed
# folder      the parent folder
# status      binary encoding of the classes we try to distinguish
# VL|VD_coo   a list column of xy matrices of the outline (for outline analyses/EFT)
# VL|VD_deep  the path towards the image used for CNN
load("rda/db/vitis_db2.rda")    # #772
load("rda/db/hordeum_db2.rda")  # 337
load("rda/db/phoenix_db2.rda")  # 562
load("rda/db/olea_db2.rda")     # 530

# load all domestic functions
# workshorses are stored there so that we keep this file as compact and 
# readable as possible.
source("R/vgg_and_eft_this.R")

# now we run. we copy paste a lot of code but we can launch each one separetely.
# Run 54 models (7, 9, 7, 4) x 2
# run 1/10
BEGIN <- Sys.time() # a cheap clock
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2329)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2329)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2329, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2329, epochs = 200)
END   <- Sys.time()
END - BEGIN         # time to run them all ~ 17.3 hours

# run 2/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2330)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2330)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2330, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2330, epochs = 200)

# run 3/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2331)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2331)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2331, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2331, epochs = 200)

# run 4/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2332)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2332)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2332, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2332, epochs = 200)

# run 5/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2333)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2333)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2333, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2333, epochs = 200)

# run 6/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2334)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2334)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2334, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2334, epochs = 200)

# run 7/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2335)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2335)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2335, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2335, epochs = 200)

# run 8/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2336)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2336)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2336, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2336, epochs = 200)

# run 9/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2337)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2337)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2337, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2337, epochs = 200)

# run 10/10
run_all(olea_db, "olea",       data_cols = c("VL_deep", "VD_deep"), seed_i = 2338)
run_all(vitis_db, "vitis",     data_cols = c("VL_deep", "VD_deep"), seed_i = 2338)
run_all(hordeum_db, "hordeum", data_cols = c("VL_deep", "VV_deep"), seed_i = 2338, epochs = 200)
run_all(phoenix_db, "phoenix", data_cols = c("VL_deep", "VD_deep"), seed_i = 2338, epochs = 200)

### end
