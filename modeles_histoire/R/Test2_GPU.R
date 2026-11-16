# =====================================================================
# Test2_GPU.R
# Meme experience que Test2.R (reconnaissance de 3 especes domestiques :
# Olea, Phoenix, Vitis ; VGG19 gele + softmax a 3 classes), avec :
#   - detection et utilisation du GPU s'il est disponible
#   - mesure des temps (lecture des images, chaque epoque, calcul pur)
#   - ajout d'une ligne de resultats dans un fichier benchmark.csv
#
# IMPORTANT : TensorFlow ne gere plus le GPU sous Windows natif (derniere
# version compatible : 2.10). Pour un GPU NVIDIA, le script doit etre lance
# depuis R installe dans WSL2 (Ubuntu sous Windows). Sous Windows natif, le
# script fonctionne mais tourne sur le processeur (utile comme reference).
#
# Pour comparer CPU et GPU : lancer le script deux fois, avec
#   mode_calcul <- "cpu"   puis   mode_calcul <- "gpu"
# (les deux lignes s'ajoutent dans benchmark.csv).
#
# Les ecarts avec Test2.R sont signales par "DIFFERENCE".
# Commentaires sans accents (encodage).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
# "auto" : GPU s'il y en a un, sinon processeur (message d'avertissement)
# "gpu"  : GPU obligatoire (arret si aucun GPU detecte)
# "cpu"  : GPU desactive (pour la mesure de reference)
mode_calcul <- "gpu"

# DIFFERENCE : racine selon le systeme (Windows natif ou WSL2)
if (.Platform$OS.type == "windows") {
  racine <- "C:/Users/fanos/Documents/Cours/Projet ML"
} else {
  # WSL2 : le disque C: est monte sous /mnt/c. La lecture des images est
  # plus lente a travers ce montage : pour un benchmark propre, copier le
  # projet dans le disque Linux (ex. ~/projet_ml) et adapter ce chemin.
  racine <- "/mnt/c/Users/fanos/Documents/Cours/Projet ML"
}

dossier_db  <- "rda/db"
dossier_res <- "rda/results/especes_domestiques_gpu"
dossier_bench <- "rda/results/benchmark"

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")

col_image <- "VD_deep"
statut_domestique <- "c"

N_model    <- 150      # images PAR CLASSE pour train + "test" (validation)
N_external <- 100     # images PAR CLASSE pour l'evaluation finale
prop_train <- 4/5
epochs     <- 100
seed       <- 2329
batch_size <- 16      # un GPU profite souvent de lots plus grands (ex. 64)
input_shape <- c(120, 90, 3)   # (hauteur, largeur, canaux)

t_script <- Sys.time()


# ---------------------------------------------------------------------
# 1. PACKAGES, PYTHON ET DETECTION DU GPU
# ---------------------------------------------------------------------
# DIFFERENCE : en mode "cpu", on masque les GPU AVANT de charger TensorFlow
if (mode_calcul == "cpu") Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

library(tidyverse)
library(keras)
py_require_legacy_keras()

# DIFFERENCE : sous Linux/WSL2, version de TensorFlow qui embarque les
# bibliotheques CUDA (pas d'installation manuelle de CUDA/cuDNN). Il faut
# seulement le pilote NVIDIA installe cote Windows.
if (.Platform$OS.type == "unix" && mode_calcul != "cpu") {
  reticulate::py_require("tensorflow[and-cuda]")
}
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")

gpus <- tf$config$list_physical_devices("GPU")
n_gpu <- length(gpus)
cat("\n--- Calcul ---\n")
cat("TensorFlow :", as.character(tf_version()), "\n")
cat("Systeme :", .Platform$OS.type, "\n")
cat("GPU detectes par TensorFlow :", n_gpu, "\n")

nom_gpu <- NA_character_
if (n_gpu > 0) {
  nom_gpu <- tryCatch(
    as.character(tf$config$experimental$get_device_details(gpus[[1]])$device_name),
    error = function(e) NA_character_)
  cat("GPU :", nom_gpu, "\n")
  # la memoire GPU est allouee a la demande plutot qu'en totalite
  for (g in gpus) try(tf$config$experimental$set_memory_growth(g, TRUE), silent = TRUE)
}

if (mode_calcul == "gpu" && n_gpu == 0) {
  stop("mode_calcul = 'gpu' mais aucun GPU detecte. Sous Windows natif, ",
       "TensorFlow (>= 2.11) n'utilise pas le GPU : lancer R dans WSL2. ",
       "Verifier aussi le pilote NVIDIA (commande nvidia-smi).")
}
if (mode_calcul == "auto" && n_gpu == 0) {
  cat("\n*** ATTENTION : aucun GPU detecte, le calcul se fera sur le processeur. ***\n")
}
mode_reel <- if (n_gpu > 0) "gpu" else "cpu"
cat("Mode reel :", mode_reel, "\n")


# ---------------------------------------------------------------------
# 2. FONCTION : CHARGER UN TAXON ET NE GARDER QUE LES DOMESTIQUES
# ---------------------------------------------------------------------
charger_taxon <- function(taxon, dossier_img) {
  cat("\n=====", toupper(taxon), "=====\n")

  fichiers <- list.files(dossier_db, pattern = taxon,
                         full.names = TRUE, ignore.case = TRUE)
  if (length(fichiers) == 0) stop("Aucun .rda '", taxon, "' dans ", dossier_db)
  prefere <- fichiers[tolower(basename(fichiers)) == paste0(taxon, "_db2.rda")]
  fichier <- if (length(prefere) > 0) prefere[1] else fichiers[1]
  cat("Fichier .rda :", fichier, "\n")

  env_tmp <- new.env()
  noms    <- load(fichier, envir = env_tmp)
  db      <- get(noms[1], envir = env_tmp)
  if (!all(c("status", col_image) %in% names(db))) {
    print(names(db))
    stop("Colonnes 'status' et/ou '", col_image, "' introuvables pour ", taxon)
  }

  cat("Valeurs de 'status' :\n"); print(table(db$status))
  if (!(statut_domestique %in% as.character(db$status))) {
    stop("Aucune ligne avec status == '", statut_domestique, "' pour ", taxon)
  }
  db <- db[as.character(db$status) == statut_domestique, ]
  cat("Graines domestiques :", nrow(db), "\n")

  chemins <- as.character(db[[col_image]])
  existe  <- file.exists(chemins)
  cat("Fichiers trouves tels quels :", sum(existe), "/", length(existe), "\n")

  if (!all(existe)) {
    cat("Recherche par nom de fichier dans", dossier_img, "...\n")
    tous  <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    noms_disque <- basename(tous)
    if (anyDuplicated(noms_disque) > 0) {
      stop("Noms de fichiers dupliques dans ", dossier_img)
    }
    idx <- match(basename(chemins), noms_disque)
    if (any(is.na(idx))) {
      print(head(basename(chemins)[is.na(idx)]))
      stop(sum(is.na(idx)), " images introuvables dans ", dossier_img)
    }
    chemins <- tous[idx]
  }

  tibble(taxon = taxon, statut = as.character(db$status), path = chemins)
}


# ---------------------------------------------------------------------
# 3. REUNION DES 3 TAXONS
# ---------------------------------------------------------------------
domestiques <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]]))
)
cat("\nGraines domestiques disponibles par taxon :\n")
print(table(domestiques$taxon))


# ---------------------------------------------------------------------
# 4. TIRAGE ET PARTITION (identique a Test2.R)
# ---------------------------------------------------------------------
N_train <- round(N_model * prop_train)

N_test  <- N_model - N_train
N_total <- N_train + N_test + N_external

if (any(table(domestiques$taxon) < N_total)) {
  stop("Pas assez de graines domestiques dans un taxon (besoin de ", N_total, ").")
}

labs <- c(rep("train",    N_train),
          rep("test",     N_test),
          rep("external", N_external))

set.seed(seed)
shuffled_labs <- sample(labs)

master_df <- domestiques %>%
  group_by(taxon) %>%
  sample_n(N_total) %>%
  mutate(partition = shuffled_labs) %>%
  ungroup() %>%
  transmute(partition, class = taxon, path, statut)

cat("\nPartitions (lignes) x taxons (colonnes) :\n")
print(table(master_df$partition, master_df$class))

dfs <- split(master_df, master_df$partition)


# ---------------------------------------------------------------------
# 5. GENERATEURS D'IMAGES
# ---------------------------------------------------------------------
all_datagen <- image_data_generator(rescale = 1/255)

flux <- function(x, melanger = TRUE) {
  flow_images_from_dataframe(
    x,
    directory   = "",
    x_col       = "path",
    y_col       = "class",
    generator   = all_datagen,
    target_size = input_shape[1:2],
    class_mode  = "categorical",
    batch_size  = batch_size,
    shuffle     = melanger,
    seed        = seed)
}

train_generator <- flux(dfs$train)
test_generator  <- flux(dfs$test)
cat("\nCodage des classes :\n"); print(train_generator$class_indices)

steps_train <- ceiling(nrow(dfs$train) / batch_size)
steps_val   <- ceiling(nrow(dfs$test)  / batch_size)


# ---------------------------------------------------------------------
# 5b. MESURE : TEMPS DE LECTURE DES IMAGES SEULES (sans modele)
# ---------------------------------------------------------------------
# Lire, decoder et redimensionner les JPEG se fait sur le PROCESSEUR, meme
# avec un GPU. Si ce temps est proche du temps d'une epoque, le GPU ne
# peut pas accelerer davantage : c'est le goulot d'etranglement.
gen_lecture <- flux(dfs$train)
t0 <- Sys.time()
for (i in seq_len(steps_train)) invisible(generator_next(gen_lecture))
t_lecture_epoque <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("\n[Mesure] lecture des images d'une epoque (",
    steps_train, "lots de", batch_size, ") :",
    round(t_lecture_epoque, 2), "s\n")


# ---------------------------------------------------------------------
# 6. MODELE : VGG19 (gele) + tete a 3 sorties softmax (comme Test2.R)
# ---------------------------------------------------------------------
try(tensorflow::set_random_seed(seed), silent = TRUE)

t0 <- Sys.time()
base_model <- application_vgg19(weights     = "imagenet",
                                include_top = FALSE,
                                input_shape = input_shape)

predictions <- base_model$output %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 3,   activation = "softmax")

model <- keras_model(inputs = base_model$input, outputs = predictions)
freeze_weights(base_model)

model %>% compile(optimizer = optimizer_rmsprop(learning_rate = 0.01),
                  loss      = "categorical_crossentropy",
                  metrics   = c("accuracy"))
t_construction <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("\n[Mesure] construction du modele (poids ImageNet inclus) :",
    round(t_construction, 1), "s\n")


# ---------------------------------------------------------------------
# 6b. MESURE : DEBIT DE CALCUL PUR (images deja en memoire)
# ---------------------------------------------------------------------
# On prend UN lot d'images deja lues et on mesure le temps de calcul du
# reseau seul (passage avant de VGG19 + tete), apres un tour d'echauffement.
lot <- generator_next(flux(dfs$train))
x   <- lot[[1]]
n_lot <- dim(x)[1]

invisible(model$predict_on_batch(x))            # echauffement (compilation, cuDNN)
n_rep <- 20
t0 <- Sys.time()
for (i in seq_len(n_rep)) invisible(model$predict_on_batch(x))
t_calcul <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
debit_calcul <- n_rep * n_lot / t_calcul
cat("[Mesure] debit de calcul pur :", round(debit_calcul, 1), "images/s\n")


# ---------------------------------------------------------------------
# 7. ENTRAINEMENT (avec chronometre par epoque)
# ---------------------------------------------------------------------
temps_epoque <- numeric(0)
debut_epoque <- Sys.time()

cb_chrono <- callback_lambda(
  on_epoch_begin = function(epoch, logs) debut_epoque <<- Sys.time(),
  on_epoch_end   = function(epoch, logs) {
    temps_epoque <<- c(temps_epoque,
                       as.numeric(difftime(Sys.time(), debut_epoque, units = "secs")))
  }
)

callbacks_list <- list(
  callback_early_stopping(monitor = "accuracy", patience = 20),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 1e-7),
  cb_chrono
)

t0 <- Sys.time()
history <- model %>% fit(
  train_generator,
  steps_per_epoch  = steps_train,
  epochs           = epochs,
  validation_data  = test_generator,
  validation_steps = steps_val,
  callbacks        = callbacks_list
)
t_fit <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

plot(history)

cat("\n[Mesure] temps par epoque (s) :\n"); print(round(temps_epoque, 2))
t_epoque1    <- temps_epoque[1]
t_epoque_moy <- if (length(temps_epoque) > 1) mean(temps_epoque[-1]) else NA_real_
cat("Epoque 1 (echauffement inclus) :", round(t_epoque1, 2), "s\n")
cat("Moyenne des epoques suivantes  :", round(t_epoque_moy, 2), "s\n")
cat("Entrainement total             :", round(t_fit, 1), "s\n")


# ---------------------------------------------------------------------
# 8. EVALUATION SUR LES DONNEES EXTERNES
# ---------------------------------------------------------------------
n_ext   <- nrow(dfs$external)
steps_e <- ceiling(n_ext / batch_size)

ext_gen <- flux(dfs$external, melanger = FALSE)
t0 <- Sys.time()
proba <- predict(model, ext_gen, steps = steps_e, verbose = 0)
t_predict <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

classe_pred  <- max.col(proba)
classe_vraie <- as.integer(ext_gen$classes) + 1
noms_classes <- names(sort(unlist(ext_gen$class_indices)))

conf <- table(vraie   = noms_classes[classe_vraie],
              predite = noms_classes[classe_pred])
acc_ext <- mean(classe_pred == classe_vraie)

cat("\nMatrice de confusion (lignes = vraie espece, colonnes = predite) :\n")
print(conf)
cat("Accuracy sur", n_ext, "images externes :", round(acc_ext, 3), "\n")
cat("Prediction de", n_ext, "images (lecture + calcul) :",
    round(t_predict, 2), "s\n")


# ---------------------------------------------------------------------
# 9. RESUME DU BENCHMARK ET SAUVEGARDE
# ---------------------------------------------------------------------
duree_totale <- as.numeric(difftime(Sys.time(), t_script, units = "secs"))

bench <- data.frame(
  date               = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  mode_reel          = mode_reel,
  systeme            = .Platform$OS.type,
  gpu                = nom_gpu,
  tensorflow         = as.character(tf_version()),
  N_model            = N_model,
  epochs             = epochs,
  batch_size         = batch_size,
  lecture_epoque_s   = round(t_lecture_epoque, 2),
  construction_s     = round(t_construction, 1),
  debit_calcul_img_s = round(debit_calcul, 1),
  epoque1_s          = round(t_epoque1, 2),
  epoque_moy_s       = round(t_epoque_moy, 2),
  fit_total_s        = round(t_fit, 1),
  predict_ext_s      = round(t_predict, 2),
  accuracy_ext       = round(acc_ext, 3),
  duree_totale_s     = round(duree_totale, 1)
)

cat("\n=============== RESUME DU BENCHMARK ===============\n")
print(t(bench))

dir.create(dossier_bench, recursive = TRUE, showWarnings = FALSE)
fichier_bench <- file.path(dossier_bench, "benchmark_Test2.csv")
write.table(bench, fichier_bench, sep = ";", row.names = FALSE,
            col.names = !file.exists(fichier_bench),
            append = file.exists(fichier_bench))
cat("Ligne ajoutee dans :", normalizePath(fichier_bench), "\n")

dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
nom_base <- sprintf("Test2_%s_N%d_ep%d_seed%d", mode_reel, N_model, epochs, seed)

resultats <- list(
  master_df    = master_df,
  historique   = as.data.frame(history$metrics),
  confusion    = conf,
  accuracy_ext = acc_ext,
  temps_epoque = temps_epoque,
  benchmark    = bench
)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))
cat("Resultats ecrits dans :", normalizePath(dossier_res), "\n")
