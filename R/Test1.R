# =====================================================================
# Test1.R
# Reproduction minimale de la methode 1 (CNN VGG19) de
# Bonhomme et al. (2025) : version simplifiee de vgg19_this()
# (fichier vgg_and_eft_this.R de l'archive).
#
# Perimetre de ce test :
#   - jeu de donnees : Vitis, vue dorsale (colonne VD_deep)
#   - 50 images par classe pour le modele (40 train + 10 "test")
#     + 100 images par classe pour l'evaluation externe
#   - 10 epoques, une seule graine
#   - methode 2 (EFT + LDA) ignoree
#
# Les ecarts avec le script original sont signales par "DIFFERENCE".
# Les commentaires sont volontairement sans accents (encodage Windows).
# Conseil : executer le script bloc par bloc (Ctrl+Entree dans RStudio)
# en lisant ce qui s'affiche dans la console.
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES (tout ce qu'on peut vouloir changer est ici)
# ---------------------------------------------------------------------
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db  <- "rda/db"            # contient les .rda de l'article
dossier_img <- "DATA/VITIS"        # contient les photos
dossier_res <- "rda/results/vitis" # ou seront ecrits les resultats

col_image   <- "VD_deep"           # vue dorsale (VL_deep = vue laterale)

N_model     <- 50     # images PAR CLASSE pour train + "test" (validation)
N_external  <- 100    # images PAR CLASSE reservees a l'evaluation finale
prop_train  <- 4/5    # part de N_model utilisee pour l'entrainement
epochs      <- 10     # epoques maximum
seed        <- 2329   # premiere graine de l'article
batch_size  <- 16
input_shape <- c(120, 90, 3)   # (hauteur, largeur, canaux)

# ---------------------------------------------------------------------
# 1. PACKAGES ET DOSSIER DE TRAVAIL
# ---------------------------------------------------------------------
# Si R ne trouve pas Python/TensorFlow, decommenter et adapter :
# reticulate::use_condaenv("nom_de_ton_environnement", required = TRUE)
# reticulate::use_virtualenv("nom_de_ton_environnement", required = TRUE)

library(tidyverse)   # dplyr (sample_n, group_by...), stringr, etc.
library(keras)       # interface R vers Keras
# Declare a reticulate les paquets Python necessaires (TensorFlow, tf-keras,
# Pillow, h5py...). Au premier lancement, ils sont telecharges et installes
# automatiquement (plusieurs centaines de Mo : c'est long, une seule fois).
# A faire AVANT toute utilisation de Python : en cas d'erreur, redemarrer R
# (Session > Restart R) puis relancer le script depuis le debut.
py_require_legacy_keras()
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")
cat("Version du package keras :", as.character(packageVersion("keras")), "\n")
begin_time <- Sys.time()


# ---------------------------------------------------------------------
# 2. CHARGEMENT DU TABLEAU DE DONNEES (.rda)
# ---------------------------------------------------------------------
# Un .rda contient un tableau (tibble) : une ligne = une graine.
# Colonnes utilisees ici : "status" (la classe) et col_image (chemin
# de l'image). Les colonnes *_coo (contours) ne servent qu'a la
# methode 2 et ne sont pas utilisees.

fichiers_db <- list.files(dossier_db, pattern = "vitis",
                          full.names = TRUE, ignore.case = TRUE)
cat("Fichiers .rda trouves pour vitis :\n"); print(fichiers_db)
if (length(fichiers_db) == 0) stop("Aucun .rda 'vitis' dans ", dossier_db)

# l'article charge "vitis_db2.rda" : on le prefere s'il existe
if (any(basename(fichiers_db) == "vitis_db2.rda")) {
  fichier_db <- fichiers_db[basename(fichiers_db) == "vitis_db2.rda"][1]
} else {
  fichier_db <- fichiers_db[1]
}
cat("Fichier utilise :", fichier_db, "\n")

# on charge dans un environnement temporaire pour ne pas dependre
# du nom de l'objet contenu dans le .rda
env_tmp     <- new.env()
noms_objets <- load(fichier_db, envir = env_tmp)
cat("Objets contenus dans le .rda :", noms_objets, "\n")
vitis_db <- get(noms_objets[1], envir = env_tmp)

cat("\nColonnes du tableau :\n"); print(names(vitis_db))
cat("\nNombre de lignes :", nrow(vitis_db), "\n")
if (!all(c("status", col_image) %in% names(vitis_db))) {
  stop("Colonnes 'status' et/ou '", col_image, "' introuvables.")
}
cat("\nEffectifs par classe (status) :\n"); print(table(vitis_db$status))


# ---------------------------------------------------------------------
# 3. VERIFICATION DES CHEMINS D'IMAGES
# ---------------------------------------------------------------------
# Les generateurs Keras lisent directement les fichiers : les chemins
# du .rda doivent pointer vers des fichiers qui existent sur ton disque.

chemins <- as.character(vitis_db[[col_image]])
cat("\nExemples de chemins stockes dans le .rda :\n"); print(head(chemins, 3))

existe <- file.exists(chemins)
cat("Fichiers trouves tels quels :", sum(existe), "/", length(existe), "\n")

if (!all(existe)) {
  # Les chemins du .rda ne correspondent pas a ton arborescence.
  # On les retrouve par nom de fichier dans DATA/VITIS (sous-dossiers inclus).
  cat("Tentative de retrouver les images par nom de fichier dans",
      dossier_img, "...\n")
  tous_les_fichiers <- list.files(dossier_img, recursive = TRUE,
                                  full.names = TRUE)
  noms_disque <- basename(tous_les_fichiers)
  if (anyDuplicated(noms_disque) > 0) {
    stop("Des noms de fichiers sont dupliques dans ", dossier_img,
         " : la correspondance par nom n'est pas fiable.")
  }
  idx <- match(basename(chemins), noms_disque)
  if (any(is.na(idx))) {
    cat("Images introuvables (exemples) :\n")
    print(head(basename(chemins)[is.na(idx)]))
    stop(sum(is.na(idx)), " images du .rda introuvables dans ", dossier_img)
  }
  vitis_db[[col_image]] <- tous_les_fichiers[idx]
  cat("Chemins corriges. Exemples :\n"); print(head(vitis_db[[col_image]], 3))
}


# ---------------------------------------------------------------------
# 4. TIRAGE ET PARTITION (train / test / external)
# ---------------------------------------------------------------------
# Meme logique que l'original : pour chaque classe on tire au hasard
# N_total images (sans remise), puis on attribue a chacune une etiquette
# de partition ("train", "test" ou "external").
#   - "train"    : sert a ajuster les poids de la tete du reseau
#   - "test"     : sert de VALIDATION (val_loss, pilote le learning rate)
#   - "external" : evaluation finale

N_train <- round(N_model * prop_train)
N_test  <- N_model - N_train
N_total <- N_train + N_test + N_external

if (any(table(vitis_db$status) < N_total)) {
  print(table(vitis_db$status))
  stop("Pas assez d'images dans une classe (besoin de ", N_total, ").")
}

labs <- c(rep("train",    N_train),
          rep("test",     N_test),
          rep("external", N_external))

set.seed(seed)                       # reproductibilite du tirage
shuffled_labs <- sample(labs)        # on melange les etiquettes de partition

master_df <- vitis_db %>%
  group_by(status) %>%
  sample_n(N_total) %>%              # N_total images par classe
  mutate(partition = shuffled_labs) %>%
  ungroup() %>%
  transmute(partition,
            class = as.character(status),
            path  = as.character(.data[[col_image]]))

cat("\nPartitions (lignes) x classes (colonnes) :\n")
print(table(master_df$partition, master_df$class))

dfs <- split(master_df, master_df$partition)   # dfs$train, dfs$test, dfs$external


# ---------------------------------------------------------------------
# 5. GENERATEURS D'IMAGES
# ---------------------------------------------------------------------
# Un generateur lit les images par paquets (batchs) de 16, les
# redimensionne a 120 x 90 (hauteur x largeur), et divise les pixels
# par 255 (-> valeurs entre 0 et 1, canal par canal).

all_datagen <- image_data_generator(rescale = 1/255)

flux <- function(x, melanger = TRUE) {
  flow_images_from_dataframe(
    x,
    directory   = "",
    x_col       = "path",
    y_col       = "class",
    generator   = all_datagen,
    target_size = input_shape[1:2],
    class_mode  = "binary",
    batch_size  = batch_size,
    shuffle     = melanger,
    seed        = seed)
}

train_generator    <- flux(dfs$train)
test_generator     <- flux(dfs$test)
external_generator <- flux(dfs$external)

# Quelle classe est codee 0 et laquelle 1 ? (ordre alphabetique)
cat("\nCodage des classes (0/1) par Keras :\n")
print(train_generator$class_indices)


# ---------------------------------------------------------------------
# 6. MODELE : VGG19 (gele) + petite tete de classification
# ---------------------------------------------------------------------
# DIFFERENCE (ajout) : graine de TensorFlow pour que l'initialisation
# des poids de la tete soit reproductible. Absent du script original.
try(tensorflow::set_random_seed(seed), silent = TRUE)

# Base convolutive VGG19 pre-entrainee sur ImageNet, sans ses couches
# denses finales (include_top = FALSE)
base_model <- application_vgg19(weights     = "imagenet",
                                include_top = FALSE,
                                input_shape = input_shape,
                                classes     = 2)

# Tete : moyenne spatiale -> couche cachee (256, ReLU) -> sortie sigmoide
predictions <- base_model$output %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 1,   activation = "sigmoid")

model <- keras_model(inputs = base_model$input, outputs = predictions)

# On gele la base : ses ~20 millions de poids ne seront pas modifies
freeze_weights(base_model)

model %>% compile(optimizer = optimizer_rmsprop(learning_rate = 0.01),
                  loss      = "binary_crossentropy",
                  metrics   = c("accuracy"))

cat("\nResume du modele (voir 'Trainable params' et 'Non-trainable params') :\n")
summary(model)


# ---------------------------------------------------------------------
# 7. ENTRAINEMENT
# ---------------------------------------------------------------------
callbacks_list <- list(
  # arret si l'accuracy d'ENTRAINEMENT ne progresse pas pendant 20 epoques
  callback_early_stopping(monitor = "accuracy", patience = 20),
  # learning rate divise par 10 si val_loss ne progresse pas pendant 10 epoques
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 1e-7)
)

# nombre de batchs de validation : au moins 1
min_val <- floor(N_test / batch_size)
if (min_val == 0) min_val <- 1

# DIFFERENCE : fit() au lieu de fit_generator() (deprecie dans les
# versions recentes de keras ; fit() accepte les generateurs).
# Le calcul est identique. steps_per_epoch est conserve comme dans
# l'original : floor(N_train/16) = 2 batchs par epoque ici.
history <- model %>% fit(
  train_generator,
  steps_per_epoch  = floor(N_train / batch_size),
  epochs           = epochs,
  validation_data  = test_generator,
  validation_steps = min_val,
  callbacks        = callbacks_list
)

plot(history)


# ---------------------------------------------------------------------
# 8. EVALUATION SUR LES DONNEES EXTERNES
# ---------------------------------------------------------------------
# (a) Comme l'original : evaluate sur floor(N_external/16) = 6 batchs
#     (soit 96 images tirees au hasard sur les 200 images externes).
ext_acc <- model %>% evaluate(external_generator,
                              steps   = floor(N_external / batch_size),
                              verbose = 0)
cat("\n[Evaluation 'comme l'original' : 6 batchs]\n"); print(ext_acc)

# (b) AJOUT : evaluation sur TOUTES les images externes (200),
#     sans melange, pour comparer avec (a).
ext_complet <- flux(dfs$external, melanger = FALSE)
n_ext       <- nrow(dfs$external)
proba       <- as.numeric(predict(model, ext_complet,
                                  steps   = ceiling(n_ext / batch_size),
                                  verbose = 0))
classe_pred  <- as.integer(proba > 0.5)
classe_vraie <- as.integer(ext_complet$classes)
acc_complete <- mean(classe_pred == classe_vraie)

noms_classes <- names(sort(unlist(ext_complet$class_indices)))
cat("\n[Evaluation sur les", n_ext, "images externes]\n")
cat("Accuracy :", round(acc_complete, 3), "\n")
cat("Matrice de confusion (lignes = vraie classe, colonnes = predite) :\n")
print(table(vraie   = noms_classes[classe_vraie + 1],
            predite = noms_classes[classe_pred  + 1]))


# ---------------------------------------------------------------------
# 9. SAUVEGARDE DES RESULTATS
# ---------------------------------------------------------------------
dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
nom_base <- sprintf("Test1_vitis_VD_N%d_ep%d_seed%d", N_model, epochs, seed)

resultats <- list(
  master_df     = master_df,                       # quelle image a servi a quoi
  historique    = as.data.frame(history$metrics),  # courbes d'entrainement
  ext_6_batchs  = ext_acc,
  ext_complet   = acc_complete,
  codage_classes = unlist(train_generator$class_indices),
  config = list(N_model = N_model, N_external = N_external,
                epochs = epochs, seed = seed, col_image = col_image,
                fichier_db = fichier_db),
  duree = Sys.time() - begin_time
)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))


# le modele est sauvegarde a part (on ne le met pas dans le .rda)
tryCatch(
  save_model_hdf5(model, file.path(dossier_res, paste0(nom_base, ".h5"))),
  error = function(e) message("Modele non sauvegarde : ", conditionMessage(e))
)

cat("\nResultats ecrits dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")
