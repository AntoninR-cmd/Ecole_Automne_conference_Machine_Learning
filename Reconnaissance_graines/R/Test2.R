# Ce script aborde la reconnaissance des 3 espèces (Olea, Phoenix, Vitis) à
# partir des seules graines domestiques, avec un VGG19 gelé et une tête de
# classification à 3 sorties.
#
# Par rapport à Test1.R, plusieurs choses changent :
#   - la tâche change : on ne distingue plus domestique / sauvage au sein
#     d'un taxon, on distingue 3 taxons entre eux
#   - 3 dossiers de photos et 3 .rda en entrée, réunis en un seul tableau
#   - on ne garde que les graines domestiques (filtre sur la colonne status)
#   - sortie du réseau : softmax à 3 classes (au lieu d'une sigmoïde)
#   - perte : categorical_crossentropy (au lieu de binary_crossentropy)
#   - évaluation : matrice de confusion 3 x 3
#
# Entraînement, validation ("test") et évaluation externe sont tous tirés
# parmi les images domestiques.
#
# Les écarts avec Test1.R sont signalés dans les commentaires par la mention
# "Différence avec Test1".


# Paramètres généraux
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db  <- "rda/db"
dossier_res <- "rda/results/especes_domestiques"

# Un taxon = un dossier de photos (les .rda sont cherchés par le nom du taxon)
taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")

col_image <- "VD_deep"   # vue dorsale pour les 3 taxons

# Valeur de la colonne "status" des .rda de base qui désigne une graine
# domestique : "c" (les graines sauvages ont une autre valeur).
statut_domestique <- "c"

N_model    <- 100     # images PAR CLASSE pour train + "test" (validation)
N_external <- 100     # images PAR CLASSE pour l'évaluation finale
prop_train <- 4/5
epochs     <- 200
seed       <- 2329
batch_size <- 32
input_shape <- c(120, 90, 3)   # (hauteur, largeur, canaux)


# Chargement des packages et préparation du dossier de travail
library(tidyverse)
library(keras)
py_require_legacy_keras()   # déclare TensorFlow / tf-keras à reticulate
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")
begin_time <- Sys.time()


# Fonction : charger un taxon et ne garder que les domestiques.
# Renvoie un tableau (taxon, statut, path) avec un chemin d'image valide
# pour chaque graine domestique.

charger_taxon <- function(taxon, dossier_img) {
  cat("\n=====", toupper(taxon), "=====\n")

  # Le .rda (on préfère <taxon>_db2.rda, comme dans l'article)
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

  # Filtre : graines domestiques
  cat("Valeurs de 'status' :\n"); print(table(db$status))
  if (!(statut_domestique %in% as.character(db$status))) {
    stop("Aucune ligne avec status == '", statut_domestique, "' pour ", taxon,
         " (valeurs presentes : ",
         paste(unique(as.character(db$status)), collapse = ", "), ")")
  }
  cat("Filtre : status ==", paste0("'", statut_domestique, "'"), "\n")
  db <- db[as.character(db$status) == statut_domestique, ]
  cat("Graines domestiques :", nrow(db), "\n")

  # Chemins d'images
  chemins <- as.character(db[[col_image]])
  cat("Exemples de chemins dans le .rda :\n"); print(head(chemins, 2))
  existe <- file.exists(chemins)
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
    cat("Chemins corriges. Exemples :\n"); print(head(chemins, 2))
  }

  tibble(taxon  = taxon,
         statut = as.character(db$status),
         path   = chemins)
}


# Réunion des 3 taxons
domestiques <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]]))
)
cat("\nGraines domestiques disponibles par taxon :\n")
print(table(domestiques$taxon))


# Tirage et partition (train / test / external).
# Même logique que Test1 : N_total images par classe (ici : par taxon),
# tirées sans remise, puis étiquetées "train", "test" (validation) ou
# "external" (évaluation finale). Classes équilibrées.

N_train <- round(N_model * prop_train)
N_test  <- N_model - N_train
N_total <- N_train + N_test + N_external

if (any(table(domestiques$taxon) < N_total)) {
  print(table(domestiques$taxon))
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


# Générateurs d'images.
# Différence avec Test1 : class_mode = "categorical" (étiquettes codées en
# vecteurs à 3 composantes, de type [0,1,0]) au lieu de "binary".

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

cat("\nCodage des classes (ordre alphabetique) :\n")
print(train_generator$class_indices)


# Modèle : VGG19 (gelé) + tête à 3 sorties softmax.
try(tensorflow::set_random_seed(seed), silent = TRUE)

base_model <- application_vgg19(weights     = "imagenet",
                                include_top = FALSE,
                                input_shape = input_shape)

# Différence avec Test1 : dernière couche = 3 neurones + softmax.
# Softmax transforme 3 scores en 3 probabilités positives de somme 1.
predictions <- base_model$output %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 3,   activation = "softmax")

model <- keras_model(inputs = base_model$input, outputs = predictions)
freeze_weights(base_model)

# Différence avec Test1 : categorical_crossentropy (perte adaptée à plusieurs
# classes avec étiquettes en vecteurs)
model %>% compile(optimizer = optimizer_rmsprop(learning_rate = 0.01),
                  loss      = "categorical_crossentropy",
                  metrics   = c("accuracy"))

summary(model)


# Entraînement du modèle
callbacks_list <- list(
  callback_early_stopping(monitor = "accuracy", patience = 20),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 1e-7)
)

# Différence avec Test1 : nombre de batchs calculé sur le NOMBRE RÉEL
# d'images (3 classes x N_train). Dans l'original (et Test1), floor(N_train/16)
# ne comptait qu'une classe, donc une "époque" ne parcourait qu'environ la
# moitié des images ; avec 3 classes ce serait environ un tiers.
steps_train <- ceiling(nrow(dfs$train) / batch_size)
steps_val   <- ceiling(nrow(dfs$test)  / batch_size)

history <- model %>% fit(
  train_generator,
  steps_per_epoch  = steps_train,
  epochs           = epochs,
  validation_data  = test_generator,
  validation_steps = steps_val,
  callbacks        = callbacks_list
)

plot(history)


# Évaluation sur les données externes (toutes les images).
# Différence avec Test1 : on utilise TOUTES les images externes (3 x 100 = 300),
# sans mélange, au lieu de floor(N_external/16) batchs.
n_ext   <- nrow(dfs$external)
steps_e <- ceiling(n_ext / batch_size)

ext_eval <- model %>% evaluate(flux(dfs$external, melanger = FALSE),
                               steps = steps_e, verbose = 0)
cat("\n[Perte et accuracy sur", n_ext, "images externes]\n"); print(ext_eval)

# Prédiction image par image (générateur neuf, sans mélange)
ext_gen      <- flux(dfs$external, melanger = FALSE)
proba        <- predict(model, ext_gen, steps = steps_e, verbose = 0)  # n x 3
classe_pred  <- max.col(proba)                      # indice de la plus grande proba
classe_vraie <- as.integer(ext_gen$classes) + 1     # Keras code à partir de 0

noms_classes <- names(sort(unlist(ext_gen$class_indices)))
conf <- table(vraie   = noms_classes[classe_vraie],
              predite = noms_classes[classe_pred])

cat("\nMatrice de confusion (lignes = vraie espece, colonnes = predite) :\n")
print(conf)
cat("\nAccuracy globale :", round(mean(classe_pred == classe_vraie), 3), "\n")
cat("Taux de bonne classification par espece :\n")
print(round(diag(conf) / rowSums(conf), 3))


# Sauvegarde des résultats
dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
nom_base <- sprintf("Test2_3especes_domestiques_N%d_ep%d_seed%d",
                    N_model, epochs, seed)

resultats <- list(
  master_df      = master_df,
  historique     = as.data.frame(history$metrics),
  evaluation_ext = ext_eval,
  confusion      = conf,
  probas_ext     = proba,
  codage_classes = unlist(train_generator$class_indices),
  config = list(taxons = taxons, N_model = N_model, N_external = N_external,
                epochs = epochs, seed = seed, col_image = col_image),
  duree = Sys.time() - begin_time
)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))

tryCatch(
  save_model_hdf5(model, file.path(dossier_res, paste0(nom_base, ".h5"))),
  error = function(e) message("Modele non sauvegarde : ", conditionMessage(e))
)

cat("\nResultats ecrits dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")