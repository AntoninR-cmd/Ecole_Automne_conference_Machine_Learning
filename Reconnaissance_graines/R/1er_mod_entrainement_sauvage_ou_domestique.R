# Le code de cette section concerne le premier niveau de notre hypothèse.
# Dans cette section on cherche à évaluer les performances de notre modèle entrainé uniquement sur une catégorie de graine (les graines sauvages ou les graines domestiquées).
# L'objectif est de savoir si un modèle est plus performant pour classer toutes les catégories de graines lorsque la base d'entrainement ne contient soit que des graines "sauvages" soit que des graines "domestiquées".

# chemin d'accès pour toutes les données
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db  <- "rda/db"
dossier_res <- "rda/results/especes_dom_sauvages"

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")

col_image <- "VD_deep"   # VD_deep concerne uniquement les graines prises en photo en vue dorsale (de face).

# Dans le dossier rda, la variable [...] identifie les graines sauvages par un "w" et les domestiques par un "c" 
statut_domestique <- "c"
statut_sauvage    <- "w"

# Ici on défini les effectifs des graines par espèce et par statut (sauvage / domestique)
# On concerve la même proportion de sauvage que de domestique dans chaque modèle
N_dom_model   <- 530
N_dom_ext     <- 100
N_wild_model  <- 530
N_wild_ext    <- 100

prop_train  <- 4/5       # Proportion des données utilisée pour la base d'entrainement
epochs      <- 100
seed        <- 2329
seed_wild   <- seed + 1  # tirage des sauvages independant de celui des domestiques
batch_size  <- 64
input_shape <- c(120, 90, 3)   # (hauteur, largeur, canaux : ici nos données sont en couleur le 3 correspond aux valeur de RGB)

# Chargement des packages :
library(tidyverse)
library(keras)
py_require_legacy_keras()
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")
begin_time <- Sys.time()


# Création d'une fonction qui permet de charger les taxons domestiques et sauvages :

# Renvoie un tableau (taxon, statut, path) avec un chemin d'image valide pour chaque graine domestique ou sauvage.

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

# Ici on filtre les données en fonction du statut des graines : c pour domestiques et w pour sauvages.
  cat("Valeurs de 'status' :\n"); print(table(db$status))
  for (s in c(statut_domestique, statut_sauvage)) {
    if (!(s %in% as.character(db$status))) {
      stop("Aucune ligne avec status == '", s, "' pour ", taxon,
           " (valeurs presentes : ",
           paste(unique(as.character(db$status)), collapse = ", "), ")")
    }
  }
  db <- db[as.character(db$status) %in% c(statut_domestique, statut_sauvage), ]
  
# Chemin d'accès des images :
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


# Réunion des taxons :
tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]]))
)
cat("\nGraines disponibles (taxon x statut) :\n")
print(table(tous_les_taxons$taxon, tous_les_taxons$statut))


# Réalisation du tiration et du partitionnement entre les bases train et test :

# Graines domestiquées :
N_train_d <- round(N_dom_model * prop_train)
N_test_d  <- N_dom_model - N_train_d
labs_dom  <- c(rep("train",    N_train_d),
               rep("test",     N_test_d),
               rep("external", N_dom_ext))

set.seed(seed)
shuffled_dom <- sample(labs_dom)

dom <- tous_les_taxons %>% filter(statut == statut_domestique)
if (any(table(dom$taxon) < length(labs_dom))) {
  stop("Pas assez de graines domestiques dans un taxon (besoin de ",
       length(labs_dom), ").")
}
df_dom <- dom %>%
  group_by(taxon) %>%
  sample_n(length(labs_dom)) %>%
  mutate(partition = shuffled_dom) %>%
  ungroup()

# Graines sauvages :
N_train_w <- round(N_wild_model * prop_train)
N_test_w  <- N_wild_model - N_train_w
labs_wild <- c(rep("train",    N_train_w),
               rep("test",     N_test_w),
               rep("external", N_wild_ext))

set.seed(seed_wild)
shuffled_wild <- sample(labs_wild)

wild <- tous_les_taxons %>% filter(statut == statut_sauvage)
if (any(table(wild$taxon) < length(labs_wild))) {
  print(table(wild$taxon))
  stop("Pas assez de graines sauvages dans un taxon (besoin de ",
       length(labs_wild), ").")
}
df_wild <- wild %>%
  group_by(taxon) %>%
  sample_n(length(labs_wild)) %>%
  mutate(partition = shuffled_wild) %>%
  ungroup()

# Construction d'un tableau contenant nos résultats avec en ligne la classe de la graine, c'est à dire l'espèce prédite par le modèle
# et le statut qui n'est utilisé que lors de l'évaluation.
lib_statut <- setNames(c("domestique", "sauvage"),
                       c(statut_domestique, statut_sauvage))

master_df <- bind_rows(df_dom, df_wild) %>%
  transmute(partition,
            class  = taxon,
            statut = unname(lib_statut[statut]),
            path)

cat("\nPartition x espece x statut :\n")
print(ftable(table(partition = master_df$partition,
                   espece    = master_df$class,
                   statut    = master_df$statut)))

dfs <- split(master_df, master_df$partition)


# Générateur d'image :
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


# Construction du modèle :
# Ici notre modèle est un réseau de neurones convolutif adoptant l'architecture VGG19.
# Ce code permet de structurer l'ensemble des 19 couches du reseau.
try(tensorflow::set_random_seed(seed), silent = TRUE)

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

summary(model)


# Entrainement du modèle :
callbacks_list <- list(
  callback_early_stopping(monitor = "accuracy", patience = 20),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 1e-7)
)

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

plot(history) # Création de graphique permettant de mesurer la loss et l'accuracy au cours des epochs.


# Evaluation des modèles sur les données d'origine (avec des graines sauvages et domestiquées simultanément)
n_ext   <- nrow(dfs$external)
steps_e <- ceiling(n_ext / batch_size)

ext_eval <- model %>% evaluate(flux(dfs$external, melanger = FALSE),
                               steps = steps_e, verbose = 0)
cat("\n[Perte et accuracy sur", n_ext, "images externes]\n"); print(ext_eval)

# prediction image par image :
ext_gen      <- flux(dfs$external, melanger = FALSE)
proba        <- predict(model, ext_gen, steps = steps_e, verbose = 0)  # n x 3
classe_pred  <- max.col(proba)
classe_vraie <- as.integer(ext_gen$classes) + 1

noms_classes <- names(sort(unlist(ext_gen$class_indices)))

# On retrouve le statut (domestique / sauvage) de chaque image evaluee,
# en associant chaque fichier lu par le generateur a sa ligne du tableau.
idx <- match(ext_gen$filenames, dfs$external$path)
if (anyNA(idx)) {
  warning("Association par nom de fichier impossible : ordre des lignes utilise.")
  idx <- seq_len(nrow(dfs$external))
}
statut_ext <- dfs$external$statut[idx]

if (!all(noms_classes[classe_vraie] == dfs$external$class[idx])) {
  stop("Incoherence entre les classes du generateur et celles du tableau.")
}

espece_vraie   <- factor(noms_classes[classe_vraie], levels = noms_classes)
espece_predite <- factor(noms_classes[classe_pred],  levels = noms_classes)
statut_ext     <- factor(statut_ext, levels = unname(lib_statut))

# Construction d'un tableau de contingence :
conf_classique <- table(vraie = espece_vraie, predite = espece_predite)
cat("\n[Tableau 1 - classique : espece vraie x espece predite]\n")
cat("(toutes les images externes, domestiques et sauvages confondues)\n")
print(conf_classique)
cat("Accuracy globale :", round(mean(espece_vraie == espece_predite), 3), "\n")

# Construction d'un deuxième tableau qui inclu les informations relatives au statut des graines
# En créant deux lignes pour chaque espèces, par exemple : Olea Sauvage / Olea Domestique
niveaux_lignes <- unlist(lapply(noms_classes,
                                function(e) paste0(e, " | ", lib_statut)))
ligne <- factor(paste0(espece_vraie, " | ", statut_ext), levels = niveaux_lignes)

conf_statut <- table(vraie_et_statut = ligne, predite = espece_predite)
cat("\n[Tableau 2 - espece vraie ET statut (lignes) x espece predite (colonnes)]\n")
print(conf_statut)

# taux de bonne classification par ligne (espece correcte, quel que soit le statut)
taux <- sapply(niveaux_lignes, function(l) {
  esp <- strsplit(l, " | ", fixed = TRUE)[[1]][1]
  conf_statut[l, esp] / sum(conf_statut[l, ])
})
cat("\nTaux de bonne classification par espece et statut :\n")
print(round(taux, 3))

cat("\nAccuracy selon le statut :\n")
print(round(tapply(espece_vraie == espece_predite, statut_ext, mean), 3))


# Sauvegarde des résultats
dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
nom_base <- sprintf("Test3_3especes_dom_sauv_Nd%d_Nw%d_ep%d_seed%d",
                    N_dom_model, N_wild_model, epochs, seed)

resultats <- list(
  master_df        = master_df,
  historique       = as.data.frame(history$metrics),
  evaluation_ext   = ext_eval,
  conf_classique   = conf_classique,
  conf_statut      = conf_statut,
  taux_par_ligne   = taux,
  probas_ext       = proba,
  statut_ext       = statut_ext,
  codage_classes   = unlist(train_generator$class_indices),
  config = list(taxons = taxons, N_dom_model = N_dom_model,
                N_dom_ext = N_dom_ext, N_wild_model = N_wild_model,
                N_wild_ext = N_wild_ext, epochs = epochs, seed = seed,
                seed_wild = seed_wild, col_image = col_image),
  duree = Sys.time() - begin_time
)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))

tryCatch(
  save_model_hdf5(model, file.path(dossier_res, paste0(nom_base, ".h5"))),
  error = function(e) message("Modele non sauvegarde : ", conditionMessage(e))
)

cat("\nResultats ecrits dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")