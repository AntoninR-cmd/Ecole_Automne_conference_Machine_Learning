# Dans ce script on aborde le niveau deux de notre analyse, il ajoute les images de graines archéobotaniques aux données
# à prédire. L'objectif est de savoir si un reseau de neurones convolutif pourrait permettre de retrouver l'espece à laquelle
# appartient des graines issues de fouilles archéologiques. On dispose d'images de graines archéologiques appartenant à nos 
# trois espèces de base et on cherche à retrouver leur espèces à partir d'un CNN entrainé uniquement sur des graines contemporaines.

# chemin d'accès pour toutes les données
racine            <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_resultats <- "rda/results/especes_dom_sauvages"
dossier_rda_ajout <- "rda/db_ajout"
dossier_sortie    <- "rda/results/evaluations"

modele_choisi <- NULL

# Nombre d'images domestiques/sauvages tirees par groupe (espece x statut)
# pour le test. 
# La veleur NULL signifie qu'on prend toutes les images "external" disponibles.
# En mettant un entier comme 30 par exemple, on équilibre les effectifs avec les graines archéobotaniques
n_par_groupe_dom_sauv <- NULL

# Fichier .rda des archeo. Contient les 3 especes (olea/phoenix/vitis, statut = "archeo") et, si le dossier AUTRES a été traité, une classe
# "autre" (autres especes modernes, statut = "autre_espece").

rda_archeo <- file.path(dossier_rda_ajout, "archeo_complet_db.rda")
seed_eval  <- 2329 # On initialise la graine pour ne plus être sujet aux variations des évaluations dues à l'aléatoire.
batch_size <- 32


# Chargement des packages :
library(tidyverse)
library(keras)
py_require_legacy_keras()
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")
begin_time <- Sys.time()


# Choix du modèle :
fichiers_h5 <- sort(list.files(dossier_resultats, pattern = "\\.h5$",
                               recursive = TRUE, full.names = TRUE))
if (length(fichiers_h5) == 0) stop("Aucun fichier .h5 dans ", dossier_resultats)
noms_courts <- substring(fichiers_h5, nchar(dossier_resultats) + 2)

if (is.null(modele_choisi)) {
  if (!interactive()) {
    cat("Modeles disponibles :\n"); print(noms_courts)
    stop("Renseigner 'modele_choisi' (numero ou morceau de nom).")
  }
  rep <- menu(noms_courts, title = "Quel modele evaluer ?")
  if (rep == 0) stop("Aucun modele choisi.")
  fichier_h5 <- fichiers_h5[rep]
} else if (is.numeric(modele_choisi)) {
  fichier_h5 <- fichiers_h5[modele_choisi]
  if (is.na(fichier_h5)) stop("Numero hors liste.")
} else {
  cand <- fichiers_h5[grepl(modele_choisi, fichiers_h5, fixed = TRUE)]
  if (length(cand) != 1) {
    cat("Modeles disponibles :\n"); print(noms_courts)
    stop(length(cand), " modele(s) correspondent a '", modele_choisi, "'.")
  }
  fichier_h5 <- cand
}
cat("\nModele choisi :", fichier_h5, "\n")


# Le dossier rda fait le lien entre les images et leur identification, grâce à ce fichier, on peut savoir si la graine
# représentée sur l'image est sauvage ou non, ou si elle est prise de face ou sur la tranche...
trouver_rda <- function(fichier_h5) {
  rda_meme_nom <- sub("\\.h5$", ".rda", fichier_h5)
  if (file.exists(rda_meme_nom)) return(rda_meme_nom)
  rdas    <- list.files(dirname(fichier_h5), pattern = "\\.rda$", full.names = TRUE)
  suffixe <- regmatches(basename(fichier_h5),
                        regexpr("ep[0-9]+_seed[0-9]+", basename(fichier_h5)))
  if (length(suffixe) == 0 || length(rdas) == 0)
    stop("Impossible de trouver le .rda associe a ", basename(fichier_h5))
  racine_h5 <- sub("_.*", "", basename(fichier_h5))
  cand <- rdas[startsWith(basename(rdas), paste0(racine_h5, "_")) &
               grepl(suffixe, basename(rdas), fixed = TRUE)]
  if (length(cand) == 0)
    cand <- rdas[grepl(suffixe, basename(rdas), fixed = TRUE)]
  if (length(cand) == 0)
    stop("Aucun .rda associe a '", suffixe, "'.")
  cand[which.max(file.mtime(cand))]
}

charger_rda <- function(f, quoi = "master_df") {
  env_tmp <- new.env()
  noms    <- load(f, envir = env_tmp)
  obj     <- get(noms[1], envir = env_tmp)
  if (!is.list(obj) || is.null(obj[[quoi]]))
    stop("Pas de '", quoi, "' dans ", f)
  obj
}

rda_modele <- trouver_rda(fichier_h5)
cat(".rda du modele :", rda_modele, "\n")

res_modele    <- charger_rda(rda_modele)
master_modele <- as_tibble(res_modele$master_df)


# Chargement du modèle :

model <- load_model_hdf5(fichier_h5, compile = FALSE)

dim_entree <- unlist(model$input_shape)
dim_entree <- dim_entree[!is.na(dim_entree) & dim_entree > 0]
n_sorties  <- tail(unlist(model$output_shape), 1)
taille_img <- dim_entree[1:2]
if (n_sorties == 1)
  stop("Ce script est prevu pour un modele multi-classes (Test3 = 3 sorties).")

classes_modele <- sort(unique(master_modele$class))
if (length(classes_modele) != n_sorties)
  stop("Le modele a ", n_sorties, " sorties mais le .rda contient ",
       length(classes_modele), " classes.")

cat("\nClasses du modele :", paste(classes_modele, collapse = " / "), "\n")
cat("Taille image      :", taille_img[1], "x", taille_img[2], "\n")


# On récupère toutes les graines archéobotaniques en s'appuyant sur le fichier rda_archeo lorsque le statut des graines est "archeo"
if (!file.exists(rda_archeo))
  stop("Fichier archeo introuvable : ", rda_archeo)

res_archeo    <- charger_rda(rda_archeo)
master_archeo <- as_tibble(res_archeo$master_df)

cat("\nArcheo (class x statut) :\n")
print(table(master_archeo$class, master_archeo$statut, useNA = "ifany"))

# especes connues du modele (archeo) vs autres especes

archeo_3 <- master_archeo %>%
  filter(class %in% classes_modele) %>%
  mutate(statut = "archeo") %>%
  select(path, class, statut)

archeo_o <- master_archeo %>%
  filter(!(class %in% classes_modele)) %>%
  select(path, class, statut)

if (nrow(archeo_3) == 0)
  stop("Aucune image archeo des 3 especes dans ", rda_archeo)

cat("\nArcheo retenus (3 especes) : ", nrow(archeo_3), " images\n")
print(table(archeo_3$class))
if (nrow(archeo_o) > 0) {
  cat("Images 'autre' (bonus OOD) : ", nrow(archeo_o), " images\n")
  print(table(archeo_o$class))
}


# Construction de la base de test :

# Ici on exclue les graines utilisées lors de l'entrainement.
vues <- master_modele$path[master_modele$partition %in% c("train", "test")]
cat("\nImages vues a l'entrainement (a exclure) :", length(vues), "\n")

# Base entrainée sur les graines sauvages et également sur les graines domestiques :
pool_id <- master_modele %>%
  filter(partition == "external") %>%
  filter(!(path %in% vues)) %>%
  select(path, class, statut) %>%
  mutate(source = "external")

cat("\nImages ID (external) disponibles :\n")
print(table(pool_id$class, pool_id$statut))
if (!all(classes_modele %in% unique(pool_id$class)))
  warning("Certaines classes du modele sont absentes du jeu ID external.")


if (any(archeo_3$path %in% vues)) {
  warning(sum(archeo_3$path %in% vues),
          " images archeo apparaissent dans les donnees vues : exclues.")
  archeo_3 <- archeo_3 %>% filter(!(path %in% vues))
}
if (nrow(archeo_o) > 0 && any(archeo_o$path %in% vues)) {
  warning(sum(archeo_o$path %in% vues),
          " images 'autre' apparaissent dans les donnees vues : exclues.")
  archeo_o <- archeo_o %>% filter(!(path %in% vues))
}

if (!is.null(n_par_groupe_dom_sauv)) {
  set.seed(seed_eval)
  pool_id <- pool_id %>%
    group_by(class, statut) %>%
    slice_sample(prop = 1) %>%
    slice_head(n = n_par_groupe_dom_sauv) %>%
    ungroup()
}

# Vérification de la présence des images dans la base :

existe_toutes <- file.exists(c(pool_id$path, archeo_3$path, archeo_o$path))
if (any(!existe_toutes)) {
  warning(sum(!existe_toutes), " image(s) de test introuvable(s) sur disque.")
  pool_id   <- pool_id  [file.exists(pool_id$path),   ]
  archeo_3  <- archeo_3 [file.exists(archeo_3$path),  ]
  if (nrow(archeo_o) > 0)
    archeo_o <- archeo_o[file.exists(archeo_o$path), ]
}

# Construction du tableau final :

test_df <- bind_rows(
  pool_id,
  archeo_3 %>% mutate(source = "archeo"),
  if (nrow(archeo_o) > 0)
    archeo_o %>% mutate(source = "autre") else NULL
) %>%
  mutate(vraie_complete = if_else(class %in% classes_modele,
                                   paste(class, statut, sep = " | "),
                                   "autre"))

cat("\nJeu de test final :\n")
print(table(test_df$vraie_complete, useNA = "ifany"))
cat("Total :", nrow(test_df), "images\n")


# Phase de prédiction des nouvelles graines par le modèle :

datagen <- image_data_generator(rescale = 1/255)

# lorsque que class_mode est NULL le generateur ne renvoie que les images. On evite les problemes lies 
# au fait que "class" contient 4 valeurs possibles alors que le modele n'a que 3 sorties.
gen <- flow_images_from_dataframe(
  as.data.frame(test_df), directory = "",
  x_col = "path", y_col = "class", generator = datagen,
  target_size = taille_img, class_mode = NULL,
  batch_size = batch_size, shuffle = FALSE)

proba <- predict(model, gen, steps = ceiling(nrow(test_df) / batch_size),
                 verbose = 0)

idx <- match(gen$filenames, test_df$path)
if (anyNA(idx)) {
  warning("Association par nom de fichier impossible : ordre des lignes utilise.")
  idx <- seq_len(nrow(test_df))
}
proba   <- proba[idx, , drop = FALSE]
test_df <- test_df[idx, ]

p_max        <- apply(proba, 1, max)
pred_species <- factor(classes_modele[max.col(proba, ties.method = "first")],
                       levels = classes_modele)


# Construction de la matrice de taille 9x3
# elle permet de visualiser les l'espèce prédite pour une image donnée ainsi que son statut archéologique ou non

niveaux_lignes <- unlist(lapply(classes_modele,
                                function(e) paste0(e, " | ",
                                                   c("domestique", "sauvage",
                                                     "archeo"))))
ligne_f <- factor(test_df$vraie_complete, levels = niveaux_lignes)

# images "autre" : on les met de cote pour la matrice 9x3
garde <- !is.na(ligne_f)

conf_9x3 <- table(vraie_et_statut = ligne_f[garde],
                  predite        = pred_species[garde])

cat("\n=====================================================\n")
cat("[Matrice principale 9 x 3]  (espece | statut) x espece predite\n")
cat("=====================================================\n")
print(conf_9x3)

# effectifs par ligne (pour interpreter les taux)
cat("\nEffectifs par ligne :\n")
print(setNames(as.integer(rowSums(conf_9x3)),
               rownames(conf_9x3)))

# taux de bonne classification par ligne (espece correcte)
taux_ligne <- sapply(niveaux_lignes, function(lbl) {
  esp     <- sub(" \\| .*", "", lbl)
  n_ligne <- sum(conf_9x3[lbl, ])
  if (n_ligne == 0) return(NA_real_)
  conf_9x3[lbl, esp] / n_ligne
})
cat("\nTaux de bonne classification par ligne (espece correcte) :\n")
print(round(taux_ligne, 3))

# accuracy par statut (toutes especes confondues)
correct_espece <- as.character(pred_species) == test_df$class
statut_9 <- factor(test_df$statut, levels = c("domestique", "sauvage", "archeo"))
cat("\nAccuracy par statut (espece correcte, toutes especes) :\n")
print(round(tapply(correct_espece[garde], statut_9[garde], mean), 3))

# accuracy par espece (tous statuts confondus)
espece_9 <- factor(test_df$class, levels = classes_modele)
cat("\nAccuracy par espece (espece correcte, tous statuts) :\n")
print(round(tapply(correct_espece[garde], espece_9[garde], mean), 3))

# Métriques globales (accuracy et loss) :

cat("\n[Metriques globales]\n")
cat("Accuracy espece (9 lignes)         :",
    round(mean(correct_espece[garde]), 3), "\n")

# logloss sur les 9 lignes (les "autre" n'ont pas de vraie classe 1..3)
if (any(garde)) {
  vrai_idx <- match(test_df$class[garde], classes_modele)
  p_vraie  <- proba[cbind(which(garde), vrai_idx)]
  logloss  <- -mean(log(pmax(p_vraie, 1e-15)))
  cat("Logloss (9 lignes seulement)       :", round(logloss, 3), "\n")
}


# Sauvegarde :
dir.create(dossier_sortie, recursive = TRUE, showWarnings = FALSE)
base <- sprintf("Test4_%s_n%d",
                sub("\\.h5$", "", basename(fichier_h5)), nrow(test_df))

predictions <- tibble(
  path           = test_df$path,
  source         = test_df$source,
  class_vraie    = test_df$class,
  statut         = test_df$statut,
  vraie_complete = test_df$vraie_complete,
  pred_species   = as.character(pred_species),
  correct_espece = ifelse(test_df$class %in% classes_modele,
                           as.character(pred_species) == test_df$class,
                           NA),
  p_max          = p_max)
for (i in seq_along(classes_modele))
  predictions[[paste0("proba_", classes_modele[i])]] <- proba[, i]

res <- list(
  modele          = fichier_h5,
  rda_modele      = rda_modele,
  rda_archeo      = rda_archeo,
  classes_modele  = classes_modele,
  niveaux_lignes  = niveaux_lignes,
  confusion_9x3   = conf_9x3,
  effectifs_ligne = rowSums(conf_9x3),
  taux_par_ligne  = taux_ligne,
  accuracy_statut = tapply(correct_espece[garde], statut_9[garde], mean),
  accuracy_espece = tapply(correct_espece[garde], espece_9[garde], mean),
  accuracy_global = mean(correct_espece[garde]),
  predictions     = predictions,
  config = list(n_par_groupe_dom_sauv = n_par_groupe_dom_sauv,
                seed_eval = seed_eval, batch_size = batch_size,
                n_vues_exclues = length(vues),
                date = Sys.time()))

save(res, file = file.path(dossier_sortie, paste0(base, ".rda")))
write.csv(predictions,
          file.path(dossier_sortie, paste0(base, "_predictions.csv")),
          row.names = FALSE)
cat("\nResultats ecrits dans :", normalizePath(dossier_sortie), "\n")
cat("Duree :", format(Sys.time() - begin_time), "\n")