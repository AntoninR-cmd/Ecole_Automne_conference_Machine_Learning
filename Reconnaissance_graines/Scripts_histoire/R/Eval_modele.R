# =====================================================================
# Eval_modele.R
# Evaluation d'un modele sauvegarde (fichier .h5 produit par Test1.R,
# Test2.R, Test3.R ou Test3_bis.R), sur des images de test "externes".
#
# Ce qui est choisi par l'utilisateur (section 0) :
#   - le modele a evaluer (menu, numero ou morceau de nom)
#   - le nombre d'images de test (par groupe, ou au total)
#   - eventuellement un autre jeu d'images de test (autre .rda)
#
# Ce qui est choisi AUTOMATIQUEMENT :
#   - le type de modele (1 sortie sigmoide = binaire ; 3 sorties
#     softmax = especes) et les classes, lus dans le modele et dans
#     le .rda de resultats associe
#   - les evaluations a produire :
#       binaire        : matrice de confusion 2x2, sensibilite,
#                        specificite, AUC, accuracy
#       multi-classes  : matrice de confusion classique, taux par classe
#       + si les images de test comprennent plusieurs statuts
#         (domestique / sauvage) : second tableau (classe vraie ET
#         statut x classe predite) et accuracy par statut
#
# Les images de test viennent de la partition "external" du .rda
# (jamais vues a l'entrainement ni a la validation). Toute image
# presente dans l'entrainement/validation du modele est exclue.
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine            <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_resultats <- "rda/results"              # recherche des .h5 (sous-dossiers inclus)
dossier_sortie    <- "rda/results/evaluations"  # ecriture des evaluations

# Modele a evaluer :
#   NULL           -> menu interactif (liste des .h5 trouves)
#   un nombre      -> numero dans la liste affichee
#   un texte       -> morceau du nom de fichier (doit etre unique), ex. "modele_sauvages"
modele_choisi <- NULL

# Nombre d'images de test.
#   Un "groupe" = une classe (modele binaire) ou une classe x un statut
#   (domestique / sauvage) quand le .rda contient plusieurs statuts.
#   n_test_total : nombre total d'images (repartie a egalite entre groupes).
#                  Si renseigne, il remplace n_test_par_groupe.
#   n_test_par_groupe : nombre d'images par groupe. NULL = toutes les
#                  images disponibles.
n_test_par_groupe <- 100
n_test_total      <- NULL

# Fichiers .rda (NULL = detection automatique)
fichier_rda_modele <- NULL   # .rda de resultats associe au modele (train/test/external)
fichier_rda_test   <- NULL   # .rda dont on prend les images de test (NULL = celui du modele)

seed_eval  <- 2329    # tirage des images de test
batch_size <- 16


# ---------------------------------------------------------------------
# 1. PACKAGES ET DOSSIER DE TRAVAIL
# ---------------------------------------------------------------------
library(tidyverse)
library(keras)
py_require_legacy_keras()
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")


# ---------------------------------------------------------------------
# 2. CHOIX DU MODELE
# ---------------------------------------------------------------------
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
  if (is.na(fichier_h5)) stop("Numero hors liste (1 a ", length(fichiers_h5), ").")
} else {
  cand <- fichiers_h5[grepl(modele_choisi, fichiers_h5, fixed = TRUE)]
  if (length(cand) != 1) {
    cat("Modeles disponibles :\n"); print(noms_courts)
    stop(length(cand), " modele(s) correspondent a '", modele_choisi,
         "' : preciser le nom.")
  }
  fichier_h5 <- cand
}
cat("\nModele choisi :", fichier_h5, "\n")


# ---------------------------------------------------------------------
# 3. .RDA ASSOCIES (resultats du modele, images de test)
# ---------------------------------------------------------------------
# Le .rda de resultats contient "resultats$master_df" : pour chaque image,
# sa partition (train / test / external), sa classe, son statut et son chemin.

trouver_rda <- function(fichier_h5) {
  rda_meme_nom <- sub("\\.h5$", ".rda", fichier_h5)
  if (file.exists(rda_meme_nom)) return(rda_meme_nom)
  rdas    <- list.files(dirname(fichier_h5), pattern = "\\.rda$", full.names = TRUE)
  suffixe <- regmatches(basename(fichier_h5),
                        regexpr("ep[0-9]+_seed[0-9]+", basename(fichier_h5)))
  if (length(suffixe) == 0 || length(rdas) == 0) {
    stop("Impossible de trouver le .rda associe a ", basename(fichier_h5),
         ". Renseigner 'fichier_rda_modele'.")
  }
  cand <- rdas[grepl(suffixe, basename(rdas), fixed = TRUE)]
  if (length(cand) == 0) {
    stop("Aucun .rda de ", dirname(fichier_h5), " ne correspond a '", suffixe,
         "'. Renseigner 'fichier_rda_modele'.")
  }
  cand[which.max(file.mtime(cand))]
}

charger_master <- function(f) {
  env_tmp <- new.env()
  noms    <- load(f, envir = env_tmp)
  obj     <- get(noms[1], envir = env_tmp)
  if (!is.list(obj) || is.null(obj$master_df)) {
    stop("Pas de 'master_df' dans ", f)
  }
  as_tibble(obj$master_df)
}

rda_modele <- if (is.null(fichier_rda_modele)) trouver_rda(fichier_h5) else fichier_rda_modele
rda_test   <- if (is.null(fichier_rda_test)) rda_modele else fichier_rda_test
cat(".rda du modele :", rda_modele, "\n")
cat(".rda des images de test :", rda_test, "\n")

master_modele <- charger_master(rda_modele)
master_test   <- if (rda_test == rda_modele) master_modele else charger_master(rda_test)
if (!("statut" %in% names(master_test))) master_test$statut <- NA_character_


# ---------------------------------------------------------------------
# 4. CHARGEMENT DU MODELE ET DETECTION DU TYPE
# ---------------------------------------------------------------------
# compile = FALSE : on n'a besoin que des predictions, les metriques sont
# calculees ici a la main.
model <- load_model_hdf5(fichier_h5, compile = FALSE)

dim_entree <- unlist(model$input_shape)          # hauteur, largeur, canaux
n_sorties  <- tail(unlist(model$output_shape), 1)
taille_img <- dim_entree[1:2]
type_modele <- if (n_sorties == 1) "binaire" else "multi-classes"

# Keras code les classes dans l'ordre alphabetique de leurs noms
classes_modele <- sort(unique(master_modele$class))
attendu <- if (n_sorties == 1) 2 else n_sorties
if (length(classes_modele) != attendu) {
  stop("Le modele a ", n_sorties, " sortie(s) mais le .rda associe contient ",
       length(classes_modele), " classes (", paste(classes_modele, collapse = ", "), ").")
}

cat("\nType de modele :", type_modele, "\n")
cat("Taille des images en entree (hauteur x largeur) :",
    taille_img[1], "x", taille_img[2], "\n")
cat("Classes (ordre de sortie du modele) :", paste(classes_modele, collapse = " / "), "\n")
if (type_modele == "binaire") {
  cat("Sortie unique = probabilite de la classe '", classes_modele[2], "'\n", sep = "")
}


# ---------------------------------------------------------------------
# 5. CONSTITUTION DES IMAGES DE TEST
# ---------------------------------------------------------------------
pool <- master_test %>% filter(partition == "external")
cat("\nImages externes disponibles dans le .rda de test :", nrow(pool), "\n")

# securite : exclure toute image vue par le modele (train / validation)
vues <- master_modele$path[master_modele$partition %in% c("train", "test")]
n_avant <- nrow(pool)
pool <- pool %>% filter(!(path %in% vues))
if (nrow(pool) < n_avant) {
  warning(n_avant - nrow(pool),
          " image(s) exclue(s) car presentes dans l'entrainement/validation du modele.")
}

# les classes du jeu de test doivent etre des classes connues du modele
if (!all(unique(pool$class) %in% classes_modele)) {
  stop("Le jeu de test contient des classes inconnues du modele : ",
       paste(setdiff(unique(pool$class), classes_modele), collapse = ", "))
}
if (!all(classes_modele %in% unique(pool$class))) {
  warning("Certaines classes du modele sont absentes du jeu de test.")
}

# groupes : classe (x statut si disponible)
a_statut <- any(!is.na(pool$statut))
pool <- pool %>%
  mutate(groupe = if (a_statut) paste(class, statut, sep = " | ") else class)

dispo <- table(pool$groupe)
cat("Images disponibles par groupe :\n"); print(dispo)

if (!is.null(n_test_total)) {
  n_par_groupe <- floor(n_test_total / length(dispo))
} else if (!is.null(n_test_par_groupe)) {
  n_par_groupe <- n_test_par_groupe
} else {
  n_par_groupe <- max(dispo)
}
if (n_par_groupe > min(dispo)) {
  warning("Demande de ", n_par_groupe, " images par groupe, mais le groupe le plus petit n'en ",
          "compte que ", min(dispo), " : on prend tout ce qui est disponible pour ces groupes.")
}
if (n_par_groupe < 1) stop("Nombre d'images de test insuffisant.")

set.seed(seed_eval)
test_df <- pool %>%
  group_by(groupe) %>%
  slice_sample(prop = 1) %>%          # melange aleatoire des images du groupe
  slice_head(n = n_par_groupe) %>%    # garde les n premieres (ou tout si moins de n)
  ungroup()

cat("\nImages de test retenues :", nrow(test_df), "\n")
print(table(test_df$groupe))


# ---------------------------------------------------------------------
# 6. PREDICTIONS
# ---------------------------------------------------------------------
datagen <- image_data_generator(rescale = 1/255)   # meme pretraitement qu'a l'entrainement
gen <- flow_images_from_dataframe(
  as.data.frame(test_df),
  directory   = "",
  x_col       = "path",
  y_col       = "class",
  generator   = datagen,
  target_size = taille_img,
  class_mode  = "categorical",
  batch_size  = batch_size,
  shuffle     = FALSE)

proba <- predict(model, gen, steps = ceiling(nrow(test_df) / batch_size), verbose = 0)

# remettre les lignes dans l'ordre de lecture du generateur
idx <- match(gen$filenames, test_df$path)
if (anyNA(idx)) {
  warning("Association par nom de fichier impossible : ordre des lignes utilise.")
  idx <- seq_len(nrow(test_df))
}
test_df <- test_df[idx, ]

vrai <- factor(test_df$class, levels = classes_modele)

if (type_modele == "binaire") {
  p1      <- as.numeric(proba)
  predit  <- factor(classes_modele[ifelse(p1 > 0.5, 2, 1)], levels = classes_modele)
  y01     <- as.integer(vrai) - 1
  logloss <- -mean(y01 * log(pmax(p1, 1e-15)) + (1 - y01) * log(pmax(1 - p1, 1e-15)))
} else {
  predit  <- factor(classes_modele[max.col(proba, ties.method = "first")],
                    levels = classes_modele)
  p_vraie <- proba[cbind(seq_len(nrow(test_df)), as.integer(vrai))]
  logloss <- -mean(log(pmax(p_vraie, 1e-15)))
}


# ---------------------------------------------------------------------
# 7. EVALUATIONS (choisies automatiquement selon le modele)
# ---------------------------------------------------------------------
cat("\n=====================================================\n")
cat("EVALUATION :", basename(fichier_h5), "\n")
cat("Type :", type_modele, "| images de test :", nrow(test_df), "\n")
cat("=====================================================\n")

accuracy <- mean(vrai == predit)
cat("\nAccuracy :", round(accuracy, 3), "\n")
cat("Perte (entropie croisee) :", round(logloss, 3), "\n")

conf <- table(vraie = vrai, predite = predit)
cat("\n[Matrice de confusion : classe vraie x classe predite]\n")
print(conf)

res <- list(type = type_modele, n_test = nrow(test_df),
            accuracy = accuracy, logloss = logloss, confusion = conf)

if (type_modele == "binaire") {
  # --- evaluations propres au modele binaire ---------------------------
  # classe "positive" = classe codee 1 (la 2e par ordre alphabetique)
  pos <- classes_modele[2]; neg <- classes_modele[1]
  sensibilite <- conf[pos, pos] / sum(conf[pos, ])
  specificite <- conf[neg, neg] / sum(conf[neg, ])
  r  <- rank(p1)
  n1 <- sum(y01 == 1); n0 <- sum(y01 == 0)
  auc <- (sum(r[y01 == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)

  cat("\nClasse positive (sortie du modele proche de 1) :", pos, "\n")
  cat("Sensibilite (", pos, " bien reconnues) :", round(sensibilite, 3), "\n", sep = "")
  cat("Specificite (", neg, " bien reconnues) :", round(specificite, 3), "\n", sep = "")
  cat("AUC :", round(auc, 3), "\n")
  res$sensibilite <- sensibilite; res$specificite <- specificite; res$auc <- auc

} else {
  # --- evaluations propres au modele multi-classes ----------------------
  taux_classe <- diag(conf) / rowSums(conf)
  cat("\nTaux de bonne classification par classe :\n")
  print(round(taux_classe, 3))
  res$taux_par_classe <- taux_classe
}

# --- si plusieurs statuts (domestique / sauvage) dans le jeu de test ---
statuts_presents <- sort(unique(na.omit(test_df$statut)))
if (length(statuts_presents) >= 2) {
  statut <- factor(test_df$statut, levels = statuts_presents)

  cat("\nAccuracy selon le statut :\n")
  acc_statut <- c(tapply(vrai == predit, statut, mean))
  print(round(acc_statut, 3))
  res$accuracy_par_statut <- acc_statut

  if (type_modele == "multi-classes") {
    # lignes : classe vraie ET statut ; colonnes : classes que le modele
    # peut predire (il ne predit jamais le statut)
    lignes <- test_df %>% distinct(class, statut) %>% arrange(class, statut) %>%
      mutate(etiquette = paste(class, statut, sep = " | "))
    ligne <- factor(paste(test_df$class, test_df$statut, sep = " | "),
                    levels = lignes$etiquette)
    conf_statut <- table(vraie_et_statut = ligne, predite = predit)
    cat("\n[Tableau 2 : classe vraie ET statut x classe predite]\n")
    print(conf_statut)

    taux_ligne <- sapply(seq_len(nrow(lignes)), function(i) {
      conf_statut[lignes$etiquette[i], lignes$class[i]] /
        sum(conf_statut[lignes$etiquette[i], ])
    })
    names(taux_ligne) <- lignes$etiquette
    cat("\nTaux de bonne classification par classe et statut :\n")
    print(round(taux_ligne, 3))
    res$confusion_statut <- conf_statut
    res$taux_par_ligne   <- taux_ligne

    conf_max <- apply(proba, 1, max)
    cat("\nConfiance moyenne du modele (probabilite maximale) selon le statut :\n")
    print(round(c(tapply(conf_max, statut, mean)), 3))
  }
}


# ---------------------------------------------------------------------
# 8. SAUVEGARDE
# ---------------------------------------------------------------------
dir.create(dossier_sortie, recursive = TRUE, showWarnings = FALSE)
base <- sprintf("Eval_%s_n%d",
                sub("\\.h5$", "", basename(fichier_h5)), nrow(test_df))

predictions <- tibble(
  path    = test_df$path,
  groupe  = test_df$groupe,
  vrai    = as.character(vrai),
  predit  = as.character(predit),
  correct = vrai == predit)
if (type_modele == "multi-classes") {
  predictions$proba_max <- apply(proba, 1, max)
} else {
  predictions$proba_classe_positive <- p1
}

res$predictions <- predictions
res$config <- list(modele = fichier_h5, rda_modele = rda_modele, rda_test = rda_test,
                   n_par_groupe = n_par_groupe, seed_eval = seed_eval)

save(res, file = file.path(dossier_sortie, paste0(base, ".rda")))
write.csv(predictions, file.path(dossier_sortie, paste0(base, "_predictions.csv")),
          row.names = FALSE)
cat("\nEvaluation ecrite dans :", normalizePath(dossier_sortie), "\n")

