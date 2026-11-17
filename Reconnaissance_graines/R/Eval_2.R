# =====================================================================
# Eval_modele.R
# Evaluation d'un modele sauvegarde (.h5) sur un jeu de test externe.
#
# Compatible avec :
#   - Test1/2/3/3_bis : binaire ou multi-classes, sans rejet
#   - Test5.R         : multi-classes AVEC classe de rejet "autre"
#                       (seuil de confiance sur p_max) et images
#                       Hordeum comme proxy OOD
#
# Pour un modele Test5, produit :
#   - matrice 7 lignes (espece | statut, + "autre") x 3 ou 4 colonnes
#     selon qu'on applique le rejet
#   - accepte_id, rejete_ood, AUC OOD, F1 "autre"
#   - taux correct par ligne (espece | statut)
#
# Pour evaluer sur un AUTRE jeu (archeo plus tard), fournir
# 'fichier_rda_test' : ce .rda doit contenir un master_df avec
# les colonnes path / class / statut / partition. Les lignes de
# statut == "archeo" apparaitront telles quelles dans la matrice.
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine            <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_resultats <- "rda/results"
dossier_sortie    <- "rda/results/evaluations"

# Modele a evaluer : NULL (menu), numero, ou morceau unique du nom
modele_choisi <- "Test5_modele_domestiques_ep200_seed2026.h5"

# Subsampling. Un "groupe" = une ligne de la future matrice de confusion :
#   "olea | domestique", ..., "autre" (Hordeum).
# n_test_total (si renseigne) remplace n_test_par_groupe.
n_test_par_groupe <- NULL
n_test_total      <- NULL

# Fichiers .rda
fichier_rda_modele <- NULL   # NULL = detection auto
fichier_rda_test   <- "rda/db_ajout/archeo_complet_db.rda"   # NULL = meme que modele

# Seuil de rejet : "auto" | "aucun" | valeur numerique
#   "auto"  : seuil calibre lu dans le .rda du modele (Test5 uniquement)
#   "aucun" : pas de classe de rejet
#   numeric : seuil impose (0-1)
seuil_utilise <- "auto"

# Partition pour le test : "external" (recommande) ou "calib"
# "calib" sert uniquement a verifier la coherence avec Test5.R.
partition_eval <- "external"

seed_eval  <- 2329
batch_size <- 32


# ---------------------------------------------------------------------
# 1. PACKAGES
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


# ---------------------------------------------------------------------
# 3. RDA ASSOCIES
# ---------------------------------------------------------------------
# Amelioration : on filtre aussi par "racine" (Test5_, Test3bis_, ...)
# pour eviter qu'un Test5 ne se retrouve associe a un rda Test3bis.

trouver_rda <- function(fichier_h5) {
  rda_meme_nom <- sub("\\.h5$", ".rda", fichier_h5)
  if (file.exists(rda_meme_nom)) return(rda_meme_nom)
  rdas    <- list.files(dirname(fichier_h5), pattern = "\\.rda$", full.names = TRUE)
  suffixe <- regmatches(basename(fichier_h5),
                        regexpr("ep[0-9]+_seed[0-9]+", basename(fichier_h5)))
  if (length(suffixe) == 0 || length(rdas) == 0)
    stop("Impossible de trouver le .rda associe a ", basename(fichier_h5))
  # filtre sur la racine (Test5_, Test3bis_, Test3_, ...)
  racine_h5 <- sub("_.*", "", basename(fichier_h5))
  cand <- rdas[startsWith(basename(rdas), paste0(racine_h5, "_")) &
               grepl(suffixe, basename(rdas), fixed = TRUE)]
  if (length(cand) == 0) {
    # fallback : ancienne logique
    cand <- rdas[grepl(suffixe, basename(rdas), fixed = TRUE)]
  }
  if (length(cand) == 0)
    stop("Aucun .rda de ", dirname(fichier_h5), " ne correspond a '", suffixe, "'.")
  cand[which.max(file.mtime(cand))]
}

charger_resultats <- function(f) {
  env_tmp <- new.env()
  noms    <- load(f, envir = env_tmp)
  obj     <- get(noms[1], envir = env_tmp)
  if (!is.list(obj) || is.null(obj$master_df))
    stop("Pas de 'master_df' dans ", f)
  obj
}

rda_modele <- if (is.null(fichier_rda_modele)) trouver_rda(fichier_h5) else fichier_rda_modele
rda_test   <- if (is.null(fichier_rda_test))   rda_modele       else fichier_rda_test
cat(".rda du modele        :", rda_modele, "\n")
cat(".rda des images test  :", rda_test,   "\n")

resultats_modele <- charger_resultats(rda_modele)
master_modele    <- as_tibble(resultats_modele$master_df)

if (rda_test == rda_modele) {
  resultats_test <- resultats_modele
} else {
  resultats_test <- charger_resultats(rda_test)
}
master_test <- as_tibble(resultats_test$master_df)
if (!("statut" %in% names(master_test))) master_test$statut <- NA_character_

# --- detection Test5 ---------------------------------------------------
# Un .rda Test5 contient 'hordeum_samp' avec les partitions des Hordeum.
# On regarde cote MODELE (pour le seuil) et cote TEST (pour les images OOD).
est_test5_modele <- !is.null(resultats_modele$hordeum_samp)
est_test5_test   <- !is.null(resultats_test$hordeum_samp)

cat("\nGeneration du modele : ",
    if (est_test5_modele) "Test5 (avec rejet)" else "ancienne (sans rejet)", "\n")
if (rda_test != rda_modele)
  cat("Generation du .rda test : ",
      if (est_test5_test) "Test5" else "sans hordeum_samp", "\n")


# ---------------------------------------------------------------------
# 4. CHARGEMENT DU MODELE
# ---------------------------------------------------------------------
model <- load_model_hdf5(fichier_h5, compile = FALSE)

dim_entree  <- unlist(model$input_shape)
n_sorties   <- tail(unlist(model$output_shape), 1)
taille_img  <- dim_entree[1:2]
type_modele <- if (n_sorties == 1) "binaire" else "multi-classes"

classes_modele <- sort(unique(master_modele$class))
attendu <- if (n_sorties == 1) 2 else n_sorties
if (length(classes_modele) != attendu)
  stop("Le modele a ", n_sorties, " sortie(s) mais le .rda associe contient ",
       length(classes_modele), " classes.")

cat("\nType de modele :", type_modele, "\n")
cat("Taille image   :", taille_img[1], "x", taille_img[2], "\n")
cat("Classes        :", paste(classes_modele, collapse = " / "), "\n")


# ---------------------------------------------------------------------
# 5. DETERMINATION DU SEUIL
# ---------------------------------------------------------------------
detecter_seuil <- function(res, fn_h5) {
  if (is.null(res$hordeum_samp)) return(NULL)   # pas un Test5
  fn_lower <- tolower(basename(fn_h5))
  if (!is.null(res$res_dom) && grepl("domestique", fn_lower)) {
    s <- res$res_dom$calib$seuil
    if (!is.null(s) && !is.na(s))
      return(list(seuil = s, source = "res_dom$calib$seuil"))
  }
  if (!is.null(res$res_wild) && grepl("sauvage", fn_lower)) {
    s <- res$res_wild$calib$seuil
    if (!is.null(s) && !is.na(s))
      return(list(seuil = s, source = "res_wild$calib$seuil"))
  }
  NULL
}

seuil <- NA_real_
source_seuil <- "aucun"
if (is.numeric(seuil_utilise)) {
  seuil <- as.numeric(seuil_utilise)
  source_seuil <- "impose par l'utilisateur"
} else if (identical(seuil_utilise, "auto")) {
  det <- detecter_seuil(resultats_modele, fichier_h5)
  if (!is.null(det)) {
    seuil <- det$seuil; source_seuil <- det$source
  } else {
    message("Seuil 'auto' introuvable ; evaluation sans rejet.")
  }
}

cat("\nSeuil de rejet : ",
    if (is.na(seuil)) "AUCUN (pas de classe 'autre' en sortie)\n"
    else sprintf("%.4f  (source : %s)\n", seuil, source_seuil))


# ---------------------------------------------------------------------
# 6. CONSTRUCTION DU JEU DE TEST (ID + OOD si Test5)
# ---------------------------------------------------------------------
# --- 6.1 images ID et OOD issues de master_test -------------------------
pool_all <- master_test %>% filter(partition == partition_eval)
is_known <- pool_all$class %in% classes_modele

# --- 6.1a ID : classes connues du modele -------------------------------
pool_id <- pool_all[is_known, ]
if (all(is.na(pool_id$statut))) {
  pool_id <- pool_id %>% mutate(vraie_complete = class)
} else {
  pool_id <- pool_id %>% mutate(vraie_complete = paste(class, statut, sep = " | "))
}
pool_id <- pool_id %>% mutate(source = "id")

vues <- master_modele$path[master_modele$partition %in% c("train", "test")]
n_avant <- nrow(pool_id)
pool_id <- pool_id %>% filter(!(path %in% vues))
if (nrow(pool_id) < n_avant)
  warning(n_avant - nrow(pool_id), " image(s) exclue(s).")

if (!all(classes_modele %in% unique(pool_id$class)))
  warning("Certaines classes du modele sont absentes du jeu de test ID.")

# --- 6.2 OOD : depuis master_test (class inconnue) et/ou hordeum_samp --
pool_ood <- NULL

ood_from_master <- pool_all[!is_known, ]
if (nrow(ood_from_master) > 0) {
  pool_ood <- tibble(
    path           = ood_from_master$path,
    class          = ood_from_master$class,
    statut         = NA_character_,
    partition      = partition_eval,
    source         = "ood",
    vraie_complete = "autre"
  )
  cat("Images OOD depuis master_df :", nrow(pool_ood), "\n")
}

if (est_test5_test) {
  h <- as_tibble(resultats_test$hordeum_samp)
  h_ext <- h %>% filter(partition == partition_eval)
  if (nrow(h_ext) > 0) {
    h_tibble <- tibble(
      path           = h_ext$path,
      class          = "hordeum",
      statut         = NA_character_,
      partition      = partition_eval,
      source         = "ood",
      vraie_complete = "autre"
    )
    pool_ood <- bind_rows(pool_ood, h_tibble)
    cat("Images OOD depuis hordeum_samp :", nrow(h_tibble), "\n")
  }
}

# --- 6.3 concaténation -------------------------------------------------
pool <- bind_rows(pool_id, pool_ood)
# --- 6.4 sous-echantillonnage -----------------------------------------
# Les "groupes" de tirage sont les lignes de la future matrice :
#   * ID : "espece | statut" (ou "espece" si statut absent)
#   * OOD : "autre"
dispo <- table(pool$vraie_complete)
cat("\nImages disponibles par groupe :\n"); print(dispo)

if (!is.null(n_test_total)) {
  n_par_groupe <- floor(n_test_total / length(dispo))
} else if (!is.null(n_test_par_groupe)) {
  n_par_groupe <- n_test_par_groupe
} else {
  n_par_groupe <- max(dispo)
}
if (n_par_groupe > min(dispo))
  warning("Demande de ", n_par_groupe, " images par groupe ; le groupe le plus ",
          "petit n'en a que ", min(dispo), " : tout ce qui existe sera pris.")

set.seed(seed_eval)
test_df <- pool %>%
  group_by(vraie_complete) %>%
  slice_sample(prop = 1) %>%
  slice_head(n = n_par_groupe) %>%
  ungroup()

cat("\nImages de test retenues :", nrow(test_df), "\n")
print(table(test_df$vraie_complete))


# ---------------------------------------------------------------------
# 7. PREDICTIONS
# ---------------------------------------------------------------------
datagen <- image_data_generator(rescale = 1/255)
gen <- flow_images_from_dataframe(
  as.data.frame(test_df), directory = "",
  x_col = "path", y_col = "class", generator = datagen,
  target_size = taille_img, class_mode = "categorical",
  batch_size = batch_size, shuffle = FALSE)

proba <- predict(model, gen, steps = ceiling(nrow(test_df) / batch_size), verbose = 0)

# reordonner comme test_df
idx <- match(gen$filenames, test_df$path)
if (anyNA(idx)) {
  warning("Association par nom de fichier impossible : ordre des lignes utilise.")
  idx <- seq_len(nrow(test_df))
}
proba   <- proba[idx, , drop = FALSE]
test_df <- test_df[idx, ]

# --- 7.1 predictions selon le type de modele et le seuil --------------
if (type_modele == "binaire") {
  p1 <- as.numeric(proba)
  predit  <- factor(classes_modele[ifelse(p1 > 0.5, 2, 1)], levels = classes_modele)
  p_max   <- pmax(p1, 1 - p1)
  pred_species <- as.character(predit)
} else {
  p_max        <- apply(proba, 1, max)
  pred_species <- classes_modele[max.col(proba, ties.method = "first")]
  if (!is.na(seuil)) {
    niveaux_predit <- c(classes_modele, "autre")
    predit <- factor(ifelse(p_max < seuil, "autre", pred_species),
                     levels = niveaux_predit)
  } else {
    niveaux_predit <- classes_modele
    predit <- factor(pred_species, levels = niveaux_predit)
  }
}

# --- 7.2 valeur attendue par ligne -----------------------------------
# ID : la vraie espece. OOD (Hordeum) : "autre".
is_ood   <- test_df$source == "ood"
expected <- ifelse(is_ood, "autre", test_df$class)

# --- 7.3 logloss (ID seulement) --------------------------------------
id_idx <- which(!is_ood)
logloss_id <- NA_real_
if (length(id_idx) > 0 && type_modele == "multi-classes") {
  vrai_idx  <- match(test_df$class[id_idx], classes_modele)
  p_vraie   <- proba[cbind(id_idx, vrai_idx)]
  logloss_id <- -mean(log(pmax(p_vraie, 1e-15)))
} else if (length(id_idx) > 0) {
  p1_id  <- p1[id_idx]
  y01_id <- as.integer(test_df$class[id_idx] == classes_modele[2])
  logloss_id <- -mean(y01_id * log(pmax(p1_id, 1e-15)) +
                      (1 - y01_id) * log(pmax(1 - p1_id, 1e-15)))
}


# ---------------------------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------------------------
cat("\n=====================================================\n")
cat("EVALUATION :", basename(fichier_h5), "\n")
cat("Type       :", type_modele,
    " | test :", nrow(test_df),
    " | seuil :", if (is.na(seuil)) "aucun" else round(seuil, 3), "\n")
cat("=====================================================\n")

# --- 8.1 metriques globales ------------------------------------------
accuracy_avec_rejet <- mean(as.character(predit) == expected)
cat("\nAccuracy avec rejet :", round(accuracy_avec_rejet, 3), "\n")
if (!is.na(logloss_id))
  cat("Logloss (ID seulement) :", round(logloss_id, 3), "\n")

if (!is.na(seuil)) {
  accepte_id <- mean(as.character(predit[!is_ood]) != "autre")
  correct_id <- mean(as.character(predit[!is_ood]) == test_df$class[!is_ood])
  rejete_ood <- if (any(is_ood)) mean(as.character(predit[is_ood]) == "autre") else NA
  cat("% ID acceptes            :", round(accepte_id, 3), "\n")
  cat("% ID correctement classes:", round(correct_id, 3), "\n")
  if (!is.na(rejete_ood))
    cat("% OOD rejetes            :", round(rejete_ood, 3), "\n")

  # AUC OOD (1 - p_max comme score d'anomalie)
  auc_ood <- NA_real_
  if (any(is_ood) && any(!is_ood)) {
    y_ood  <- as.integer(is_ood)
    scores <- 1 - p_max
    r  <- rank(scores)
    n1 <- sum(y_ood == 1); n0 <- sum(y_ood == 0)
    auc_ood <- (sum(r[y_ood == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
    cat("AUC OOD                 :", round(auc_ood, 3), "\n")
  }

  # F1 "autre"
  tp <- sum(predit == "autre" & is_ood)
  fp <- sum(predit == "autre" & !is_ood)
  fn <- sum(predit != "autre" &  is_ood)
  prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  rec  <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0)
            2 * prec * rec / (prec + rec) else NA_real_
  cat("F1 'autre'              :", round(f1, 3),
      "(prec", round(prec, 3), "/ rec", round(rec, 3), ")\n")
}

# --- 8.2 matrice detaillee -------------------------------------------
cat("\n[Matrice : vraie espece|statut  x  classe predite]\n")
niveaux_vraie <- unique(test_df$vraie_complete)
if ("autre" %in% niveaux_vraie){
  niveaux_vraie <- c(sort(setdiff(niveaux_vraie, "autre")), "autre")}
  niveaux_vraie <- sort(niveaux_vraie)
vraie_f  <- factor(test_df$vraie_complete, levels = niveaux_vraie)
predite_f <- predit

conf_espece_statut <- table(vraie = vraie_f, predite = predite_f)
print(conf_espece_statut)

# taux de bonne classification par ligne
taux_correct <- sapply(niveaux_vraie, function(lbl) {
  idx <- which(as.character(vraie_f) == lbl)
  if (length(idx) == 0) NA_real_
  else mean(as.character(predit[idx]) == sub(" \\|.*", "", lbl) |
            (lbl == "autre" & as.character(predit[idx]) == "autre"))
})
names(taux_correct) <- niveaux_vraie
cat("\nTaux de bonne classification par ligne :\n")
print(round(taux_correct, 3))

# --- 8.3 accuracy par statut -----------------------------------------
if ("statut" %in% names(test_df) && length(unique(na.omit(test_df$statut))) >= 2) {
  statut <- factor(test_df$statut, levels = sort(unique(na.omit(test_df$statut))))
  is_correct <- as.character(predit) == test_df$class  # uniquement pour ID
  cat("\nAccuracy ID selon le statut :\n")
  print(round(c(tapply(is_correct[!is_ood], statut[!is_ood], mean)), 3))
}

# --- 8.4 rappel : coherence avec les metriques de Test5.R -------------
if (!is.na(seuil) && est_test5_modele) {
  cat("\n[Coherence avec Test5.R - rappel]\n")
  ref <- if (grepl("domestique", tolower(basename(fichier_h5))))
           resultats_modele$res_dom$eval_officiel
         else if (grepl("sauvage", tolower(basename(fichier_h5))))
           resultats_modele$res_wild$eval_officiel
         else NULL
  if (!is.null(ref)) {
    cat("  AUC_test (Test5.R)      :", round(ref$auc_ood, 3), "\n")
    cat("  accepte_id (Test5.R)    :", round(ref$accepte_id, 3), "\n")
    cat("  rejete_ood (Test5.R)    :", round(ref$rejete_ood, 3), "\n")
    cat("  (les valeurs ci-dessus different legerement de celles\n")
    cat("   calculees ici si n_test_par_groupe < effectif complet)\n")
  }
}


# ---------------------------------------------------------------------
# 9. SAUVEGARDE
# ---------------------------------------------------------------------
dir.create(dossier_sortie, recursive = TRUE, showWarnings = FALSE)
base <- sprintf("Eval_%s_n%d",
                sub("\\.h5$", "", basename(fichier_h5)), nrow(test_df))

predictions <- tibble(
  path           = test_df$path,
  source         = test_df$source,
  vraie_complete = test_df$vraie_complete,
  vrai_espece    = test_df$class,
  pred_species   = pred_species,
  predit         = as.character(predit),
  expected       = expected,
  correct        = as.character(predit) == expected,
  p_max          = p_max)
if (type_modele == "multi-classes") {
  for (i in seq_along(classes_modele))
    predictions[[paste0("proba_", classes_modele[i])]] <- proba[, i]
} else {
  predictions$proba_classe_positive <- p1
}

res <- list(
  type = type_modele, seuil = seuil, source_seuil = source_seuil,
  n_test = nrow(test_df),
  accuracy_avec_rejet = accuracy_avec_rejet, logloss_id = logloss_id,
  confusion = conf_espece_statut, taux_correct_par_ligne = taux_correct,
  predictions = predictions,
  config = list(modele = fichier_h5, rda_modele = rda_modele,
                rda_test = rda_test, n_par_groupe = n_par_groupe,
                seed_eval = seed_eval, partition_eval = partition_eval))

save(res, file = file.path(dossier_sortie, paste0(base, ".rda")))
write.csv(predictions,
          file.path(dossier_sortie, paste0(base, "_predictions.csv")),
          row.names = FALSE)
cat("\nEvaluation ecrite dans :", normalizePath(dossier_sortie), "\n")
