# =====================================================================
# Test5.R  (version adaptee a 23 images archeo)
# Modele UNIQUE entraine sur domestiques + sauvages confondus (1:1),
# avec classe de rejet "autre" via seuil de confiance sur p_max.
#
# CONTRAINTE MAJEURE : 23 images archeo au total, dont certaines
# especes avec 1 seule image. Consequences assumees :
#   - Pas de partition calib/external sur l'archeo : les 23 images
#     servent a la fois a calibrer et a evaluer (flag explicite).
#   - Critere de calibration robuste : COMPTAGE d'images correctes,
#     pas de taux (instable a n=1).
#   - L'evaluation archeo n'est PAS une validation independante :
#     c'est une mesure de coherence du seuil sur les images qui ont
#     servi a le choisir. Documente comme tel.
#
# Sources :
#   olea_db2.rda, phoenix_db2.rda, vitis_db2.rda  -> ID (dom + sauv)
#   archeo_complet_db.rda                         -> archeo (23 images)
#   autres_especes_db.rda                         -> autre (OOD)
#
# Critere de calibration (robuste petit n) :
#   1) Contrainte : taux_ood_rejete >= seuil_ood_min_cible
#   2) Parmi les seuils admissibles : max N_arch_correct (entier)
#   3) Egalite : min N_arch_rejetees, puis plus petit seuil
#   Si aucun seuil n'est admissible : repli = max(N_arch_correct -
#   N_arch_rejetees + taux_ood_rejete*N_ood), documente comme tel.
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db       <- "rda/db"
dossier_db_ajout <- "rda/db_ajout"
dossier_res <- "rda/results/especes_avec_rejet"

fichier_archeo <- file.path(dossier_db_ajout, "archeo_complet_db.rda")
fichier_autre  <- file.path(dossier_db_ajout, "autres_especes_db.rda")

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")
col_image <- "VD_deep"

statut_domestique <- "c"
statut_sauvage    <- "w"

# Effectifs ID (domestiques / sauvages). Inchange.
N_train_par_statut <- 240
N_test_par_statut  <- 60
N_ext_par_statut   <- 100

# Autre (OOD)
N_autre_ext <- 650

# Archeo : AUCUN parametre d'effectif. On prend TOUT ce qui existe.
# [MODIF] Flag de separation calib/eval archeo.
#   FALSE tant qu'on a <= ~50 archeo par espece : les memes images
#   servent a calibrer et a evaluer (assume explicitement).
#   TRUE si tu obtiens assez d'archeo plus tard : le script separe
#   alors en deux moities (a activer manuellement).
calib_et_eval_sur_memes_archeo <- TRUE

prop_calib <- 0.5   # ne s'applique qu'a 'autre'

# --- CALIBRATION DU SEUIL -------------------------------------------
seuil_ood_min_cible <- 0.80
grille_seuils       <- seq(0.05, 0.99, by = 0.01)

# --- Entrainement ----------------------------------------------------
epochs      <- 5
seed        <- 2026
seed_wild   <- seed + 1
seed_autre  <- seed + 2
batch_size  <- 32
input_shape <- c(120, 90, 3)


# ---------------------------------------------------------------------
# 1. PACKAGES
# ---------------------------------------------------------------------
library(tidyverse)
library(keras)
py_require_legacy_keras()
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")
begin_time <- Sys.time()


# ---------------------------------------------------------------------
# 2. FONCTIONS DE CHARGEMENT  (identiques a la version precedente)
# ---------------------------------------------------------------------
charger_taxon <- function(taxon, dossier_img) {
  cat("\n=====", toupper(taxon), "=====\n")
  fichiers <- list.files(dossier_db, pattern = taxon,
                         full.names = TRUE, ignore.case = TRUE)
  if (length(fichiers) == 0) stop("Aucun .rda '", taxon, "' dans ", dossier_db)
  prefere <- fichiers[tolower(basename(fichiers)) == paste0(taxon, "_db2.rda")]
  fichier <- if (length(prefere) > 0) prefere[1] else fichiers[1]
  cat("Fichier .rda :", fichier, "\n")
  env_tmp <- new.env(); noms <- load(fichier, envir = env_tmp)
  db <- get(noms[1], envir = env_tmp)
  if (!all(c("status", col_image) %in% names(db))) {
    print(names(db)); stop("Colonnes manquantes pour ", taxon)
  }
  db <- db[as.character(db$status) %in% c(statut_domestique, statut_sauvage), ]
  chemins <- as.character(db[[col_image]])
  if (!all(file.exists(chemins))) {
    tous <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    idx  <- match(basename(chemins), basename(tous))
    if (any(is.na(idx))) stop(sum(is.na(idx)), " images introuvables")
    chemins <- tous[idx]
  }
  tibble(taxon = taxon, statut = as.character(db$status), path = chemins)
}

charger_archeo <- function(fichier_rda, especes_valides) {
  cat("\n===== ARCHEO (", basename(fichier_rda), ") =====\n")
  if (!file.exists(fichier_rda)) {
    warning("Fichier archeo introuvable : ", fichier_rda)
    return(tibble(taxon = character(), path = character()))
  }
  env_tmp <- new.env(); noms <- load(fichier_rda, envir = env_tmp)
  res <- get(noms[1], envir = env_tmp)
  if (is.null(res$master_df)) stop("Pas de master_df dans ", fichier_rda)
  master <- as_tibble(res$master_df)
  
  archeo <- master %>%
    filter(statut == "archeo") %>%
    transmute(taxon = as.character(class), path = as.character(path))
  
  hors_espece <- setdiff(unique(archeo$taxon), especes_valides)
  if (length(hors_espece) > 0) {
    warning("Archeo avec classes hors modele exclues : ",
            paste(hors_espece, collapse = ", "))
    archeo <- archeo %>% filter(taxon %in% especes_valides)
  }
  manquants <- !file.exists(archeo$path)
  if (any(manquants)) {
    warning(sum(manquants), " images archeo introuvables. Exclues.")
    archeo <- archeo[!manquants, ]
  }
  cat("Archeo par espece :\n"); print(table(archeo$taxon))
  cat("TOTAL archeo :", nrow(archeo), "\n")
  archeo
}

charger_autre <- function(fichier_rda) {
  cat("\n===== AUTRE / OOD (", basename(fichier_rda), ") =====\n")
  if (!file.exists(fichier_rda)) {
    warning("Fichier autre introuvable : ", fichier_rda)
    return(tibble(path = character()))
  }
  env_tmp <- new.env(); noms <- load(fichier_rda, envir = env_tmp)
  res <- get(noms[1], envir = env_tmp)
  if (is.null(res$master_df)) stop("Pas de master_df dans ", fichier_rda)
  master <- as_tibble(res$master_df)
  autre <- master %>%
    filter(class == "autre" | statut == "autre_espece") %>%
    transmute(path = as.character(path))
  manquants <- !file.exists(autre$path)
  if (any(manquants)) {
    warning(sum(manquants), " images autre introuvables. Exclues.")
    autre <- autre[!manquants, ]
  }
  cat("Autres especes (OOD) disponibles :", nrow(autre), "\n")
  autre
}


# ---------------------------------------------------------------------
# 3. CHARGEMENT
# ---------------------------------------------------------------------
tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]])))
print(table(tous_les_taxons$taxon, tous_les_taxons$statut))

especes_valides <- sort(names(taxons))
archeo_all <- charger_archeo(fichier_archeo, especes_valides)
autre_all  <- charger_autre(fichier_autre)

if (nrow(archeo_all) == 0) {
  stop("Aucune image archeo disponible : impossible de calibrer le seuil ",
       "selon le critere 'maximiser les archeo correctes'.")
}
if (nrow(autre_all) == 0) {
  stop("Aucune image 'autre' disponible : impossible de calibrer la ",
       "contrainte de rejet OOD.")
}


# ---------------------------------------------------------------------
# 4. PARTITIONS
# ---------------------------------------------------------------------
# 4.1 & 4.2  ID : identiques a la version precedente
N_train_d <- N_train_par_statut; N_test_d <- N_test_par_statut
N_ext_d   <- N_ext_par_statut
labs_dom  <- c(rep("train", N_train_d), rep("test", N_test_d),
               rep("external", N_ext_d))
set.seed(seed); shuffled_dom <- sample(labs_dom)
dom <- tous_les_taxons %>% filter(statut == statut_domestique)
if (any(table(dom$taxon) < length(labs_dom)))
  stop("Pas assez de graines domestiques par taxon (besoin ",
       length(labs_dom), "). Phoenix c est le facteur limitant.")
df_dom <- dom %>% group_by(taxon) %>% sample_n(length(labs_dom)) %>%
  mutate(partition = shuffled_dom) %>% ungroup()

N_train_w <- N_train_par_statut; N_test_w <- N_test_par_statut
N_ext_w   <- N_ext_par_statut
labs_wild <- c(rep("train", N_train_w), rep("test", N_test_w),
               rep("external", N_ext_w))
set.seed(seed_wild); shuffled_wild <- sample(labs_wild)
wild <- tous_les_taxons %>% filter(statut == statut_sauvage)
if (any(table(wild$taxon) < length(labs_wild)))
  stop("Pas assez de graines sauvages par taxon (besoin ",
       length(labs_wild), "). Olea w est le facteur limitant.")
df_wild <- wild %>% group_by(taxon) %>% sample_n(length(labs_wild)) %>%
  mutate(partition = shuffled_wild) %>% ungroup()

lib_statut <- setNames(c("domestique", "sauvage"),
                       c(statut_domestique, statut_sauvage))
master_df <- bind_rows(df_dom, df_wild) %>%
  transmute(partition, class = taxon,
            statut = unname(lib_statut[statut]), path)

cat("\nPartition x espece x statut (ID) :\n")
print(ftable(table(partition = master_df$partition,
                   espece    = master_df$class,
                   statut    = master_df$statut)))

train_df    <- master_df %>% filter(partition == "train")
val_df      <- master_df %>% filter(partition == "test")
calib_id_df <- master_df %>% filter(partition == "external")
test_id_df  <- master_df %>% filter(partition == "external")

# 4.3 [MODIF] Archeo : une seule partition archeo_eval (les 23 images)
# Si calib_et_eval_sur_memes_archeo = FALSE (futur, avec + d'images),
# on separe en deux moities par espece.
if (calib_et_eval_sur_memes_archeo) {
  archeo_calib <- archeo_all
  archeo_test  <- archeo_all
  cat("\n[ARCHEO] Une seule partition utilisee pour calib ET eval :",
      nrow(archeo_all), "images.\n")
  cat("  -> L'evaluation archeo N'EST PAS une validation independante.\n")
} else {
  set.seed(seed_autre + 1)
  archeo_split <- archeo_all %>%
    group_by(taxon) %>%
    mutate(partition_arch = sample(rep(c("calib", "external"),
                                       length.out = n()))) %>%
    ungroup()
  archeo_calib <- archeo_split %>% filter(partition_arch == "calib") %>%
    select(taxon, path)
  archeo_test  <- archeo_split %>% filter(partition_arch == "external") %>%
    select(taxon, path)
  cat("\n[ARCHEO] Separation calib/eval :\n")
  cat("  calib   :", nrow(archeo_calib), "\n")
  cat("  external:", nrow(archeo_test),  "\n")
  if (any(table(archeo_test$taxon) < 2))
    warning("Certaines especes ont <2 images en external : taux non interpretables.")
}

# 4.4 Autre (OOD) : calib / external (inchange)
N_autre_total <- min(N_autre_ext, nrow(autre_all))
N_autre_calib <- round(N_autre_total * prop_calib)
N_autre_test  <- N_autre_total - N_autre_calib
set.seed(seed_autre)
autre_samp <- autre_all %>%
  slice_sample(n = N_autre_total) %>%
  mutate(partition = sample(c(rep("calib", N_autre_calib),
                              rep("external", N_autre_test))))
autre_calib <- autre_samp %>% filter(partition == "calib")
autre_test  <- autre_samp %>% filter(partition == "external")
ood_calib_df <- autre_calib %>% transmute(path)
ood_test_df  <- autre_test  %>% transmute(path)

cat("\nTailles des partitions :\n")
cat("  train              :", nrow(train_df), "\n")
cat("  test (validation)  :", nrow(val_df), "\n")
cat("  external ID        :", nrow(test_id_df), "\n")
cat("  calib archeo       :", nrow(archeo_calib), "\n")
cat("  external archeo    :", nrow(archeo_test), "\n")
cat("  calib OOD (autre)  :", nrow(ood_calib_df), "\n")
cat("  external OOD       :", nrow(ood_test_df), "\n")


# ---------------------------------------------------------------------
# 5. FONCTIONS COMMUNES  (identiques a la version precedente)
# ---------------------------------------------------------------------
all_datagen <- image_data_generator(rescale = 1/255)

flux <- function(x, melanger = TRUE) {
  flow_images_from_dataframe(
    x, directory = "", x_col = "path", y_col = "class",
    generator = all_datagen, target_size = input_shape[1:2],
    class_mode = "categorical", batch_size = batch_size,
    shuffle = melanger, seed = seed)
}

predire_sur_chemins <- function(model, chemins, target_size, batch_size) {
  n <- length(chemins); H <- target_size[1]; W <- target_size[2]
  if (n == 0) return(matrix(numeric(0), nrow = 0, ncol = 3))
  arr <- array(0, dim = c(n, H, W, 3))
  for (i in seq_len(n)) {
    img <- image_load(chemins[i], target_size = target_size, color_mode = "rgb")
    arr[i,,,] <- image_to_array(img) / 255
  }
  predict(model, arr, batch_size = batch_size, verbose = 0)
}

predire_sur_df <- function(model, df) {
  if (nrow(df) == 0) return(matrix(numeric(0), nrow = 0, ncol = 3))
  gen   <- flux(df, melanger = FALSE)
  n     <- nrow(df)
  steps <- ceiling(n / batch_size)
  proba <- predict(model, gen, steps = steps, verbose = 0)
  if (nrow(proba) > n) proba <- proba[seq_len(n), , drop = FALSE]
  idx <- match(gen$filenames, df$path)
  if (!anyNA(idx) && nrow(proba) == n) proba <- proba[idx, , drop = FALSE]
  proba
}

auc_id_ood <- function(proba_id, proba_ood) {
  if (nrow(proba_id) == 0 || nrow(proba_ood) == 0) return(NA_real_)
  c_id  <- apply(proba_id,  1, max)
  c_ood <- apply(proba_ood, 1, max)
  scores <- c(1 - c_id, 1 - c_ood)
  labels <- c(rep(0L, length(c_id)), rep(1L, length(c_ood)))
  r  <- rank(scores)
  n1 <- sum(labels == 1L); n0 <- sum(labels == 0L)
  (sum(r[labels == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}


# ---------------------------------------------------------------------
# 6. [MODIF] CALIBRATION DU SEUIL — COMPTAGE, pas taux
# ---------------------------------------------------------------------
# On renvoie le nombre d'archeo correctes (entier) a chaque seuil, pas
# le taux. Sur 23 images, un comptage est plus stable qu'une moyenne.
choisir_seuil_archeo <- function(proba_arch, espece_arch,
                                 proba_ood, noms_classes,
                                 seuil_ood_min = 0.80,
                                 grille = grille_seuils) {
  if (nrow(proba_arch) == 0) stop("Pas d'images archeo.")
  if (nrow(proba_ood)  == 0) stop("Pas d'images OOD.")
  
  conf_arch       <- apply(proba_arch, 1, max)
  pred_arch_brute <- noms_classes[max.col(proba_arch)]
  conf_ood        <- apply(proba_ood,  1, max)
  pred_ood_brute  <- noms_classes[max.col(proba_ood)]
  n_arch <- length(conf_arch)
  n_ood  <- length(conf_ood)
  
  metriques <- lapply(grille, function(s) {
    pred_arch <- ifelse(conf_arch < s, "autre", pred_arch_brute)
    pred_ood  <- ifelse(conf_ood  < s, "autre", pred_ood_brute)
    data.frame(
      seuil             = s,
      n_arch_correct    = sum(pred_arch == espece_arch),
      n_arch_rejetees   = sum(pred_arch == "autre"),
      taux_arch_correct = mean(pred_arch == espece_arch),
      taux_ood_rejete   = mean(pred_ood  == "autre"))
  })
  metriques <- bind_rows(metriques)
  
  admissibles <- metriques %>% filter(taux_ood_rejete >= seuil_ood_min)
  
  if (nrow(admissibles) == 0) {
    warning("Aucun seuil ne garantit taux_ood_rejete >= ", seuil_ood_min,
            ". Repli : score = n_arch_correct - n_arch_rejetees + ",
            "taux_ood_rejete * n_ood (documente comme non optimal).")
    metriques <- metriques %>%
      mutate(score = n_arch_correct - n_arch_rejetees +
               taux_ood_rejete * n_ood)
    best <- metriques %>% slice_max(score, n = 1, with_ties = FALSE)
    critere <- "repli_compromis"
  } else {
    # Priorite : max n_arch_correct ; egalite : min n_arch_rejetees ;
    # egalite : plus petit seuil (moins agressif).
    best <- admissibles %>%
      arrange(desc(n_arch_correct), n_arch_rejetees, seuil) %>%
      slice(1)
    critere <- "sous_contrainte_ood"
  }
  
  # Stabilite : combien de seuils donnent n_arch_correct = max ?
  n_max_correct <- max(metriques$n_arch_correct)
  seuils_opt    <- metriques %>% filter(n_arch_correct == n_max_correct) %>%
    pull(seuil)
  etendue_opt   <- range(seuils_opt)
  
  list(seuil             = best$seuil,
       n_arch_correct    = best$n_arch_correct,
       n_arch_total      = n_arch,
       n_arch_rejetees   = best$n_arch_rejetees,
       taux_arch_correct = best$taux_arch_correct,
       taux_ood_rejete   = best$taux_ood_rejete,
       critere           = critere,
       courbe            = metriques,
       admissibles       = admissibles,
       seuil_ood_min     = seuil_ood_min,
       n_max_correct     = n_max_correct,
       etendue_opt_seuils = etendue_opt)
}


# ---------------------------------------------------------------------
# 7. EVALUATION : matrice 10 x 4 (identique a la version precedente)
# ---------------------------------------------------------------------
evaluer_10x4 <- function(proba_id, espece_id, statut_id,
                         proba_arch, espece_arch,
                         proba_ood,
                         seuil, noms_classes) {
  pred_id   <- ifelse(apply(proba_id,   1, max) < seuil, "autre",
                      noms_classes[max.col(proba_id)])
  pred_arch <- ifelse(apply(proba_arch, 1, max) < seuil, "autre",
                      noms_classes[max.col(proba_arch)])
  pred_ood  <- ifelse(apply(proba_ood,  1, max) < seuil, "autre",
                      noms_classes[max.col(proba_ood)])
  
  vrai_id   <- paste(as.character(espece_id), as.character(statut_id),
                     sep = " | ")
  vrai_arch <- paste(as.character(espece_arch), "archeo", sep = " | ")
  vrai_ood  <- rep("autre", length(pred_ood))
  
  niveaux_vrai <- c(
    as.vector(sapply(noms_classes, function(e)
      c(paste(e, "domestique", sep = " | "),
        paste(e, "sauvage",    sep = " | "),
        paste(e, "archeo",     sep = " | ")))),
    "autre")
  niveaux_pred <- c(noms_classes, "autre")
  
  vrai_f <- factor(c(vrai_id, vrai_arch, vrai_ood), levels = niveaux_vrai)
  pred_f <- factor(c(pred_id, pred_arch, pred_ood), levels = niveaux_pred)
  conf   <- table(vraie = vrai_f, predite = pred_f)
  
  taux_espece_brute <- sapply(niveaux_vrai, function(lbl) {
    if (lbl == "autre") return(NA_real_)
    esp <- sub(" \\|.*", "", lbl)
    idx <- which(as.character(vrai_f) == lbl)
    if (length(idx) == 0) NA_real_ else mean(pred_f[idx] == esp)
  })
  names(taux_espece_brute) <- niveaux_vrai
  
  taux_correct_final <- sapply(niveaux_vrai, function(lbl) {
    idx <- which(as.character(vrai_f) == lbl)
    if (length(idx) == 0) return(NA_real_)
    if (lbl == "autre") mean(pred_f[idx] == "autre")
    else mean(pred_f[idx] == sub(" \\|.*", "", lbl))
  })
  names(taux_correct_final) <- niveaux_vrai
  
  n_total <- length(pred_id) + length(pred_arch) + length(pred_ood)
  correct_id   <- pred_id   == as.character(espece_id)
  correct_arch <- pred_arch == as.character(espece_arch)
  correct_ood  <- pred_ood  == "autre"
  acc_4 <- (sum(correct_id) + sum(correct_arch) + sum(correct_ood)) / n_total
  
  list(seuil             = seuil,
       confusion          = conf,
       taux_espece_brute  = taux_espece_brute,
       taux_correct_final = taux_correct_final,
       acc_4              = acc_4,
       accepte_id         = mean(pred_id  != "autre"),
       rejete_ood         = mean(pred_ood == "autre"),
       rejete_arch        = mean(pred_arch == "autre"),
       taux_arch_correct  = mean(correct_arch),
       pred_id            = pred_id,
       pred_arch          = pred_arch,
       pred_ood           = pred_ood)
}


# ---------------------------------------------------------------------
# 8. ENTRAINEMENT + CALIBRATION + EVALUATION
# ---------------------------------------------------------------------
k_clear_session()
try(tensorflow::set_random_seed(seed), silent = TRUE)

train_generator <- flux(train_df)
val_generator   <- flux(val_df)
cat("Codage des classes :\n"); print(train_generator$class_indices)

base_model <- application_vgg19(weights = "imagenet",
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
callbacks_list <- list(
  callback_early_stopping(monitor = "accuracy", patience = 20),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 1e-7))
history <- model %>% fit(
  train_generator, steps_per_epoch = ceiling(nrow(train_df)/batch_size),
  epochs = epochs, validation_data = val_generator,
  validation_steps = ceiling(nrow(val_df)/batch_size),
  callbacks = callbacks_list)
print(plot(history))

cat("\nPredictions sur la partition calib...\n")
proba_calib_id   <- predire_sur_df(model, calib_id_df)
proba_calib_arch <- predire_sur_chemins(model, archeo_calib$path,
                                        input_shape[1:2], batch_size)
proba_calib_ood  <- predire_sur_chemins(model, ood_calib_df$path,
                                        input_shape[1:2], batch_size)

noms_classes <- names(sort(unlist(train_generator$class_indices)))

calib <- choisir_seuil_archeo(
  proba_arch    = proba_calib_arch,
  espece_arch   = as.character(archeo_calib$taxon),
  proba_ood     = proba_calib_ood,
  noms_classes  = noms_classes,
  seuil_ood_min = seuil_ood_min_cible,
  grille        = grille_seuils)

cat("\n[CALIBRATION]\n")
cat("  Critere            :", calib$critere, "\n")
cat("  Contrainte OOD min :", calib$seuil_ood_min, "\n")
cat("  Seuil retenu       :", round(calib$seuil, 3), "\n")
cat("  Archeo correctes   :", calib$n_arch_correct, "/", calib$n_arch_total,
    sprintf("(%.1f%%)", 100*calib$taux_arch_correct), "\n")
cat("  Archeo rejetees    :", calib$n_arch_rejetees, "\n")
cat("  Taux OOD rejete    :", round(calib$taux_ood_rejete, 3), "\n")
cat("  Stabilite : le max de correctes (", calib$n_max_correct,
    ") est atteint pour seuil dans [",
    round(calib$etendue_opt_seuils[1], 3), ",",
    round(calib$etendue_opt_seuils[2], 3), "]\n")
if (diff(calib$etendue_opt_seuils) > 0.30) {
  cat("  ATTENTION : etendue > 0.30 -> choix du seuil peu contraint par\n")
  cat("  les archeo. La contrainte OOD fait l'essentiel du travail.\n")
}

auc_calib <- auc_id_ood(proba_calib_id, proba_calib_ood)
cat("  AUC calib ID vs OOD:", round(auc_calib, 3), "(diagnostic)\n")
if (!is.na(auc_calib) && auc_calib < 0.65) {
  cat("  AVERTISSEMENT : AUC faible ->")}
# --- predictions sur test ---------------------------------------------
cat("\nPredictions sur la partition external...\n")
proba_test_id   <- predire_sur_df(model, test_id_df)
proba_test_arch <- predire_sur_chemins(model, archeo_test$path,
                                        input_shape[1:2], batch_size)
proba_test_ood  <- predire_sur_chemins(model, ood_test_df$path,
                                        input_shape[1:2], batch_size)

gen_test     <- flux(test_id_df, melanger = FALSE)
n_test       <- nrow(test_id_df)
classe_vraie <- as.integer(gen_test$classes) + 1
idx_match    <- match(gen_test$filenames, test_id_df$path)
if (anyNA(idx_match)) idx_match <- seq_len(n_test)
espece_test      <- factor(noms_classes[classe_vraie], levels = noms_classes)
statut_test      <- factor(test_id_df$statut[idx_match], levels = unname(lib_statut))
espece_arch_test <- as.character(archeo_test$taxon)

eval_officiel <- evaluer_10x4(
  proba_id    = proba_test_id,   espece_id   = espece_test,
  statut_id   = statut_test,
  proba_arch  = proba_test_arch, espece_arch = espece_arch_test,
  proba_ood   = proba_test_ood,
  seuil       = calib$seuil,     noms_classes = noms_classes)

cat("\n[EVALUATION AU SEUIL CALIBRE =", round(calib$seuil, 3), "]\n")
cat("\nMatrice 10 (verite) x 4 (prediction) :\n")
print(eval_officiel$confusion)

cat("\nTaux espece correcte (sans tenir compte du rejet) :\n")
print(round(eval_officiel$taux_espece_brute, 3))
cat("\nTaux correct avec rejet applique :\n")
print(round(eval_officiel$taux_correct_final, 3))

cat("\nMetriques globales :\n")
cat("  Accuracy 4 classes                :", round(eval_officiel$acc_4, 3), "\n")
cat("  % ID acceptes                     :", round(eval_officiel$accepte_id, 3), "\n")
cat("  % OOD rejetes                     :", round(eval_officiel$rejete_ood, 3), "\n")
cat("  % archeo rejetees                 :", round(eval_officiel$rejete_arch, 3), "\n")
cat("  % archeo correctes (dans espece)  :",
    round(eval_officiel$taux_arch_correct, 3), "\n")

auc_test <- auc_id_ood(proba_test_id, proba_test_ood)
cat("\n[STABILITE AUC]\n")
cat("  AUC calib :", round(auc_calib, 3), "\n")
cat("  AUC test  :", round(auc_test,  3), "\n")

cat("\n[Grille exploratoire - TEST] (ne remplace pas le seuil calibre)\n")
grille_test <- lapply(c(0.33, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99), function(s) {
  e <- evaluer_10x4(proba_test_id, espece_test, statut_test,
                    proba_test_arch, espece_arch_test,
                    proba_test_ood, s, noms_classes)
  data.frame(seuil             = s,
             accepte_id        = e$accepte_id,
             rejete_ood        = e$rejete_ood,
             rejete_arch       = e$rejete_arch,
             taux_arch_correct = e$taux_arch_correct,
             acc_4             = e$acc_4)
})
print(bind_rows(grille_test), row.names = FALSE)

dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
fichier_h5 <- file.path(dossier_res,
                        sprintf("Test5_domwild_modele_ep%d_seed%d.h5",
                                epochs, seed))
tryCatch(save_model_hdf5(model, fichier_h5),
         error = function(e) message("Modele non sauvegarde : ",
                                     conditionMessage(e)))


# ---------------------------------------------------------------------
# 9. VISUALISATION
# ---------------------------------------------------------------------
df_conf <- bind_rows(
  data.frame(confiance = apply(proba_test_id,   1, max),
             groupe = "ID (dom + sauv)"),
  data.frame(confiance = apply(proba_test_arch, 1, max),
             groupe = "Archeo"),
  data.frame(confiance = apply(proba_test_ood,  1, max),
             groupe = "Autre (OOD)"))

print(
  ggplot(df_conf, aes(x = confiance, fill = groupe)) +
    geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
    geom_vline(xintercept = calib$seuil, linetype = "dashed") +
    labs(x = "p_max", y = "Nombre d'images",
         title = "Distribution de p_max sur external",
         subtitle = sprintf(
           "Seuil = %.3f | archeo correct=%.2f | OOD rejete=%.2f",
           calib$seuil, calib$taux_arch_correct, calib$taux_ood_rejete)) +
    theme_minimal())

print(
  ggplot(calib$courbe, aes(x = seuil)) +
    geom_line(aes(y = taux_arch_correct, color = "Archeo correct"), size = 1) +
    geom_line(aes(y = taux_ood_rejete,   color = "OOD rejete"),    size = 1) +
    geom_vline(xintercept = calib$seuil, linetype = "dashed") +
    geom_hline(yintercept = calib$seuil_ood_min, linetype = "dotted") +
    labs(x = "Seuil", y = "Taux", color = "",
         title = "Courbe de calibration",
         subtitle = "Trait plein = seuil retenu | pointille = contrainte OOD") +
    theme_minimal())


# ---------------------------------------------------------------------
# 10. SAUVEGARDE
# ---------------------------------------------------------------------
nom_base <- sprintf("Test5_domwild_N%d_ep%d_seed%d",
                    N_train_par_statut, epochs, seed)
resultats <- list(
  master_df       = master_df,
  archeo_calib    = archeo_calib, archeo_test = archeo_test,
  autre_calib     = autre_calib,  autre_test  = autre_test,
  ood_calib_df    = ood_calib_df, ood_test_df = ood_test_df,
  autre_samp      = autre_samp,
  historique      = as.data.frame(history$metrics),
  calib           = calib,
  eval_officiel   = eval_officiel,
  auc_calib       = auc_calib,
  auc_test        = auc_test,
  noms_classes    = noms_classes,
  proba_test_id   = proba_test_id,
  proba_test_arch = proba_test_arch,
  proba_test_ood  = proba_test_ood,
  config = list(
    taxons             = taxons,
    fichier_archeo     = fichier_archeo,
    fichier_autre      = fichier_autre,
    N_train_par_statut = N_train_par_statut,
    N_test_par_statut  = N_test_par_statut,
    N_ext_par_statut   = N_ext_par_statut,
    N_archeo_calib_par_espece = N_archeo_calib_par_espece,
    N_archeo_ext_par_espece   = N_archeo_ext_par_espece,
    N_autre_ext        = N_autre_ext,
    seuil_ood_min_cible = seuil_ood_min_cible,
    epochs = epochs, seed = seed, seed_wild = seed_wild,
    seed_autre = seed_autre, seed_arch = seed_arch,
    col_image = col_image),
  duree = Sys.time() - begin_time)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))

cat("\nResultats ecrits dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")

