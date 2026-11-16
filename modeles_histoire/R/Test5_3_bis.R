# =====================================================================
# Test5.R
# Classe de rejet "autre" via seuil de confiance sur p_max,
# avec partition de CALIBRATION dediee et critere de Youden (AUC).
#
# Partitions par espece/statut :
#   train    -> entrainement CNN
#   test     -> validation CNN (early stopping)
#   calib    -> choix du seuil (ID + Hordeum)
#   external -> evaluation finale (ID + Hordeum), jamais vu avant
#
# Le seuil est choisi par Youden's J = TPR + TNR - 1 (aucun a priori
# sur les couts). L'AUC estime si la separation est structurellement
# possible (AUC proche de 0.5 -> seuil inutile).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db  <- "rda/db"
dossier_res <- "rda/results/especes_avec_rejet"

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")
dossier_hordeum <- "DATA/HORDEUM"
col_image <- "VD_deep"

statut_domestique <- "c"
statut_sauvage    <- "w"

# Chaque paramètre est PAR ESPECE (3 especes).
# Phoenix 'c' = 650 : N_dom_model + N_dom_ext = 650, OK pile.
# Olea   'w' = 630 : N_wild_model + N_wild_ext = 630, OK pile.
N_dom_model   <- 325    # 325 x 3 = 975 images d'entrainement domestiques
N_dom_ext     <- 325    # 325 x 3 = 975 images externes domestiques
#   -> 650 Phoix Max atteint, ok

N_wild_model  <- 315    # 315 x 3 = 945
N_wild_ext    <- 315    # 315 x 3 = 945
#   -> 630 Olea Max atteint, ok

N_hordeum_ext <- 650 

prop_calib    <- 0.5 # 50% des "external" servent a la calibration

prop_train  <- 4/5
epochs      <- 5
seed        <- 2026
seed_wild   <- seed + 1
seed_hord   <- seed + 2
batch_size  <- 32
input_shape <- c(120, 90, 3)

# Grille de seuils pour la comparaison exploratoire (pas de choix).
seuils_a_tester <- c(0.33, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


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
# 2. FONCTIONS DE CHARGEMENT (identiques a Test5.R original)
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

charger_hordeum <- function(taxon, dossier_img) {
  cat("\n===== HORDEUM (classe 'autre') =====\n")
  fichiers <- list.files(dossier_db, pattern = taxon,
                         full.names = TRUE, ignore.case = TRUE)
  if (length(fichiers) == 0) stop("Aucun .rda '", taxon, "' dans ", dossier_db)
  prefere <- fichiers[tolower(basename(fichiers)) == paste0(taxon, "_db2.rda")]
  fichier <- if (length(prefere) > 0) prefere[1] else fichiers[1]
  env_tmp <- new.env(); noms <- load(fichier, envir = env_tmp)
  db <- get(noms[1], envir = env_tmp)
  chemins <- as.character(db[[col_image]])
  if (!all(file.exists(chemins))) {
    tous <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    idx  <- match(basename(chemins), basename(tous))
    if (any(is.na(idx))) stop(sum(is.na(idx)), " images introuvables")
    chemins <- tous[idx]
  }
  tibble(taxon = taxon, path = chemins)
}


# ---------------------------------------------------------------------
# 3. CHARGEMENT
# ---------------------------------------------------------------------
tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]])))
print(table(tous_les_taxons$taxon, tous_les_taxons$statut))

hordeum_all <- charger_hordeum("hordeum", dossier_hordeum)
cat("Nombre total de Hordeum disponibles :", nrow(hordeum_all), "\n")


# ---------------------------------------------------------------------
# 4. TIRAGE ET PARTITION (4 partitions : train / test / calib / external)
# ---------------------------------------------------------------------
# --- 4.1 domestiques -------------------------------------------------
N_train_d <- round(N_dom_model * prop_train)
N_test_d  <- N_dom_model - N_train_d
N_calib_d <- round(N_dom_ext  * prop_calib)
N_exter_d <- N_dom_ext - N_calib_d
labs_dom  <- c(rep("train",    N_train_d),
               rep("test",     N_test_d),
               rep("calib",    N_calib_d),
               rep("external", N_exter_d))
set.seed(seed)
shuffled_dom <- sample(labs_dom)
dom <- tous_les_taxons %>% filter(statut == statut_domestique)
if (any(table(dom$taxon) < length(labs_dom)))
  stop("Pas assez de graines domestiques par taxon (besoin ",
       length(labs_dom), ")")
df_dom <- dom %>% group_by(taxon) %>% sample_n(length(labs_dom)) %>%
  mutate(partition = shuffled_dom) %>% ungroup()

# --- 4.2 sauvages ----------------------------------------------------
N_train_w <- round(N_wild_model * prop_train)
N_test_w  <- N_wild_model - N_train_w
N_calib_w <- round(N_wild_ext  * prop_calib)
N_exter_w <- N_wild_ext - N_calib_w
labs_wild <- c(rep("train",    N_train_w),
               rep("test",     N_test_w),
               rep("calib",    N_calib_w),
               rep("external", N_exter_w))
set.seed(seed_wild)
shuffled_wild <- sample(labs_wild)
wild <- tous_les_taxons %>% filter(statut == statut_sauvage)
if (any(table(wild$taxon) < length(labs_wild)))
  stop("Pas assez de graines sauvages par taxon")
df_wild <- wild %>% group_by(taxon) %>% sample_n(length(labs_wild)) %>%
  mutate(partition = shuffled_wild) %>% ungroup()

# --- 4.3 tableau final -----------------------------------------------
lib_statut <- setNames(c("domestique", "sauvage"),
                       c(statut_domestique, statut_sauvage))
master_df <- bind_rows(df_dom, df_wild) %>%
  transmute(partition, class = taxon,
            statut = unname(lib_statut[statut]), path)

cat("\nPartition x espece x statut :\n")
print(ftable(table(partition = master_df$partition,
                   espece    = master_df$class,
                   statut    = master_df$statut)))

# Modeles : train + test (validation CNN)
jeux <- list(
  domestiques = list(
    train = master_df %>% filter(partition == "train", statut == "domestique"),
    val   = master_df %>% filter(partition == "test",  statut == "domestique")),
  sauvages = list(
    train = master_df %>% filter(partition == "train", statut == "sauvage"),
    val   = master_df %>% filter(partition == "test",  statut == "sauvage")))

# ID a evaluer : calib (choix seuil) et external (test final)
calib_id_df <- master_df %>% filter(partition == "calib")
test_id_df  <- master_df %>% filter(partition == "external")

# --- 4.4 tirage Hordeum ----------------------------------------------
N_hord_total <- min(N_hordeum_ext, nrow(hordeum_all))
set.seed(seed_hord)
hordeum_samp <- hordeum_all %>% sample_n(N_hord_total)
N_hord_calib <- round(N_hord_total * prop_calib)
N_hord_test  <- N_hord_total - N_hord_calib
hordeum_samp$partition <- sample(c(rep("calib", N_hord_calib),
                                    rep("external", N_hord_test)))
hordeum_calib <- hordeum_samp %>% filter(partition == "calib")
hordeum_test  <- hordeum_samp %>% filter(partition == "external")

cat("\nTailles :\n")
cat("  ID calib   :", nrow(calib_id_df), " | ID test   :", nrow(test_id_df), "\n")
cat("  OOD calib  :", nrow(hordeum_calib), " | OOD test  :", nrow(hordeum_test), "\n")


# ---------------------------------------------------------------------
# 5. FONCTIONS COMMUNES
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
  arr <- array(0, dim = c(n, H, W, 3))
  for (i in seq_len(n)) {
    img <- image_load(chemins[i], target_size = target_size, color_mode = "rgb")
    arr[i,,,] <- image_to_array(img) / 255
  }
  predict(model, arr, batch_size = batch_size, verbose = 0)
}

# Predictions sur un dataframe d'ID ; l'ordre est preserve
# car shuffle = FALSE ; on remet dans l'ordre du df par securite.
predire_sur_df <- function(model, df) {
  gen   <- flux(df, melanger = FALSE)
  n     <- nrow(df)
  steps <- ceiling(n / batch_size)
  proba <- predict(model, gen, steps = steps, verbose = 0)
  if (nrow(proba) > n) proba <- proba[seq_len(n), , drop = FALSE]
  # verification coherence par nom de fichier
  idx <- match(gen$filenames, df$path)
  if (!anyNA(idx) && nrow(proba) == n) proba <- proba[idx, , drop = FALSE]
  proba
}


# ---------------------------------------------------------------------
# 6. CHOIX DU SEUIL PAR YOUDEN (sur la partition calib)
# ---------------------------------------------------------------------
# Score d'anomalie = 1 - p_max. Le rejet a lieu si p_max < seuil,
# i.e. score > 1 - seuil. Sensibilite = P(rejet | OOD) = P(p_max < s | OOD).
# Specificite = P(accepte | ID) = P(p_max >= s | ID).
choisir_seuil_youden <- function(proba_id, proba_ood, resolution = 1001) {
  conf_id  <- apply(proba_id,  1, max)
  conf_ood <- apply(proba_ood, 1, max)

  # AUC via Mann-Whitney
  scores <- c(1 - conf_id, 1 - conf_ood)
  labels <- c(rep(0L, length(conf_id)), rep(1L, length(conf_ood)))
  r  <- rank(scores)
  n1 <- sum(labels == 1L); n0 <- sum(labels == 0L)
  auc <- (sum(r[labels == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)

  # Courbe ROC sur une grille fine de seuils p_max
  seuils <- seq(0, 1, length.out = resolution)
  tpr <- sapply(seuils, function(s) mean(conf_ood <  s))   # sensibilite
  tnr <- sapply(seuils, function(s) mean(conf_id  >= s))   # specificite
  fpr <- 1 - tnr
  J   <- tpr + tnr - 1
  best_idx <- which.max(J)

  list(auc        = auc,
       seuil      = seuils[best_idx],
       J          = J[best_idx],
       tpr        = tpr[best_idx],
       tnr        = tnr[best_idx],
       courbe     = data.frame(seuil = seuils, tpr = tpr, tnr = tnr,
                               fpr = fpr, J = J),
       conf_id    = conf_id, conf_ood = conf_ood)
}


# ---------------------------------------------------------------------
# 7. EVALUATION AVEC SEUIL
# ---------------------------------------------------------------------
# Matrice de confusion : vraie classe detaillee (espece | statut) x
# espece predite. Structure extensible : pour ajouter l'archeo, il
# suffira d'ajouter "espece | archeo" dans niveaux_vraie et de fournir
# statut_ext = "archeo" pour ces images.
evaluer_avec_rejet <- function(proba_ext, espece_ext, statut_ext,
                               proba_hord, seuil, noms_classes) {
  
  # --- predictions par seuil
  conf_ext    <- apply(proba_ext, 1, max)
  pred_ext    <- max.col(proba_ext)
  classe_ext  <- ifelse(conf_ext < seuil, "autre", noms_classes[pred_ext])
  
  conf_hord   <- apply(proba_hord, 1, max)
  classe_hord <- ifelse(conf_hord < seuil, "autre",
                        noms_classes[max.col(proba_hord)])
  
  niveaux_pred <- c(noms_classes, "autre")
  
  # --- vraie etiquette detaillee des ID : "espece | statut"
  vraie_id_label <- paste(as.character(espece_ext),
                          as.character(statut_ext), sep = " | ")
  
  # --- niveaux de vraie classe (ordre stable, extensible)
  #     ordre : pour chaque espece, domestique puis sauvage, puis "autre"
  niveaux_vraie <- as.vector(sapply(noms_classes, function(e) c(
    paste(e, "domestique", sep = " | "),
    paste(e, "sauvage",    sep = " | ")
  )))
  niveaux_vraie <- c(niveaux_vraie, "autre")
  # -> pour l'archeo : c(niveaux_vraie_precedent,
  #                      as.vector(sapply(noms_classes, function(e)
  #                        paste(e, "archeo", sep = " | "))),
  #                      "autre")
  
  # --- construction des facteurs
  vraie_combinee   <- factor(c(vraie_id_label,
                               rep("autre", length(classe_hord))),
                             levels = niveaux_vraie)
  predite_combinee <- factor(c(classe_ext, classe_hord),
                             levels = niveaux_pred)
  
  # --- matrice 7x4 et taux ligne a ligne (proportions, pas le "bon" taux)
  conf_espece_statut <- table(vraie   = vraie_combinee,
                              predite = predite_combinee)
  taux_par_ligne     <- prop.table(conf_espece_statut, margin = 1)
  
  # --- taux de bonne classification par ligne (espece | statut)
  taux_correct_par_ligne <- sapply(niveaux_vraie, function(lbl) {
    if (lbl == "autre") {
      mean(classe_hord == "autre")
    } else {
      parts <- strsplit(lbl, " | ", fixed = TRUE)[[1]]
      esp   <- parts[1]; stat <- parts[2]
      idx   <- which(as.character(espece_ext) == esp &
                       as.character(statut_ext) == stat)
      if (length(idx) == 0) NA_real_ else mean(classe_ext[idx] == esp)
    }
  })
  
  # --- metriques globales (inchangees)
  accepte_id <- mean(classe_ext != "autre")
  correct_id <- mean(classe_ext == as.character(espece_ext))
  rejete_ood <- mean(classe_hord == "autre")
  acc_4      <- (sum(classe_ext == as.character(espece_ext)) +
                   sum(classe_hord == "autre")) /
    (length(classe_ext) + length(classe_hord))
  
  scores_anomalie <- c(1 - conf_ext, 1 - conf_hord)
  labels_ood      <- c(rep(0, length(conf_ext)), rep(1, length(conf_hord)))
  r  <- rank(scores_anomalie)
  n1 <- sum(labels_ood == 1); n0 <- sum(labels_ood == 0)
  auc_ood <- (sum(r[labels_ood == 1]) - n1*(n1+1)/2) / (n1*n0)
  
  tp <- sum(predite_combinee == "autre" & vraie_combinee == "autre")
  fp <- sum(predite_combinee == "autre" & vraie_combinee != "autre")
  fn <- sum(predite_combinee != "autre" & vraie_combinee == "autre")
  prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  rec  <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0)
    2 * prec * rec / (prec + rec) else NA_real_
  
  acc_par_statut <- tapply(classe_ext == as.character(espece_ext),
                           statut_ext, mean)
  
  list(seuil                = seuil,
       conf_espece_statut   = conf_espece_statut,
       taux_par_ligne       = taux_par_ligne,
       taux_correct_par_ligne = taux_correct_par_ligne,
       acc_4                = acc_4,
       accepte_id           = accepte_id,
       correct_id           = correct_id,
       rejete_ood           = rejete_ood,
       auc_ood              = auc_ood,
       precision_autre      = prec,
       rappel_autre         = rec,
       f1_autre             = f1,
       acc_par_statut       = acc_par_statut,
       classe_ext           = classe_ext,
       classe_hord          = classe_hord,
       conf_ext             = conf_ext,
       conf_hord            = conf_hord)
}

# ---------------------------------------------------------------------
# 8. ENTRAINEMENT + CALIBRATION + EVALUATION
# ---------------------------------------------------------------------
entrainer_calibrer_evaluer <- function(nom, train_df, val_df,
                                       calib_id_df, test_id_df,
                                       calib_hord_df, test_hord_df,
                                       seuils) {
  cat("\n#####################################################\n")
  cat("MODELE :", nom, "\n")
  cat("#####################################################\n")

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

  # --- predictions sur calib ------------------------------------------
  cat("Predictions sur la partition calib...\n")
  proba_calib_id   <- predire_sur_df(model, calib_id_df)
  proba_calib_hord <- predire_sur_chemins(model, calib_hord_df$path,
                                          input_shape[1:2], batch_size)

  # --- calibration : AUC + seuil de Youden
  calib <- choisir_seuil_youden(proba_calib_id, proba_calib_hord)
  cat("\n[CALIBRATION -", nom, "]\n")
  cat("  AUC (calib)          :", round(calib$auc, 3), "\n")
  cat("  Seuil de Youden      :", round(calib$seuil, 3), "\n")
  cat("  J max                :", round(calib$J, 3),
      "(TPR =", round(calib$tpr, 3), ", TNR =", round(calib$tnr, 3), ")\n")
  if (calib$auc < 0.65) {
    cat("  AVERTISSEMENT : AUC faible -> le seuil de confiance ne separe\n")
    cat("  pas bien ID et OOD sur cette partition. Le seuil retenu sera\n")
    cat("  peu informatif, independamment de sa valeur.\n")
  }

  # --- predictions sur test -------------------------------------------
  cat("\nPredictions sur la partition test...\n")
  proba_test_id   <- predire_sur_df(model, test_id_df)
  proba_test_hord <- predire_sur_chemins(model, test_hord_df$path,
                                         input_shape[1:2], batch_size)

  # recuperation des etiquettes ID test
  gen_test <- flux(test_id_df, melanger = FALSE)
  n_test   <- nrow(test_id_df)
  classe_vraie <- as.integer(gen_test$classes) + 1
  noms_classes <- names(sort(unlist(gen_test$class_indices)))
  idx <- match(gen_test$filenames, test_id_df$path)
  if (anyNA(idx)) idx <- seq_len(n_test)
  espece_test <- factor(noms_classes[classe_vraie], levels = noms_classes)
  statut_test <- factor(test_id_df$statut[idx], levels = unname(lib_statut))

  # --- evaluation au seuil calibre (RESULTAT OFFICIEL) ----------------
  eval_officiel <- evaluer_avec_rejet(
    proba_test_id, espece_test, statut_test,
    proba_test_hord, calib$seuil, noms_classes)

  cat("\n[TEST - seuil calibre =", round(calib$seuil, 3), "]\n")
  cat("\nMatrice : vraie espece|statut  x  espece predite\n")
  print(eval_officiel$conf_espece_statut)
  cat("\nTaux de bonne classification par ligne (espece|statut) :\n")
  print(round(eval_officiel$taux_correct_par_ligne, 3))
  cat("\nTaux de bonne classification par statut (toutes especes confondues) :\n")
  print(round(eval_officiel$acc_par_statut, 3))
  cat("\nAccuracy 4 classes        :", round(eval_officiel$acc_4, 3), "\n")
  cat("% ID acceptes            :", round(eval_officiel$accepte_id, 3), "\n")
  cat("% ID correctement classes:", round(eval_officiel$correct_id, 3), "\n")
  cat("% OOD rejetes            :", round(eval_officiel$rejete_ood, 3), "\n")
  cat("F1 'autre'               :", round(eval_officiel$f1_autre, 3), "\n")

  # --- verification stabilite AUC (test vs calib)
  eval_baseline <- evaluer_avec_rejet(
    proba_test_id, espece_test, statut_test,
    proba_test_hord, 0, noms_classes)
  cat("\n[STABILITE AUC]\n")
  cat("  AUC calib :", round(calib$auc,           3), "\n")
  cat("  AUC test  :", round(eval_baseline$auc_ood, 3), "\n")
  cat("  (les deux doivent etre proches ; un ecart > 0.10 signale une\n")
  cat("   calibration instable ou une partition trop petite)\n")

  # --- grille exploratoire sur test (pour comparaison, pas pour choisir)
  evals_test <- lapply(seuils, function(s)
    evaluer_avec_rejet(proba_test_id, espece_test, statut_test,
                       proba_test_hord, s, noms_classes))
  tab_seuils <- data.frame(
    seuil      = sapply(evals_test, `[[`, "seuil"),
    accepte_id = sapply(evals_test, `[[`, "accepte_id"),
    correct_id = sapply(evals_test, `[[`, "correct_id"),
    rejete_ood = sapply(evals_test, `[[`, "rejete_ood"),
    acc_4      = sapply(evals_test, `[[`, "acc_4"),
    f1_autre   = sapply(evals_test, `[[`, "f1_autre"))
  cat("\n[Grille exploratoire - TEST] (ne remplace pas le seuil calibre)\n")
  print(tab_seuils, row.names = FALSE)

  # --- sauvegarde du modele
  dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
  fichier_h5 <- file.path(dossier_res,
                          sprintf("Test5_modele_%s_ep%d_seed%d.h5",
                                  nom, epochs, seed))
  tryCatch(save_model_hdf5(model, fichier_h5),
           error = function(e) message("Modele non sauvegarde : ",
                                        conditionMessage(e)))

  list(historique = as.data.frame(history$metrics),
       calib = calib,
       eval_officiel = eval_officiel,
       eval_baseline = eval_baseline,
       tab_seuils = tab_seuils,
       noms_classes = noms_classes,
       proba_test_id = proba_test_id, proba_test_hord = proba_test_hord,
       espece_test = espece_test, statut_test = statut_test)
}


# ---------------------------------------------------------------------
# 9. ENTRAINEMENT DES DEUX MODELES
# ---------------------------------------------------------------------
res_dom  <- entrainer_calibrer_evaluer("domestiques",
                                       jeux$domestiques$train,
                                       jeux$domestiques$val,
                                       calib_id_df, test_id_df,
                                       hordeum_calib, hordeum_test,
                                       seuils_a_tester)
res_wild <- entrainer_calibrer_evaluer("sauvages",
                                       jeux$sauvages$train,
                                       jeux$sauvages$val,
                                       calib_id_df, test_id_df,
                                       hordeum_calib, hordeum_test,
                                       seuils_a_tester)


# ---------------------------------------------------------------------
# 10. COMPARAISON
# ---------------------------------------------------------------------
cat("\n#####################################################\n")
cat("COMPARAISON DES MODELES (resultats sur test)\n")
cat("#####################################################\n")

synthese <- data.frame(
  modele             = c("domestiques", "sauvages"),
  AUC_calib          = c(res_dom$calib$auc,           res_wild$calib$auc),
  seuil_youden       = c(res_dom$calib$seuil,          res_wild$calib$seuil),
  AUC_test           = c(res_dom$eval_baseline$auc_ood,
                          res_wild$eval_baseline$auc_ood),
  accepte_id_test    = c(res_dom$eval_officiel$accepte_id,
                          res_wild$eval_officiel$accepte_id),
  rejete_ood_test    = c(res_dom$eval_officiel$rejete_ood,
                          res_wild$eval_officiel$rejete_ood),
  acc_4_test         = c(res_dom$eval_officiel$acc_4,
                          res_wild$eval_officiel$acc_4),
  f1_autre_test      = c(res_dom$eval_officiel$f1_autre,
                          res_wild$eval_officiel$f1_autre))
print(synthese, row.names = FALSE)


# ---------------------------------------------------------------------
# 11. VISUALISATION : histogrammes + courbes ROC
# ---------------------------------------------------------------------
df_conf <- bind_rows(
  data.frame(confiance = res_dom$eval_baseline$conf_ext,
             population = "ID (test)",      modele = "domestiques"),
  data.frame(confiance = res_dom$eval_baseline$conf_hord,
             population = "Hordeum (OOD)",  modele = "domestiques"),
  data.frame(confiance = res_wild$eval_baseline$conf_ext,
             population = "ID (test)",      modele = "sauvages"),
  data.frame(confiance = res_wild$eval_baseline$conf_hord,
             population = "Hordeum (OOD)",  modele = "sauvages"))

print(
  ggplot(df_conf, aes(x = confiance, fill = population)) +
    geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
    facet_wrap(~ modele) +
    geom_vline(data = data.frame(
                 modele = c("domestiques", "sauvages"),
                 seuil  = c(res_dom$calib$seuil, res_wild$calib$seuil)),
               aes(xintercept = seuil),
               linetype = "dashed", color = "black") +
    labs(x = "p_max", y = "Nombre d'images",
         title = "Distribution de p_max sur le jeu de test",
         subtitle = "Trait noir = seuil de Youden calibre sur calib") +
    theme_minimal())

df_roc <- bind_rows(
  data.frame(res_dom$calib$courbe,  modele = "domestiques",
             partition = "calib"),
  data.frame(res_wild$calib$courbe, modele = "sauvages",
             partition = "calib"))

# point Youden
df_youden <- data.frame(
  modele = c("domestiques", "sauvages"),
  fpr    = c(1 - res_dom$calib$tnr, 1 - res_wild$calib$tnr),
  tpr    = c(res_dom$calib$tpr,     res_wild$calib$tpr))

print(
  ggplot(df_roc, aes(x = fpr, y = tpr, color = modele)) +
    geom_line(size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
    geom_point(data = df_youden, aes(x = fpr, y = tpr, color = modele),
               size = 4, shape = 18) +
    labs(x = "FPR (1 - specificite)", y = "TPR (sensibilite OOD)",
         title = "Courbe ROC sur la partition calib",
         subtitle = "Losange = point de Youden retenu") +
    theme_minimal())


# ---------------------------------------------------------------------
# 12. SAUVEGARDE
# ---------------------------------------------------------------------
nom_base <- sprintf("Test5_avec_rejet_N%d_ep%d_seed%d",
                    N_dom_model, epochs, seed)
resultats <- list(
  master_df = master_df, hordeum_samp = hordeum_samp,
  calib_id_df = calib_id_df, test_id_df = test_id_df,
  hordeum_calib = hordeum_calib, hordeum_test = hordeum_test,
  res_dom = res_dom, res_wild = res_wild,
  synthese = synthese, prop_calib = prop_calib,
  config = list(taxons = taxons, dossier_hordeum = dossier_hordeum,
                N_dom_model = N_dom_model, N_dom_ext = N_dom_ext,
                N_wild_model = N_wild_model, N_wild_ext = N_wild_ext,
                N_hordeum_ext = N_hordeum_ext, epochs = epochs,
                seed = seed, seed_wild = seed_wild, seed_hord = seed_hord,
                col_image = col_image),
  duree = Sys.time() - begin_time)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))

cat("\nResultats ecrits dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")



