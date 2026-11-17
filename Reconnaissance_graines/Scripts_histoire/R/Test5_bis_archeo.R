# =====================================================================
# Test5_bis.R
#
# Variante de Test5.R pour MAXIMISER la bonne classification des
# graines ARCHEOBOTANIQUES, sur un jeu tres reduit (~23 images).
#
# Differences avec Test5.R :
#   - Proxy OOD : "autres especes" (modernes) au lieu de Hordeum.
#   - Classe positive de calibration : les graines ARCHEO (au lieu
#     des modernes ID).
#   - Critere de calibration : Youden modifie
#         J(s) = P(autres rejetes | s)
#              + P(archeo corrects ET acceptes | s) - 1
#     -> on ne recompense pas l'acceptation d'archeo mal classes.
#   - Bootstrap (500 reechantillonnages) pour un IC 95% sur le seuil.
#   - Avertissement si l'accuracy argmax sur archeo (plafond du taux
#     d'archeo correct) est trop basse : le seuil ne peut pas reparer
#     un modele qui ne generalise pas.
#
# Structure .rda conservee (compatible Eval_Modele.R) :
#   res_dom$calib$seuil, res_wild$calib$seuil,
#   master_df (modernes), archeo_samp (archeo), autres_samp (autres)
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db  <- "rda/db"
dossier_ajout <- "rda/db_ajout"
dossier_res <- "rda/results/especes_avec_rejet_bis"

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")
dossier_hordeum <- "DATA/HORDEUM"   # non utilise par defaut (backup)
col_image <- "VD_deep"

statut_domestique <- "c"
statut_sauvage    <- "w"

# Effectifs modernes : laisses moderés puisque le focus est archeo.
N_dom_model   <- 325
N_dom_ext     <- 325    # decoupe calib/external pour la mesure modern
N_wild_model  <- 315
N_wild_ext    <- 315
prop_calib_modern <- 0.5

# Effectifs archeo/autres (issus des .rda Creer_rda_archeo.R)
# -- archeo : on prend TOUT (23 images, pas de split)
# -- autres : on decoupe en calib / external
prop_calib_autres <- 0.5

# Partition pour la partie modern (train / val CNN)
prop_train  <- 4/5
epochs      <- 10
seed        <- 2027
seed_wild   <- seed + 1
seed_hord   <- seed + 2
seed_autres <- seed + 3
seed_boot   <- seed + 4
batch_size  <- 32
input_shape <- c(120, 90, 3)

# Nombre de bootstraps pour l'IC sur le seuil (23 archeo : tres petit)
n_boot <- 500

# Grille fine de seuils explores (utilisee pour la courbe et le bootstrap)
resolution_grille <- 1001


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
# 2. FONCTIONS DE CHARGEMENT
# ---------------------------------------------------------------------
charger_taxon <- function(taxon, dossier_img) {
  cat("\n=====", toupper(taxon), "=====\n")
  fichiers <- list.files(dossier_db, pattern = taxon,
                         full.names = TRUE, ignore.case = TRUE)
  if (length(fichiers) == 0) stop("Aucun .rda '", taxon, "' dans ", dossier_db)
  prefere <- fichiers[tolower(basename(fichiers)) == paste0(taxon, "_db2.rda")]
  fichier <- if (length(prefere) > 0) prefere[1] else fichiers[1]
  env_tmp <- new.env(); noms <- load(fichier, envir = env_tmp)
  db <- get(noms[1], envir = env_tmp)
  if (!all(c("status", col_image) %in% names(db)))
    stop("Colonnes manquantes pour ", taxon)
  db <- db[as.character(db$status) %in% c(statut_domestique, statut_sauvage), ]
  chemins <- as.character(db[[col_image]])
  if (!all(file.exists(chemins))) {
    tous <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    idx  <- match(basename(chemins), basename(tous))
    if (any(is.na(idx))) stop(sum(is.na(idx)), " images introuvables pour ", taxon)
    chemins <- tous[idx]
  }
  tibble(taxon = taxon, statut = as.character(db$status), path = chemins)
}

charger_rda_ajout <- function(fichier) {
  if (!file.exists(fichier)) stop("Fichier .rda absent : ", fichier,
                                  " (lancer Creer_rda_archeo.R).")
  env <- new.env(); noms <- load(fichier, envir = env)
  obj <- get(noms[1], envir = env)
  if (!is.list(obj) || is.null(obj$master_df))
    stop("Pas de 'master_df' dans ", fichier)
  as_tibble(obj$master_df)
}


# ---------------------------------------------------------------------
# 3. CHARGEMENT DES DONNEES
# ---------------------------------------------------------------------
# --- 3.1 modernes (dom / sauv) ----------------------------------------
tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]])))
cat("\nGraines modernes disponibles :\n")
print(table(tous_les_taxons$taxon, tous_les_taxons$statut))

# --- 3.2 archeo -------------------------------------------------------
archeo_df <- bind_rows(
  charger_rda_ajout(file.path(dossier_ajout, "archeo_olea_db.rda")),
  charger_rda_ajout(file.path(dossier_ajout, "archeo_phoenix_db.rda")),
  charger_rda_ajout(file.path(dossier_ajout, "archeo_vitis_db.rda")))
cat("\nArcheo disponibles :\n")
print(table(archeo_df$class, archeo_df$statut, useNA = "ifany"))
if (nrow(archeo_df) == 0) stop("Aucune image archeo trouvee.")

# --- 3.3 autres especes ----------------------------------------------
autres_df <- charger_rda_ajout(file.path(dossier_ajout, "autres_especes_db.rda"))
cat("\nAutres especes disponibles :\n")
print(table(autres_df$class, autres_df$statut, useNA = "ifany"))
if (nrow(autres_df) == 0) stop("Aucune image 'autres' trouvee.")


# ---------------------------------------------------------------------
# 4. PARTITIONS
# ---------------------------------------------------------------------
# --- 4.1 modernes : train / test / calib / external (comme Test5) -----
N_train_d <- round(N_dom_model * prop_train)
N_test_d  <- N_dom_model - N_train_d
N_calib_d <- round(N_dom_ext  * prop_calib_modern)
N_exter_d <- N_dom_ext - N_calib_d
labs_dom  <- c(rep("train",    N_train_d),
               rep("test",     N_test_d),
               rep("calib",    N_calib_d),
               rep("external", N_exter_d))
set.seed(seed)
shuffled_dom <- sample(labs_dom)
dom <- tous_les_taxons %>% filter(statut == statut_domestique)
if (any(table(dom$taxon) < length(labs_dom)))
  stop("Pas assez de graines domestiques par taxon.")
df_dom <- dom %>% group_by(taxon) %>% sample_n(length(labs_dom)) %>%
  mutate(partition = shuffled_dom) %>% ungroup()

N_train_w <- round(N_wild_model * prop_train)
N_test_w  <- N_wild_model - N_train_w
N_calib_w <- round(N_wild_ext  * prop_calib_modern)
N_exter_w <- N_wild_ext - N_calib_w
labs_wild <- c(rep("train",    N_train_w),
               rep("test",     N_test_w),
               rep("calib",    N_calib_w),
               rep("external", N_exter_w))
set.seed(seed_wild)
shuffled_wild <- sample(labs_wild)
wild <- tous_les_taxons %>% filter(statut == statut_sauvage)
if (any(table(wild$taxon) < length(labs_wild)))
  stop("Pas assez de graines sauvages par taxon.")
df_wild <- wild %>% group_by(taxon) %>% sample_n(length(labs_wild)) %>%
  mutate(partition = shuffled_wild) %>% ungroup()

lib_statut <- setNames(c("domestique", "sauvage"),
                       c(statut_domestique, statut_sauvage))
master_df <- bind_rows(df_dom, df_wild) %>%
  transmute(partition, class = taxon,
            statut = unname(lib_statut[statut]), path)

jeux <- list(
  domestiques = list(
    train = master_df %>% filter(partition == "train", statut == "domestique"),
    val   = master_df %>% filter(partition == "test",  statut == "domestique")),
  sauvages = list(
    train = master_df %>% filter(partition == "train", statut == "sauvage"),
    val   = master_df %>% filter(partition == "test",  statut == "sauvage")))

calib_id_df <- master_df %>% filter(partition == "calib")
test_id_df  <- master_df %>% filter(partition == "external")

# --- 4.2 archeo : tout en calibration (pas de test separe) -----------
# Choix assume : 23 images sont trop peu pour un split calib/test.
# Le taux d'archeo correct mesure sur calib EST le resultat archeo.
archeo_calib <- archeo_df %>% mutate(partition = "calib")
archeo_test  <- archeo_df %>% filter(FALSE)   # vide, par clarte

# --- 4.3 autres : split calib / external -----------------------------
set.seed(seed_autres)
autres_df <- autres_df %>% mutate(
  partition = sample(c(rep("calib",    round(n() * prop_calib_autres)),
                       rep("external", n() - round(n() * prop_calib_autres)))))
autres_calib <- autres_df %>% filter(partition == "calib")
autres_test  <- autres_df %>% filter(partition == "external")

cat("\n=== Tailles ===")
cat("\nModernes :")
cat("\n  train (par statut) :", nrow(jeux$domestiques$train),
    " /", nrow(jeux$sauvages$train))
cat("\n  val   (par statut) :", nrow(jeux$domestiques$val),
    " /", nrow(jeux$sauvages$val))
cat("\n  calib (par statut) :", nrow(calib_id_df %>% filter(statut=="domestique")),
    " /", nrow(calib_id_df %>% filter(statut=="sauvage")))
cat("\n  test  (par statut) :", nrow(test_id_df %>% filter(statut=="domestique")),
    " /", nrow(test_id_df %>% filter(statut=="sauvage")))
cat("\nArcheo :", nrow(archeo_calib), " (toutes en calib)")
cat("\nAutres :", nrow(autres_calib), " calib +", nrow(autres_test), " test\n")


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


# ---------------------------------------------------------------------
# 6. CALIBRATION DU SEUIL SUR ARCHEO vs AUTRES
# ---------------------------------------------------------------------
# Critere : J(s) = P(autres rejetes | s) + P(archeo corrects ET acceptes | s) - 1
#
# Renvoie, pour un seuil s :
#   - tpr_autres     : P(p_max < s | autres)             (rejet correct)
#   - tnr_archeo     : P(p_max >= s ET argmax correct)   (archeo correct accepte)
#   - J              : tpr_autres + tnr_archeo - 1
#   - archeo_argmax  : P(argmax correct) - PLAFOND du taux d'archeo correct
#   - archeo_accepte : P(p_max >= s) - pour info
evaluer_seuil_archeo <- function(conf_archeo, is_correct_argmax,
                                  conf_autres, seuil) {
  accepted       <- conf_archeo >= seuil
  correct_accept <- accepted & is_correct_argmax
  tpr  <- mean(conf_autres  <  seuil)
  tnr  <- mean(correct_accept)
  list(seuil = seuil,
       tpr_autres       = tpr,
       tnr_archeo       = tnr,
       J                = tpr + tnr - 1,
       archeo_accepte   = mean(accepted),
       archeo_argmax    = mean(is_correct_argmax))
}

choisir_seuil_youden_archeo <- function(proba_archeo, espece_archeo,
                                         proba_autres, noms_classes,
                                         resolution = resolution_grille) {
  conf_arch  <- apply(proba_archeo, 1, max)
  pred_arch  <- max.col(proba_archeo)
  pred_esp   <- noms_classes[pred_arch]
  is_correct <- (pred_esp == as.character(espece_archeo))
  conf_aut   <- apply(proba_autres, 1, max)

  seuils <- seq(0, 1, length.out = resolution)
  tab <- lapply(seuils, function(s)
    evaluer_seuil_archeo(conf_arch, is_correct, conf_aut, s))
  df <- do.call(rbind.data.frame, tab)
  best <- which.max(df$J)

  list(auc_note = NA_real_,   # voir plus bas
       seuil    = df$seuil[best],
       J        = df$J[best],
       tpr      = df$tpr_autres[best],
       tnr      = df$tnr_archeo[best],
       archeo_accepte = df$archeo_accepte[best],
       archeo_argmax  = df$archeo_argmax[best],
       courbe   = df,
       conf_archeo = conf_arch,
       is_correct_argmax = is_correct,
       conf_autres = conf_aut)
}

# bootstrap du seuil (IC 95%)
bootstrap_seuil <- function(proba_archeo, espece_archeo, proba_autres,
                             noms_classes, n_boot = n_boot) {
  n_a <- nrow(proba_archeo); n_o <- nrow(proba_autres)
  seuils <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    ia <- sample(n_a, replace = TRUE)
    io <- sample(n_o, replace = TRUE)
    r  <- choisir_seuil_youden_archeo(proba_archeo[ia, , drop = FALSE],
                                      espece_archeo[ia],
                                      proba_autres[io, , drop = FALSE],
                                      noms_classes)
    seuils[b] <- r$seuil
  }
  quantile(seuils, c(0.025, 0.50, 0.975))
}


# ---------------------------------------------------------------------
# 7. EVALUATION AVEC SEUIL (comme Test5 : matrice 7x4)
# ---------------------------------------------------------------------
evaluer_avec_rejet <- function(proba_id, espece_id, statut_id,
                                proba_ood, seuil, noms_classes) {
  conf_id    <- apply(proba_id,  1, max)
  pred_id    <- max.col(proba_id)
  classe_id  <- ifelse(conf_id < seuil, "autre", noms_classes[pred_id])

  conf_ood   <- apply(proba_ood, 1, max)
  classe_ood <- ifelse(conf_ood < seuil, "autre",
                       noms_classes[max.col(proba_ood)])

  niveaux_pred <- c(noms_classes, "autre")
  niveaux_vraie <- c(as.vector(sapply(noms_classes, function(e) c(
    paste(e, "domestique", sep = " | "),
    paste(e, "sauvage",    sep = " | "),
    paste(e, "archeo",     sep = " | ")
  ))), "autre")

  vraie_lab <- c(ifelse(is.na(statut_id),
                        as.character(espece_id),
                        paste(espece_id, statut_id, sep = " | ")),
                 rep("autre", length(classe_ood)))
  pred_lab  <- c(classe_id, classe_ood)

  conf_espece_statut <- table(
    vraie   = factor(vraie_lab,   levels = niveaux_vraie),
    predite = factor(pred_lab,    levels = niveaux_pred))

  taux_correct <- sapply(niveaux_vraie, function(lbl) {
    if (lbl == "autre") return(mean(classe_ood == "autre"))
    idx <- which(vraie_lab == lbl)
    if (length(idx) == 0) return(NA_real_)
    mean(classe_id[idx] == sub(" \\|.*", "", lbl))
  })

  accepte_id <- mean(classe_id != "autre")
  correct_id <- mean(classe_id == as.character(espece_id))
  rejete_ood <- mean(classe_ood == "autre")
  acc_globale <- mean(c(classe_id == as.character(espece_id),
                        classe_ood == "autre"))

  list(seuil = seuil, conf_espece_statut = conf_espece_statut,
       taux_correct_par_ligne = taux_correct,
       accepte_id = accepte_id, correct_id = correct_id,
       rejete_ood = rejete_ood, acc_globale = acc_globale,
       classe_id = classe_id, classe_ood = classe_ood)
}


# ---------------------------------------------------------------------
# 8. ENTRAINEMENT + CALIBRATION + EVALUATION
# ---------------------------------------------------------------------
entrainer_calibrer_evaluer <- function(nom, train_df, val_df,
                                        test_id_df, archeo_df,
                                        calib_autres, test_autres) {
  cat("\n#####################################################\n")
  cat("MODELE :", nom, "\n")
  cat("#####################################################\n")

  k_clear_session()
  try(tensorflow::set_random_seed(seed), silent = TRUE)

  train_gen <- flux(train_df)
  val_gen   <- flux(val_df)
  cat("Codage des classes :\n"); print(train_gen$class_indices)

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
                    loss = "categorical_crossentropy",
                    metrics = c("accuracy"))
  history <- model %>% fit(
    train_gen, steps_per_epoch = ceiling(nrow(train_df)/batch_size),
    epochs = epochs, validation_data = val_gen,
    validation_steps = ceiling(nrow(val_df)/batch_size),
    callbacks = list(
      callback_early_stopping(monitor = "accuracy", patience = 20),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                     patience = 10, min_lr = 1e-7)))
  print(plot(history))

  # --- predictions sur archeo et autres (calibration) ----------------
  cat("\nPredictions archeo (calibration)...\n")
  proba_archeo <- predire_sur_chemins(model, archeo_df$path,
                                       input_shape[1:2], batch_size)
  espece_archeo <- factor(archeo_df$class,
                          levels = names(train_gen$class_indices)[
                            order(unlist(train_gen$class_indices))])
  # recuperer les noms des classes dans l'ordre du modele
  noms_classes <- names(sort(unlist(train_gen$class_indices)))

  cat("Predictions autres (calibration)...\n")
  proba_autres_calib <- predire_sur_chemins(model, calib_autres$path,
                                             input_shape[1:2], batch_size)

  # --- calibration
  calib <- choisir_seuil_youden_archeo(proba_archeo, espece_archeo,
                                        proba_autres_calib, noms_classes)
  cat("\n[CALIBRATION ARCHEO -", nom, "]\n")
  cat("  Archeo (n =", nrow(archeo_df), ") argmax accuracy (plafond) :",
      round(calib$archeo_argmax, 3), "\n")
  if (calib$archeo_argmax < 0.5) {
    cat("  AVERTISSEMENT : le modele classe mal les archeo en argmax.\n")
    cat("  Aucun seuil ne peut depasser ce plafond de bon classement.\n")
  }
  cat("  Seuil retenu (Youden modifie) :", round(calib$seuil, 3), "\n")
  cat("    J =", round(calib$J, 3),
      "| autres rejetes =", round(calib$tpr, 3),
      "| archeo corrects acceptes =", round(calib$tnr, 3), "\n")
  cat("    archeo acceptes (toutes classes confondues) =",
      round(calib$archeo_accepte, 3), "\n")

  # --- bootstrap IC 95%
  cat("\nBootstrap du seuil (", n_boot, " reechantillonnages)...\n")
  ic <- bootstrap_seuil(proba_archeo, espece_archeo,
                         proba_autres_calib, noms_classes, n_boot)
  cat("  IC 95% : [", round(ic[1], 3), ",", round(ic[3], 3), "]",
      " (mediane :", round(ic[2], 3), ")\n")
  if ((ic[3] - ic[1]) > 0.5) {
    cat("  AVERTISSEMENT : IC tres large - seuil mal determine\n")
    cat("  (typique avec si peu d'archeo). Interpretable comme\n")
    cat("  un ordre de grandeur, pas une valeur precise.\n")
  }

  # --- predictions sur test -----------------------------------------
  cat("\nPredictions sur le test externe...\n")
  test_gen <- flux(test_id_df, melanger = FALSE)
  n_test   <- nrow(test_id_df)
  proba_id <- predict(model, test_gen, steps = ceiling(n_test/batch_size),
                      verbose = 0)
  if (nrow(proba_id) > n_test) proba_id <- proba_id[seq_len(n_test), , drop = FALSE]
  classe_vraie <- as.integer(test_gen$classes) + 1
  idx <- match(test_gen$filenames, test_id_df$path)
  if (anyNA(idx)) idx <- seq_len(n_test)
  espece_id <- factor(noms_classes[classe_vraie], levels = noms_classes)
  statut_id <- factor(test_id_df$statut[idx],
                      levels = unname(lib_statut))

  proba_autres_test <- predire_sur_chemins(model, test_autres$path,
                                            input_shape[1:2], batch_size)

  # --- evaluation officielle au seuil calibre -----------------------
  eval_modern <- evaluer_avec_rejet(proba_id, espece_id, statut_id,
                                     proba_autres_test, calib$seuil,
                                     noms_classes)
  cat("\n[TEST modern + autres - seuil =", round(calib$seuil, 3), "]\n")
  cat("Matrice :\n"); print(eval_modern$conf_espece_statut)
  cat("\nTaux correct par ligne :\n")
  print(round(eval_modern$taux_correct_par_ligne, 3))
  cat("\nAutres rejetes (test) :", round(eval_modern$rejete_ood, 3), "\n")

  # --- archeo : mesure sur calib (le seul jeu qu'on ait) -------------
  eval_archeo <- evaluer_avec_rejet(proba_archeo, espece_archeo,
                                     factor(rep("archeo", nrow(archeo_df)),
                                            levels = "archeo"),
                                     proba_autres_test, calib$seuil,
                                     noms_classes)
  cat("\n[ARCHEO - mesure sur calib, biais optimiste]\n")
  cat("Taux correct par espece archeo :\n")
  print(round(eval_archeo$taux_correct_par_ligne[
    grep("archeo", names(eval_archeo$taux_correct_par_ligne))], 3))
  cat("Autres rejetes (test) :", round(eval_archeo$rejete_ood, 3), "\n")

  # --- sauvegarde
  dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
  h5 <- file.path(dossier_res,
                  sprintf("Test5bis_%s_ep%d_seed%d.h5", nom, epochs, seed))
  tryCatch(save_model_hdf5(model, h5),
           error = function(e) message("Modele non sauvegarde : ", conditionMessage(e)))

  list(historique = as.data.frame(history$metrics),
       calib = calib, ic_seuil = ic,
       eval_modern = eval_modern, eval_archeo = eval_archeo,
       noms_classes = noms_classes,
       proba_archeo = proba_archeo, proba_autres_test = proba_autres_test)
}


# ---------------------------------------------------------------------
# 9. ENTRAINEMENT DES DEUX MODELES
# ---------------------------------------------------------------------
res_dom  <- entrainer_calibrer_evaluer("domestiques",
                                        jeux$domestiques$train,
                                        jeux$domestiques$val,
                                        test_id_df, archeo_calib,
                                        autres_calib, autres_test)
res_wild <- entrainer_calibrer_evaluer("sauvages",
                                        jeux$sauvages$train,
                                        jeux$sauvages$val,
                                        test_id_df, archeo_calib,
                                        autres_calib, autres_test)


# ---------------------------------------------------------------------
# 10. COMPARAISON
# ---------------------------------------------------------------------
cat("\n#####################################################\n")
cat("COMPARAISON (focus archeo)\n")
cat("#####################################################\n")

synthese <- data.frame(
  modele                 = c("domestiques", "sauvages"),
  archeo_argmax_plafond  = c(res_dom$calib$archeo_argmax,
                              res_wild$calib$archeo_argmax),
  seuil_youden           = c(res_dom$calib$seuil,
                              res_wild$calib$seuil),
  seuil_IC_bas           = c(res_dom$ic_seuil[1],
                              res_wild$ic_seuil[1]),
  seuil_IC_haut          = c(res_dom$ic_seuil[3],
                              res_wild$ic_seuil[3]),
  archeo_correct_calib   = c(res_dom$calib$tnr,
                              res_wild$calib$tnr),
  autres_rejetes         = c(res_dom$calib$tpr,
                              res_wild$calib$tpr),
  J_youden               = c(res_dom$calib$J,
                              res_wild$calib$J),
  rejete_autres_test     = c(res_dom$eval_modern$rejete_ood,
                              res_wild$eval_modern$rejete_ood),
  correct_id_test        = c(res_dom$eval_modern$correct_id,
                              res_wild$eval_modern$correct_id))
print(synthese, row.names = FALSE)

cat("\nLecture :")
cat("\n  - 'archeo_argmax_plafond' = accuracy argmax sur archeo (indep. du seuil)")
cat("\n  - 'archeo_correct_calib'  = P(archeo correct ET accepte) au seuil retenu")
cat("\n  - Comparer les deux : l'ecart est ce que le seuil coute en archeo.")
cat("\n\n  Note : mesure archeo faite sur le meme jeu que la calibration,")
cat("\n  donc biais optimiste. Avec 23 images, aucune autre option.")
cat("\n")


# ---------------------------------------------------------------------
# 11. VISUALISATION
# ---------------------------------------------------------------------
# distributions de p_max : archeo vs autres (par modele)
df_conf <- bind_rows(
  data.frame(confiance = res_dom$calib$conf_archeo,
             population = "archeo", modele = "domestiques"),
  data.frame(confiance = res_dom$calib$conf_autres,
             population = "autres", modele = "domestiques"),
  data.frame(confiance = res_wild$calib$conf_archeo,
             population = "archeo", modele = "sauvages"),
  data.frame(confiance = res_wild$calib$conf_autres,
             population = "autres", modele = "sauvages"))
print(
  ggplot(df_conf, aes(x = confiance, fill = population)) +
    geom_histogram(position = "identity", alpha = 0.5, bins = 25) +
    facet_wrap(~ modele) +
    geom_vline(data = data.frame(
                 modele = c("domestiques", "sauvages"),
                 seuil  = c(res_dom$calib$seuil, res_wild$calib$seuil)),
               aes(xintercept = seuil),
               linetype = "dashed", color = "black") +
    labs(x = "p_max", y = "Nombre d'images",
         title = "Distribution de p_max : archeo vs autres",
         subtitle = "Trait noir = seuil Youden modifie") +
    theme_minimal())

# courbes J(s) par modele
df_J <- bind_rows(
  data.frame(res_dom$calib$courbe,  modele = "domestiques"),
  data.frame(res_wild$calib$courbe, modele = "sauvages"))
print(
  ggplot(df_J, aes(x = seuil, y = J, color = modele)) +
    geom_line(size = 1) +
    geom_hline(yintercept = 0, linetype = "dotted") +
    labs(x = "Seuil de rejet", y = "J = P(autres rejetes) + P(archeo corrects) - 1",
         title = "Critere de calibration J(s)") +
    theme_minimal())


# ---------------------------------------------------------------------
# 12. SAUVEGARDE
# ---------------------------------------------------------------------
nom_base <- sprintf("Test5bis_archeo_N%d_ep%d_seed%d",
                    N_dom_model, epochs, seed)
resultats <- list(
  master_df       = master_df,
  archeo_calib    = archeo_calib,
  archeo_test     = archeo_test,
  autres_calib    = autres_calib,
  autres_test     = autres_test,
  res_dom         = res_dom,
  res_wild        = res_wild,
  synthese        = synthese,
  config = list(taxons = taxons, N_dom_model = N_dom_model,
                N_dom_ext = N_dom_ext, N_wild_model = N_wild_model,
                N_wild_ext = N_wild_ext, epochs = epochs, seed = seed,
                seed_wild = seed_wild, seed_autres = seed_autres,
                seed_boot = seed_boot, n_boot = n_boot,
                col_image = col_image, prop_calib_autres = prop_calib_autres),
  duree = Sys.time() - begin_time)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))

cat("\nResultats dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")