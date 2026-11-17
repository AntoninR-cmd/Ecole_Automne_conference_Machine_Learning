# =====================================================================
# Test5.R
#
# Ajout d'une classe de rejet "autre" au modele 3 especes de Test3_bis.R
# via un SEUIL DE CONFIANCE sur la probabilite softmax maximale.
#
# Principe :
#   - Meme modele que Test3_bis (VGG19 gele + softmax 3 sorties),
#     entraine sur les graines DOMESTIQUES ou SAUVAGES d'Olea/Phoenix/Vitis.
#   - A l'inference on calcule p_max = max(probas softmax).
#       * si p_max >= seuil  -> espece predite = argmax
#       * si p_max <  seuil  -> classe predite = "autre"
#   - Jeu externe commun (Olea/Phoenix/Vitis dom+sauv) : accepte si p_max >= seuil.
#   - Jeu HORDEUM (espece inconnue du modele)         : rejete si p_max < seuil.
#   - Le seuil est un hyperparametre : on en teste plusieurs.
#
# AVERTISSEMENT METHODOLOGIQUE
#   Un CNN entraine sur 3 classes est souvent SUR-CONFIANT sur des
#   images hors distribution (Hordeum). Le seuil seul ne garantit rien.
#   Un Hordeum predit "olea" avec p=0.95 ne sera pas rejete : le modele
#   se trompe avec confiance. On mesure donc aussi l'AUC (independante
#   du seuil) pour diagnostiquer cette limite.
#
# Commentaires sans accents (encodage Windows).
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

# Tailles de jeu (par taxon et par statut)
N_dom_model  <- 250     # train+val domestique par taxon (200 train / 50 val)
N_dom_ext    <- 400     # calib+test domestique par taxon (200 calib / 200 test)

N_wild_model <- 250     # idem sauvage
N_wild_ext   <- 380     # 190 calib / 190 test

# Fraction du jeu externe utilisee pour CALIBRER le seuil.
# L'autre partie sert de TEST final.
prop_calib <- 0.5

# Hordeum : tailles fixes pour calib et test (separation nette).
N_hordeum_calib <- 712  # 0.5 * 1424 arrondi
N_hordeum_test  <- 712  # reste

prop_train  <- 4/5
epochs      <- 200
seed        <- 2026
seed_wild   <- seed + 1
seed_hord   <- seed + 2
batch_size  <- 32
input_shape <- c(120, 90, 3)

# Seuils testes. 0 = "aucun rejet" (baseline).
seuils_a_tester <- seq(from = 0.34, to = 0.99, by = 0.01)

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

# 2.1 Charger un taxon d'espece (filtre status c/w)
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
    print(names(db)); stop("Colonnes manquantes pour ", taxon)
  }
  cat("Valeurs de 'status' :\n"); print(table(db$status))
  for (s in c(statut_domestique, statut_sauvage)) {
    if (!(s %in% as.character(db$status)))
      stop("Aucune ligne avec status == '", s, "' pour ", taxon)
  }
  db <- db[as.character(db$status) %in% c(statut_domestique, statut_sauvage), ]

  chemins <- as.character(db[[col_image]])
  existe  <- file.exists(chemins)
  cat("Fichiers trouves tels quels :", sum(existe), "/", length(existe), "\n")
  if (!all(existe)) {
    cat("Recherche par nom de fichier dans", dossier_img, "...\n")
    tous        <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    noms_disque <- basename(tous)
    if (anyDuplicated(noms_disque) > 0)
      stop("Noms de fichiers dupliques dans ", dossier_img)
    idx <- match(basename(chemins), noms_disque)
    if (any(is.na(idx))) {
      print(head(basename(chemins)[is.na(idx)]))
      stop(sum(is.na(idx)), " images introuvables")
    }
    chemins <- tous[idx]
  }
  tibble(taxon = taxon,
         statut = as.character(db$status),
         path = chemins)
}

# 2.2 Charger TOUS les Hordeum (sans filtrer par statut : 2r et 6r)
charger_hordeum <- function(taxon, dossier_img) {
  cat("\n===== HORDEUM (classe 'autre') =====\n")
  fichiers <- list.files(dossier_db, pattern = taxon,
                         full.names = TRUE, ignore.case = TRUE)
  if (length(fichiers) == 0) stop("Aucun .rda '", taxon, "' dans ", dossier_db)
  prefere <- fichiers[tolower(basename(fichiers)) == paste0(taxon, "_db2.rda")]
  fichier <- if (length(prefere) > 0) prefere[1] else fichiers[1]
  cat("Fichier .rda :", fichier, "\n")

  env_tmp <- new.env()
  noms    <- load(fichier, envir = env_tmp)
  db      <- get(noms[1], envir = env_tmp)
  if (!(col_image %in% names(db))) {
    print(names(db)); stop("Colonne '", col_image, "' introuvable")
  }
  if ("status" %in% names(db)) {
    cat("Distribution des statuts Hordeum (non filtree) :\n")
    print(table(db$status))
  }

  chemins <- as.character(db[[col_image]])
  existe  <- file.exists(chemins)
  cat("Fichiers trouves tels quels :", sum(existe), "/", length(existe), "\n")
  if (!all(existe)) {
    cat("Recherche par nom de fichier dans", dossier_img, "...\n")
    tous        <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    noms_disque <- basename(tous)
    if (anyDuplicated(noms_disque) > 0)
      stop("Noms de fichiers dupliques dans ", dossier_img)
    idx <- match(basename(chemins), noms_disque)
    if (any(is.na(idx))) {
      print(head(basename(chemins)[is.na(idx)]))
      stop(sum(is.na(idx)), " images introuvables")
    }
    chemins <- tous[idx]
  }
  tibble(taxon = taxon, path = chemins)
}


# ---------------------------------------------------------------------
# 3. CHARGEMENT DES 3 ESPECES
# ---------------------------------------------------------------------
tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]]))
)
cat("\nGraines disponibles (taxon x statut) :\n")
print(table(tous_les_taxons$taxon, tous_les_taxons$statut))


# ---------------------------------------------------------------------
# 4. CHARGEMENT DE HORDEUM
# ---------------------------------------------------------------------
hordeum_all <- charger_hordeum("hordeum", dossier_hordeum)
cat("\nNombre total de Hordeum disponibles :", nrow(hordeum_all), "\n")
if (nrow(hordeum_all) == 0)
  stop("Aucune image Hordeum. Verifier 'dossier_hordeum' et le .rda.")

# ---------------------------------------------------------------------
# 5. TIRAGE ET PARTITION
# ---------------------------------------------------------------------
# --- 5.1 domestiques : train / val / calib / test ---------------------
N_train_d <- round(N_dom_model * prop_train)         # ex. 40
N_test_d  <- N_dom_model - N_train_d                  # ex. 10
N_calib_d <- round(N_dom_ext * prop_calib)            # ex. 50
N_ext_d   <- N_dom_ext - N_calib_d                    # ex. 50
labs_dom  <- c(rep("train",    N_train_d),
               rep("test",     N_test_d),
               rep("calib",    N_calib_d),
               rep("external", N_ext_d))
set.seed(seed)
shuffled_dom <- sample(labs_dom)

dom <- tous_les_taxons %>% filter(statut == statut_domestique)
if (any(table(dom$taxon) < length(labs_dom)))
  stop("Pas assez de graines domestiques dans un taxon (besoin de ",
       length(labs_dom), ").")
df_dom <- dom %>%
  group_by(taxon) %>% sample_n(length(labs_dom)) %>%
  mutate(partition = shuffled_dom) %>% ungroup()

# --- 5.2 sauvages -----------------------------------------------------
N_train_w <- round(N_wild_model * prop_train)
N_test_w  <- N_wild_model - N_train_w
N_calib_w <- round(N_wild_ext * prop_calib)
N_ext_w   <- N_wild_ext - N_calib_w
labs_wild <- c(rep("train",    N_train_w),
               rep("test",     N_test_w),
               rep("calib",    N_calib_w),
               rep("external", N_ext_w))
set.seed(seed_wild)
shuffled_wild <- sample(labs_wild)

wild <- tous_les_taxons %>% filter(statut == statut_sauvage)
if (any(table(wild$taxon) < length(labs_wild)))
  stop("Pas assez de graines sauvages dans un taxon (besoin de ",
       length(labs_wild), ").")
df_wild <- wild %>%
  group_by(taxon) %>% sample_n(length(labs_wild)) %>%
  mutate(partition = shuffled_wild) %>% ungroup()

# --- 5.3 tableau final -----------------------------------------------
lib_statut <- setNames(c("domestique", "sauvage"),
                       c(statut_domestique, statut_sauvage))
master_df <- bind_rows(df_dom, df_wild) %>%
  transmute(partition, class = taxon,
            statut = unname(lib_statut[statut]), path)

cat("\nPartition x espece x statut :\n")
print(ftable(table(partition = master_df$partition,
                   espece    = master_df$class,
                   statut    = master_df$statut)))

calib_id_df <- master_df %>% filter(partition == "calib")
test_id_df  <- master_df %>% filter(partition == "external")

jeux <- list(
  domestiques = list(
    train = master_df %>% filter(partition == "train", statut == "domestique"),
    val   = master_df %>% filter(partition == "test",  statut == "domestique")),
  sauvages = list(
    train = master_df %>% filter(partition == "train", statut == "sauvage"),
    val   = master_df %>% filter(partition == "test",  statut == "sauvage"))
)

# --- 5.4 split Hordeum en calib et test ------------------------------
set.seed(seed_hord)
N_hordeum_total <- N_hordeum_calib + N_hordeum_test
if (nrow(hordeum_all) < N_hordeum_total)
  stop("Pas assez de Hordeum (", nrow(hordeum_all), "), besoin de ",
       N_hordeum_total, ".")
idx_hord      <- sample(nrow(hordeum_all), N_hordeum_total)
hordeum_calib <- hordeum_all[idx_hord[1:N_hordeum_calib], ]
hordeum_test  <- hordeum_all[idx_hord[(N_hordeum_calib+1):N_hordeum_total], ]

cat("\nJeux de calibration / test :\n")
cat("  ID calib :", nrow(calib_id_df), "  |  ID test :", nrow(test_id_df), "\n")
cat("  OOD calib:", nrow(hordeum_calib), " |  OOD test:", nrow(hordeum_test), "\n")

# ---------------------------------------------------------------------
# 6. FONCTIONS COMMUNES
# ---------------------------------------------------------------------
all_datagen <- image_data_generator(rescale = 1/255)

flux <- function(x, melanger = TRUE) {
  flow_images_from_dataframe(
    x, directory = "", x_col = "path", y_col = "class",
    generator = all_datagen, target_size = input_shape[1:2],
    class_mode = "categorical", batch_size = batch_size,
    shuffle = melanger, seed = seed)
}

# Predictions sur Hordeum : chargement direct (pas de generateur).
# On evite ainsi les problemes de generateurs dont la classe "hordeum"
# n'existe pas dans le modele. Chargement sequentiel + rescale 1/255
# pour rester coherent avec all_datagen.
predire_sur_chemins <- function(model, chemins, target_size, batch_size) {
  n <- length(chemins)
  H <- target_size[1]; W <- target_size[2]
  arr <- array(0, dim = c(n, H, W, 3))
  for (i in seq_len(n)) {
    img <- image_load(chemins[i], target_size = target_size, color_mode = "rgb")
    arr[i,,,] <- image_to_array(img) / 255
  }
  predict(model, arr, batch_size = batch_size, verbose = 0)
}
# Prediction sur un data.frame (ID : colonnes path, class, statut)
# et extraction des vraies etiquettes dans l'ordre du generateur.
predire_sur_df <- function(model, df, batch_size) {
  n     <- nrow(df)
  steps <- ceiling(n / batch_size)
  gen   <- flux(df, melanger = FALSE)
  proba <- predict(model, gen, steps = steps, verbose = 0)
  if (nrow(proba) > n) proba <- proba[seq_len(n), , drop = FALSE]
  if (nrow(proba) != n) stop("predict a retourne ", nrow(proba),
                             " predictions pour ", n, " images")
  classe_vraie <- as.integer(gen$classes) + 1
  noms_classes <- names(sort(unlist(gen$class_indices)))
  espece <- factor(noms_classes[classe_vraie], levels = noms_classes)
  idx    <- match(gen$filenames, df$path)
  if (anyNA(idx)) idx <- seq_len(n)
  statut <- factor(df$statut[idx], levels = unname(lib_statut))
  list(proba = proba, espece = espece, statut = statut,
       noms_classes = noms_classes)
}

# ---------------------------------------------------------------------
# 7. EVALUATION AVEC SEUIL DE REJET
# ---------------------------------------------------------------------
# proba_ext    : (n_ext, 3)  probabilites sur les 3 especes
# espece_ext   : facteur des vraies especes
# statut_ext   : facteur dom/sauvage
# proba_hord   : (n_hord, 3) probabilites sur les Hordeum
# seuil        : p_max < seuil -> "autre"
# noms_classes : c("olea","phoenix","vitis") dans l'ordre du generateur
evaluer_avec_rejet <- function(proba_ext, espece_ext, statut_ext,
                               proba_hord, seuil, noms_classes) {

  # --- decision sur le jeu externe
  conf_ext   <- apply(proba_ext, 1, max)
  pred_ext   <- max.col(proba_ext)
  classe_ext <- ifelse(conf_ext < seuil, "autre", noms_classes[pred_ext])

  # --- decision sur Hordeum
  conf_hord   <- apply(proba_hord, 1, max)
  classe_hord <- ifelse(conf_hord < seuil, "autre",
                        noms_classes[max.col(proba_hord)])

  # --- matrice 4x4 (vraie classe x classe predite)
  vraie_combinee   <- c(as.character(espece_ext),
                        rep("autre", length(classe_hord)))
  predite_combinee <- c(classe_ext, classe_hord)
  niveaux_4 <- c(noms_classes, "autre")
  conf_4x4 <- table(
    vraie   = factor(vraie_combinee,   levels = niveaux_4),
    predite = factor(predite_combinee, levels = niveaux_4)
  )

  # --- metriques
  n_ext  <- length(classe_ext)
  n_hord <- length(classe_hord)

  accepte_id <- mean(classe_ext != "autre")                          # % acceptes parmi ID
  correct_id <- mean(classe_ext == as.character(espece_ext))         # % bien classes parmi ID
  rejete_ood <- mean(classe_hord == "autre")                         # % rejetes parmi OOD
  acc_4      <- mean(predite_combinee == vraie_combinee)             # accuracy 4 classes

  # AUC OOD : score d'anomalie = 1 - p_max.
  # AUC = P(score_OOD > score_ID), independante du seuil.
  scores_anomalie <- c(1 - conf_ext, 1 - conf_hord)
  labels_ood      <- c(rep(0, n_ext), rep(1, n_hord))
  r  <- rank(scores_anomalie)
  n1 <- sum(labels_ood == 1); n0 <- sum(labels_ood == 0)
  auc_ood <- (sum(r[labels_ood == 1]) - n1*(n1+1)/2) / (n1*n0)

  # F1 sur la classe "autre"
  tp <- sum(predite_combinee == "autre" & vraie_combinee == "autre")
  fp <- sum(predite_combinee == "autre" & vraie_combinee != "autre")
  fn <- sum(predite_combinee != "autre" & vraie_combinee == "autre")
  prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  rec  <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0)
            2 * prec * rec / (prec + rec) else NA_real_

  acc_par_statut <- tapply(classe_ext == as.character(espece_ext),
                            statut_ext, mean)

  list(seuil = seuil, conf_4x4 = conf_4x4,
       acc_4 = acc_4, accepte_id = accepte_id, correct_id = correct_id,
       rejete_ood = rejete_ood, auc_ood = auc_ood,
       precision_autre = prec, rappel_autre = rec, f1_autre = f1,
       acc_par_statut = acc_par_statut,
       classe_ext = classe_ext, classe_hord = classe_hord,
       conf_ext = conf_ext, conf_hord = conf_hord)
}
# Selection naive du seuil : maximise acc_4 sur la CALIBRATION.
# En cas d'egalite, on retient le seuil median (compromis central).
choisir_seuil_naif <- function(evals) {
  acc_4  <- sapply(evals, `[[`, "acc_4")
  best   <- which(acc_4 == max(acc_4, na.rm = TRUE))
  if (length(best) > 1)
    cat("  (", length(best), "seuils a egalite sur acc_4 : ",
        paste(sapply(evals[best], `[[`, "seuil"), collapse = ", "),
        ")\n", sep = "")
  idx_opt <- best[ceiling(length(best) / 2)]
  list(seuil = evals[[idx_opt]]$seuil,
       detail = evals[[idx_opt]],
       toutes = evals)
}
entrainer_et_evaluer <- function(nom, train_df, val_df,
                                 calib_id_df, calib_ood_df,
                                 test_id_df,  test_ood_df,
                                 seuils) {
  cat("\n#####################################################\n")
  cat("MODELE :", nom, "\n")
  cat("#####################################################\n")
  
  k_clear_session()
  try(tensorflow::set_random_seed(seed), silent = TRUE)
  
  train_generator <- flux(train_df)
  val_generator   <- flux(val_df)
  cat("Codage des classes :\n"); print(train_generator$class_indices)
  
  # --- architecture identique a Test3_bis
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
  
  # --- predictions sur la CALIBRATION
  cat("\nPrediction sur calibration (ID)...\n")
  pred_calib_id <- predire_sur_df(model, calib_id_df, batch_size)
  cat("Prediction sur calibration (OOD)...\n")
  proba_calib_ood <- predire_sur_chemins(model, calib_ood_df$path,
                                         input_shape[1:2], batch_size)
  noms_classes <- pred_calib_id$noms_classes
  
  # --- predictions sur le TEST
  cat("Prediction sur test (ID)...\n")
  pred_test_id <- predire_sur_df(model, test_id_df, batch_size)
  cat("Prediction sur test (OOD)...\n")
  proba_test_ood <- predire_sur_chemins(model, test_ood_df$path,
                                        input_shape[1:2], batch_size)
  
  # --- selection naive du seuil sur CALIB
  cat("\n--- Selection naive du seuil (max acc_4 sur calibration) ---\n")
  evals_calib <- lapply(seuils, function(s)
    evaluer_avec_rejet(pred_calib_id$proba, pred_calib_id$espece,
                       pred_calib_id$statut, proba_calib_ood,
                       s, noms_classes))
  choix <- choisir_seuil_naif(evals_calib)
  seuil_opt <- choix$seuil
  cat("Seuil retenu :", seuil_opt, "\n")
  
  tab_calib <- data.frame(
    seuil         = sapply(evals_calib, `[[`, "seuil"),
    accepte_id    = sapply(evals_calib, `[[`, "accepte_id"),
    correct_id    = sapply(evals_calib, `[[`, "correct_id"),
    rejete_ood    = sapply(evals_calib, `[[`, "rejete_ood"),
    acc_4         = sapply(evals_calib, `[[`, "acc_4"),
    f1_autre      = sapply(evals_calib, `[[`, "f1_autre"),
    auc_ood       = sapply(evals_calib, `[[`, "auc_ood"))
  cat("\nCourbe de calibration :\n")
  print(tab_calib, row.names = FALSE)
  
  # --- evaluation sur TEST au seuil retenu
  ev_test_opt <- evaluer_avec_rejet(pred_test_id$proba, pred_test_id$espece,
                                    pred_test_id$statut, proba_test_ood,
                                    seuil_opt, noms_classes)
  cat("\n--- Evaluation sur TEST au seuil", seuil_opt, "---\n")
  cat("Matrice 4x4 :\n"); print(ev_test_opt$conf_4x4)
  cat("Accuracy 4 classes          :", round(ev_test_opt$acc_4, 3), "\n")
  cat("% ID acceptes                :", round(ev_test_opt$accepte_id, 3), "\n")
  cat("% ID correctement classés    :", round(ev_test_opt$correct_id, 3), "\n")
  cat("% OOD rejetes                :", round(ev_test_opt$rejete_ood, 3), "\n")
  cat("F1 classe 'autre'           :", round(ev_test_opt$f1_autre, 3), "\n")
  cat("AUC OOD (diagnostic)        :", round(ev_test_opt$auc_ood, 3), "\n")
  cat("Accuracy ID par statut :\n")
  print(round(ev_test_opt$acc_par_statut, 3))
  
  # --- sweep sur TEST pour comparaison
  evals_test <- lapply(seuils, function(s)
    evaluer_avec_rejet(pred_test_id$proba, pred_test_id$espece,
                       pred_test_id$statut, proba_test_ood,
                       s, noms_classes))
  tab_test <- data.frame(
    seuil      = sapply(evals_test, `[[`, "seuil"),
    accepte_id = sapply(evals_test, `[[`, "accepte_id"),
    rejete_ood = sapply(evals_test, `[[`, "rejete_ood"),
    acc_4      = sapply(evals_test, `[[`, "acc_4"),
    f1_autre   = sapply(evals_test, `[[`, "f1_autre"))
  cat("\nCourbe de test (comparaison) :\n")
  print(tab_test, row.names = FALSE)
  
  # --- sauvegarde
  dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
  fichier_h5 <- file.path(dossier_res,
                          sprintf("Test5_modele_%s_ep%d_seed%d.h5",
                                  nom, epochs, seed))
  tryCatch(save_model_hdf5(model, fichier_h5),
           error = function(e) message("Modele non sauvegarde : ",
                                       conditionMessage(e)))
  
  list(historique    = as.data.frame(history$metrics),
       noms_classes  = noms_classes,
       seuil_optimal = seuil_opt,
       tab_calib     = tab_calib,
       tab_test      = tab_test,
       ev_test_opt   = ev_test_opt,
       evals_calib   = evals_calib,
       evals_test    = evals_test,
       pred_calib_id = pred_calib_id,
       proba_calib_ood = proba_calib_ood,
       pred_test_id  = pred_test_id,
       proba_test_ood  = proba_test_ood)
}

res_dom  <- entrainer_et_evaluer("domestiques",
                                 jeux$domestiques$train,
                                 jeux$domestiques$val,
                                 calib_id_df, hordeum_calib,
                                 test_id_df,  hordeum_test,
                                 seuils_a_tester)
res_wild <- entrainer_et_evaluer("sauvages",
                                 jeux$sauvages$train,
                                 jeux$sauvages$val,
                                 calib_id_df, hordeum_calib,
                                 test_id_df,  hordeum_test,
                                 seuils_a_tester)

# ---------------------------------------------------------------------
# 10. COMPARAISON DES MODELES
# ---------------------------------------------------------------------
cat("\n#####################################################\n")
cat("COMPARAISON DES MODELES\n")
cat("#####################################################\n")

resume_modele <- function(res) {
  c(seuil        = res$seuil_optimal,
    acc_4_test   = res$ev_test_opt$acc_4,
    accepte_id   = res$ev_test_opt$accepte_id,
    correct_id   = res$ev_test_opt$correct_id,
    rejete_ood   = res$ev_test_opt$rejete_ood,
    f1_autre     = res$ev_test_opt$f1_autre,
    auc_ood      = res$ev_test_opt$auc_ood)
}
tab_final <- rbind(
  "entraine sur domestiques" = resume_modele(res_dom),
  "entraine sur sauvages"    = resume_modele(res_wild))
cat("\nResume (jeu de TEST au seuil choisi sur CALIBRATION) :\n")
print(round(tab_final, 3))

cat("\nDiagnostic AUC (independant du seuil) :\n")
cat("  < 0.65 : seuil de confiance insuffisant -> changer de score d'anomalie\n")
cat("  >= 0.75 : compromis possible, interpreter les courbes\n")

# ---------------------------------------------------------------------
# 11. VISUALISATION
# ---------------------------------------------------------------------
# --- 11.1 Compromis calibration : accepte_id vs rejete_ood --------------
df_calib <- bind_rows(lapply(c("domestiques", "sauvages"), function(nom) {
  res <- if (nom == "domestiques") res_dom else res_wild
  res$tab_calib %>% mutate(modele = nom, source = "calibration")
}))
df_test <- bind_rows(lapply(c("domestiques", "sauvages"), function(nom) {
  res <- if (nom == "domestiques") res_dom else res_wild
  res$tab_test %>% mutate(modele = nom, source = "test")
}))

pts_opt <- bind_rows(
  data.frame(modele = "domestiques",
             accepte_id = res_dom$ev_test_opt$accepte_id,
             rejete_ood = res_dom$ev_test_opt$rejete_ood,
             seuil = res_dom$seuil_optimal),
  data.frame(modele = "sauvages",
             accepte_id = res_wild$ev_test_opt$accepte_id,
             rejete_ood = res_wild$ev_test_opt$rejete_ood,
             seuil = res_wild$seuil_optimal))

print(
  ggplot() +
    geom_path(data = df_calib, aes(x = accepte_id, y = rejete_ood,
                                   color = modele), linewidth = 1) +
    geom_point(data = df_calib, aes(x = accepte_id, y = rejete_ood,
                                    color = modele), size = 2) +
    geom_point(data = pts_opt, aes(x = accepte_id, y = rejete_ood),
               shape = 4, size = 5, stroke = 1.5, color = "black") +
    geom_text(data = pts_opt,
              aes(x = accepte_id, y = rejete_ood,
                  label = paste0("seuil=", seuil)),
              vjust = -1.2, size = 3.2) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(x = "Fraction d'ID acceptees", y = "Fraction d'OOD rejetees",
         title = "Compromis de rejet sur la calibration",
         subtitle = "X = seuil retenu ; diagonale = hasard",
         color = "Modele") +
    theme_minimal())

# --- 11.2 Accuracy 4 classes vs seuil ----------------------------------
df_acc <- bind_rows(lapply(c("domestiques", "sauvages"), function(nom) {
  res <- if (nom == "domestiques") res_dom else res_wild
  rbind(
    res$tab_calib %>% transmute(seuil, acc_4, modele = nom,
                                source = "calibration"),
    res$tab_test  %>% transmute(seuil, acc_4, modele = nom,
                                source = "test"))
}))
seuils_opt <- c(domestiques = res_dom$seuil_optimal,
                sauvages    = res_wild$seuil_optimal)
df_opt <- data.frame(modele = names(seuils_opt),
                     seuil  = unname(seuils_opt),
                     acc_4  = sapply(names(seuils_opt), function(n) {
                       r <- if (n == "domestiques") res_dom else res_wild
                       r$ev_test_opt$acc_4
                     }))

print(
  ggplot(df_acc, aes(x = seuil, y = acc_4,
                     color = modele)) +
    geom_line() + geom_point() +
    geom_vline(data = df_opt,
               aes(xintercept = seuil, color = modele, linetype = "dotted"),
               inherit.aes = FALSE, show.legend = FALSE) +
    geom_point(data = df_opt, aes(x = seuil, y = acc_4, color = modele),
               shape = 4, size = 4, stroke = 1.5) +
    labs(x = "Seuil de confiance", y = "Accuracy 4 classes",
         title = "Accuracy 4 classes vs seuil",
         subtitle = "Trait plein = calibration ; pointille = test ; croix = seuil retenu") +
    theme_minimal())

# --- 11.3 Distribution des confiances au seuil retenu ------------------
df_conf <- bind_rows(
  data.frame(confiance = apply(res_dom$pred_test_id$proba, 1, max),
             population = "ID (test)", modele = "domestiques"),
  data.frame(confiance = apply(res_dom$proba_test_ood, 1, max),
             population = "Hordeum (test)", modele = "domestiques"),
  data.frame(confiance = apply(res_wild$pred_test_id$proba, 1, max),
             population = "ID (test)", modele = "sauvages"),
  data.frame(confiance = apply(res_wild$proba_test_ood, 1, max),
             population = "Hordeum (test)", modele = "sauvages"))

print(
  ggplot(df_conf, aes(x = confiance, fill = population)) +
    geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
    facet_wrap(~ modele) +
    geom_vline(data = data.frame(modele = c("domestiques", "sauvages"),
                                 x = c(res_dom$seuil_optimal,
                                       res_wild$seuil_optimal)),
               aes(xintercept = x), linetype = "dashed", color = "red") +
    labs(x = "p_max (softmax max)", y = "Nombre d'images",
         title = "Distribution de la confiance (jeu de test)",
         subtitle = "Trait rouge = seuil retenu") +
    theme_minimal())
# ---------------------------------------------------------------------
# 12. SAUVEGARDE
# ---------------------------------------------------------------------
nom_base <- sprintf("Test5_2_bis_N%d_ep%d_seed%d",
                    N_dom_model, epochs, seed)
resultats <- list(
  master_df = master_df,
  hordeum_calib = hordeum_calib, hordeum_test = hordeum_test,
  res_dom = res_dom, res_wild = res_wild,
  seuils = seuils_a_tester,
  config = list(taxons = taxons, dossier_hordeum = dossier_hordeum,
                N_dom_model = N_dom_model, N_dom_ext = N_dom_ext,
                N_wild_model = N_wild_model, N_wild_ext = N_wild_ext,
                N_hordeum_calib = N_hordeum_calib,
                N_hordeum_test  = N_hordeum_test,
                prop_calib = prop_calib,
                epochs = epochs, seed = seed, seed_wild = seed_wild,
                seed_hord = seed_hord, col_image = col_image),
  duree = Sys.time() - begin_time)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))
cat("\nResultats ecrits dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")