# =====================================================================
# Multi_seed_Test3_Test3bis.R
# Entraine 5 fois chacun des 3 types de modeles suivants, sur des
# partitions aleatoires differentes, puis evalue chaque modele sur :
#   1) le jeu externe (domestiques + sauvages, partition "external")
#   2) les donnees archeo (3 especes, depuis archeo_complet_db.rda)
#
# Types de modeles (tous 3 sorties softmax = 3 especes) :
#   - Test3     : entraine sur domestiques + sauvages
#   - Test3bisD : entraine sur domestiques seuls
#   - Test3bisW : entraine sur sauvages seuls
#
# Les 3 modeles d'une meme repetition partagent EXACTEMENT la meme
# partition (memes images train/val/external, memes tirages). C'est ce
# qui rend leurs performances comparables.
#
# Sortie :
#   - 15 fichiers .h5 et 15 fichiers .rda (un par modele et par rep)
#   - un .rda agrege avec moyennes et ecart-types sur les 5 repetitions
#   - tableau recapitulatif affiche en console
#
# Bloc Test5 (Sauvage / Dom / Dom+Sauv -> + Autre) : a completer plus
# tard, non implemente ici (commentaire en fin de script).
#
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine            <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db        <- "rda/db"
dossier_rda_ajout <- "rda/db_ajout"
dossier_sortie    <- "rda/results/multi_seed_Test3_3bis"
dossier_modeles   <- file.path(dossier_sortie, "modeles")

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")
col_image <- "VD_deep"
statut_domestique <- "c"
statut_sauvage    <- "w"

# Effectifs identiques a Test3.R / Test3_bis.R
N_dom_model   <- 530
N_dom_ext     <- 100
N_wild_model  <- 530
N_wild_ext    <- 100
prop_train    <- 4/5

# --- taille de la campagne ------------------------------------------
N_REPS      <- 5       # nombre de repetitions (mettre 1 pour tester)
epochs      <- 100     # max, early stopping s'en charge (baisser pour test)
batch_size  <- 32
input_shape <- c(120, 90, 3)

# Seeds par repetition : 5 paires (dom, wild) sur la plage 2329-2338
# rep 1 : dom=2329 wild=2330 (== Test3.R d'origine)
# rep 2 : dom=2331 wild=2332
# ...
# rep 5 : dom=2337 wild=2338
seeds_dom  <- 2329 + 2 * (0:(N_REPS - 1))
seeds_wild <- seeds_dom + 1

# Fichier .rda des archeo (issu de Creer_rda_archeo.R / RDA.R)
rda_archeo <- file.path(dossier_rda_ajout, "archeo_complet_db.rda")

# Mettre TRUE pour reentrainer un modele deja present (par defaut on
# reutilise le .h5 si disponible : pratique pour relancer l'agregation).
reecrire_modeles <- TRUE


# ---------------------------------------------------------------------
# 1. PACKAGES ET DOSSIER DE TRAVAIL
# ---------------------------------------------------------------------
library(tidyverse)
library(keras)
py_require_legacy_keras()
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")
begin_time <- Sys.time()

dir.create(dossier_modeles, recursive = TRUE, showWarnings = FALSE)


# ---------------------------------------------------------------------
# 2. CHARGEMENT DES 3 ESPECES MODERNES (dom + wild)
# ---------------------------------------------------------------------
# Meme fonction que Test3.R (copie fidele, pour garantir les memes
# chemins d'images et les memes tirages).
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
    tous  <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    noms_disque <- basename(tous)
    if (anyDuplicated(noms_disque) > 0)
      stop("Noms de fichiers dupliques dans ", dossier_img)
    idx <- match(basename(chemins), noms_disque)
    if (any(is.na(idx))) stop(sum(is.na(idx)), " images introuvables.")
    chemins <- tous[idx]
  }
  tibble(taxon = taxon, statut = as.character(db$status), path = chemins)
}

tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]]))
)
cat("\nGraines disponibles (taxon x statut) :\n")
print(table(tous_les_taxons$taxon, tous_les_taxons$statut))


# ---------------------------------------------------------------------
# 3. CHARGEMENT DES ARCHEO (une seule fois, reutilise par toutes les evals)
# ---------------------------------------------------------------------
if (!file.exists(rda_archeo))
  stop("Fichier archeo introuvable : ", rda_archeo)

env_a <- new.env()
noms_a <- load(rda_archeo, envir = env_a)
res_archeo <- get(noms_a[1], envir = env_a)
if (!is.list(res_archeo) || is.null(res_archeo$master_df))
  stop("Pas de 'master_df' dans ", rda_archeo)

master_archeo <- as_tibble(res_archeo$master_df)
classes_modele <- sort(unique(tous_les_taxons$taxon))   # c("olea","phoenix","vitis")

archeo_3 <- master_archeo %>%
  filter(class %in% classes_modele) %>%
  mutate(statut = "archeo") %>%
  select(path, class, statut)
archeo_o <- master_archeo %>%
  filter(!(class %in% classes_modele)) %>%
  select(path, class, statut)

cat("\nArcheo retenus (3 especes) : ", nrow(archeo_3), "\n")
print(table(archeo_3$class, useNA = "ifany"))
if (nrow(archeo_o) > 0) {
  cat("Archeo 'autre' (bonus OOD) : ", nrow(archeo_o), "\n")
  print(table(archeo_o$class, useNA = "ifany"))
}
if (nrow(archeo_3) == 0) stop("Aucune image archeo des 3 especes.")
# filtre existence
archeo_3 <- archeo_3[file.exists(archeo_3$path), ]
if (nrow(archeo_o) > 0) archeo_o <- archeo_o[file.exists(archeo_o$path), ]


# ---------------------------------------------------------------------
# 4. FONCTIONS
# ---------------------------------------------------------------------

# --- 4.1 construction de la partition (copie de Test3.R) -------------
construire_partitions <- function(seed_dom, seed_wild) {
  N_train_d <- round(N_dom_model * prop_train)
  N_test_d  <- N_dom_model - N_train_d
  labs_dom  <- c(rep("train", N_train_d), rep("test", N_test_d),
                 rep("external", N_dom_ext))
  set.seed(seed_dom)
  shuffled_dom <- sample(labs_dom)

  dom <- tous_les_taxons %>% filter(statut == statut_domestique)
  if (any(table(dom$taxon) < length(labs_dom)))
    stop("Pas assez de graines domestiques dans un taxon.")
  df_dom <- dom %>%
    group_by(taxon) %>%
    slice_sample(n = length(labs_dom)) %>%
    mutate(partition = shuffled_dom) %>%
    ungroup()

  N_train_w <- round(N_wild_model * prop_train)
  N_test_w  <- N_wild_model - N_train_w
  labs_wild <- c(rep("train", N_train_w), rep("test", N_test_w),
                 rep("external", N_wild_ext))
  set.seed(seed_wild)
  shuffled_wild <- sample(labs_wild)

  wild <- tous_les_taxons %>% filter(statut == statut_sauvage)
  if (any(table(wild$taxon) < length(labs_wild)))
    stop("Pas assez de graines sauvages dans un taxon.")
  df_wild <- wild %>%
    group_by(taxon) %>%
    slice_sample(n = length(labs_wild)) %>%
    mutate(partition = shuffled_wild) %>%
    ungroup()

  lib_statut <- setNames(c("domestique", "sauvage"),
                         c(statut_domestique, statut_sauvage))
  bind_rows(df_dom, df_wild) %>%
    transmute(partition,
              class  = taxon,
              statut = unname(lib_statut[statut]),
              path)
}

# --- 4.2 entrainement d'un modele (VGG19 gele + tete 3 sorties) -----
entrainer_modele <- function(train_df, val_df, seed, verbose = 2) {
  datagen <- image_data_generator(rescale = 1/255)
  flux <- function(x, melanger = TRUE) {
    flow_images_from_dataframe(
      x, directory = "", x_col = "path", y_col = "class",
      generator = datagen, target_size = input_shape[1:2],
      class_mode = "categorical", batch_size = batch_size,
      shuffle = melanger, seed = seed)
  }
  base_model <- application_vgg19(weights = "imagenet", include_top = FALSE,
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
  cbs <- list(
    callback_early_stopping(monitor = "accuracy", patience = 20),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                  patience = 10, min_lr = 1e-7))
  hist <- model %>% fit(
    flux(train_df),
    steps_per_epoch  = ceiling(nrow(train_df) / batch_size),
    epochs           = epochs,
    validation_data  = flux(val_df),
    validation_steps = ceiling(nrow(val_df) / batch_size),
    callbacks        = cbs,
    verbose          = verbose)
  list(model = model, history = hist)
}

# --- 4.3 evaluation externe (dom + wild, 3 especes) ------------------
evaluer_externe <- function(model, ext_df) {
  if (nrow(ext_df) == 0) return(NULL)
  datagen <- image_data_generator(rescale = 1/255)
  gen <- flow_images_from_dataframe(
    as.data.frame(ext_df), directory = "", x_col = "path", y_col = "class",
    generator = datagen, target_size = input_shape[1:2],
    class_mode = NULL, batch_size = batch_size, shuffle = FALSE)
  proba <- predict(model, gen, steps = ceiling(nrow(ext_df) / batch_size),
                   verbose = 0)
  idx <- match(gen$filenames, ext_df$path)
  if (anyNA(idx)) idx <- seq_len(nrow(ext_df))
  proba  <- proba[idx, , drop = FALSE]
  ext_df <- ext_df[idx, ]

  pred <- factor(classes_modele[max.col(proba, ties.method = "first")],
                 levels = classes_modele)
  correct <- pred == ext_df$class

  list(
    n                  = nrow(ext_df),
    accuracy_globale   = mean(correct),
    acc_par_statut     = tapply(correct, ext_df$statut, mean),
    acc_par_espece     = tapply(correct, ext_df$class, mean),
    acc_espece_statut  = tapply(correct,
                                paste(ext_df$class, ext_df$statut, sep = "|"),
                                mean),
    conf               = table(vraie = ext_df$class, predite = pred),
    proba              = proba)
}

# --- 4.4 evaluation archeo (3 especes + eventuel autre OOD) ----------
evaluer_archeo <- function(model, archeo_3, archeo_o) {
  if (nrow(archeo_3) == 0) return(NULL)
  datagen <- image_data_generator(rescale = 1/255)

  gen <- flow_images_from_dataframe(
    as.data.frame(archeo_3), directory = "", x_col = "path", y_col = "class",
    generator = datagen, target_size = input_shape[1:2],
    class_mode = NULL, batch_size = batch_size, shuffle = FALSE)
  proba <- predict(model, gen, steps = ceiling(nrow(archeo_3) / batch_size),
                   verbose = 0)
  idx <- match(gen$filenames, archeo_3$path)
  if (anyNA(idx)) idx <- seq_len(nrow(archeo_3))
  proba    <- proba[idx, , drop = FALSE]
  archeo_3 <- archeo_3[idx, ]

  pred <- factor(classes_modele[max.col(proba, ties.method = "first")],
                 levels = classes_modele)
  correct <- pred == archeo_3$class

  niveaux_lignes <- paste0(classes_modele, " | archeo")
  ligne_f <- factor(paste0(archeo_3$class, " | archeo"), levels = niveaux_lignes)
  conf_archeo <- table(ligne = ligne_f, predite = pred)

  # images "autre" (OOD), si presentes
  conf_autre <- NULL
  if (!is.null(archeo_o) && nrow(archeo_o) > 0) {
    gen_o <- flow_images_from_dataframe(
      as.data.frame(archeo_o), directory = "", x_col = "path", y_col = "class",
      generator = datagen, target_size = input_shape[1:2],
      class_mode = NULL, batch_size = batch_size, shuffle = FALSE)
    proba_o <- predict(model, gen_o,
                       steps = ceiling(nrow(archeo_o) / batch_size), verbose = 0)
    idx_o <- match(gen_o$filenames, archeo_o$path)
    if (anyNA(idx_o)) idx_o <- seq_len(nrow(archeo_o))
    proba_o   <- proba_o[idx_o, , drop = FALSE]
    archeo_o  <- archeo_o[idx_o, ]
    pred_o <- factor(classes_modele[max.col(proba_o, ties.method = "first")],
                     levels = classes_modele)
    conf_autre <- table(vraie = archeo_o$class, predite = pred_o)
  }

  list(
    n                    = nrow(archeo_3),
    conf_archeo          = conf_archeo,
    acc_par_espece       = tapply(correct, archeo_3$class, mean),
    accuracy_globale     = mean(correct),
    conf_autre           = conf_autre,
    proba                = proba)
}


# ---------------------------------------------------------------------
# 5. BOUCLE PRINCIPALE
# ---------------------------------------------------------------------
resultats_par_rep <- list()

for (r in seq_len(N_REPS)) {
  seed_dom  <- seeds_dom[r]
  seed_wild <- seeds_wild[r]
  cat("\n\n########################################################\n")
  cat("REPETITION", r, "/", N_REPS,
      " (seed_dom =", seed_dom, ", seed_wild =", seed_wild, ")\n")
  cat("########################################################\n")

  master_df <- construire_partitions(seed_dom, seed_wild)
  ext_df    <- master_df %>% filter(partition == "external")

  cat("Partition x espece x statut :\n")
  print(ftable(table(partition = master_df$partition,
                     espece    = master_df$class,
                     statut    = master_df$statut)))

  res_rep <- list()

  for (modele_type in c("Test3", "Test3bisD", "Test3bisW")) {
    cat("\n----- Modele :", modele_type, "-----\n")

    if (modele_type == "Test3") {
      train_df <- master_df %>% filter(partition == "train")
      val_df   <- master_df %>% filter(partition == "test")
    } else if (modele_type == "Test3bisD") {
      train_df <- master_df %>% filter(partition == "train", statut == "domestique")
      val_df   <- master_df %>% filter(partition == "test",  statut == "domestique")
    } else {  # Test3bisW
      train_df <- master_df %>% filter(partition == "train", statut == "sauvage")
      val_df   <- master_df %>% filter(partition == "test",  statut == "sauvage")
    }
    cat("  train :", nrow(train_df), "| val :", nrow(val_df), "\n")

    nom_base   <- sprintf("%s_rep%d_seed%d_ep%d",
                          modele_type, r, seed_dom, epochs)
    fichier_h5 <- file.path(dossier_modeles, paste0(nom_base, ".h5"))
    fichier_rda<- file.path(dossier_modeles, paste0(nom_base, ".rda"))

    if (file.exists(fichier_h5) && !reecrire_modeles) {
      cat("  (modele existant, on le recharge)\n")
      model <- load_model_hdf5(fichier_h5, compile = FALSE)
      hist  <- NULL
    } else {
      k_clear_session()
      try(tensorflow::set_random_seed(seed_dom), silent = TRUE)
      ent <- entrainer_modele(train_df, val_df, seed_dom, verbose = 2)
      model <- ent$model
      hist  <- as.data.frame(ent$history$metrics)
      tryCatch(save_model_hdf5(model, fichier_h5),
               error = function(e)
                 message("Modele non sauvegarde : ", conditionMessage(e)))
    }

    ev_ext  <- evaluer_externe(model, ext_df)
    ev_arch <- evaluer_archeo(model, archeo_3, archeo_o)

    if (!is.null(ev_ext)) {
      cat("  [external] accuracy globale =", round(ev_ext$accuracy_globale, 3),
          "| par statut :\n"); print(round(ev_ext$acc_par_statut, 3))
    }
    if (!is.null(ev_arch)) {
      cat("  [archeo]   accuracy globale =", round(ev_arch$accuracy_globale, 3),
          "| par espece :\n"); print(round(ev_arch$acc_par_espece, 3))
    }

    save(master_df, ev_ext, ev_arch, hist,
         file = fichier_rda)

    res_rep[[modele_type]] <- list(ev_ext = ev_ext, ev_arch = ev_arch,
                                   fichier_h5 = fichier_h5)
  }

  resultats_par_rep[[r]] <- res_rep
}


# ---------------------------------------------------------------------
# 6. AGREGATION SUR LES 5 REPETITIONS
# ---------------------------------------------------------------------
statuts <- c("domestique", "sauvage")
especes <- classes_modele
esp_stat_lbl <- as.vector(t(outer(especes, statuts, paste, sep = "|")))

agreger <- function(modele_type) {
  # recuperer les evals pour ce type de modele sur toutes les reps
  evs_ext  <- lapply(resultats_par_rep, function(r) r[[modele_type]]$ev_ext)
  evs_arch <- lapply(resultats_par_rep, function(r) r[[modele_type]]$ev_arch)

  # --- externe : accuracy globale
  acc_glob <- sapply(evs_ext, `[[`, "accuracy_globale")

  # --- externe : par statut
  acc_stat <- sapply(evs_ext, function(e) e$acc_par_statut[statuts])
  rownames(acc_stat) <- statuts

  # --- externe : par espece
  acc_esp <- sapply(evs_ext, function(e) e$acc_par_espece[especes])
  rownames(acc_esp) <- especes

  # --- externe : par espece x statut
  acc_esp_stat <- sapply(evs_ext, function(e) e$acc_espece_statut[esp_stat_lbl])
  rownames(acc_esp_stat) <- esp_stat_lbl

  # --- externe : matrice de confusion cumulee
  conf_ext_sum <- Reduce(`+`, lapply(evs_ext, `[[`, "conf"))

  # --- archeo : accuracy par espece (peut contenir NA si espece absente)
  acc_arch <- sapply(evs_arch, function(e) {
    out <- rep(NA_real_, length(especes)); names(out) <- especes
    if (!is.null(e)) out[names(e$acc_par_espece)] <- e$acc_par_espece
    out
  })
  rownames(acc_arch) <- especes

  # --- archeo : matrice 9x3 cumulee (que les 3 especes x archeo)
  conf_arch_sum <- Reduce(`+`, lapply(evs_arch, `[[`, "conf_archeo"))

  list(
    acc_globale         = c(mean = mean(acc_glob), sd = sd(acc_glob)),
    acc_par_statut      = data.frame(statut = statuts,
                                     mean = rowMeans(acc_stat),
                                     sd   = apply(acc_stat, 1, sd)),
    acc_par_espece      = data.frame(espece = especes,
                                     mean = rowMeans(acc_esp),
                                     sd   = apply(acc_esp, 1, sd)),
    acc_espece_statut   = data.frame(combinaison = esp_stat_lbl,
                                     mean = rowMeans(acc_esp_stat, na.rm = TRUE),
                                     sd   = apply(acc_esp_stat, 1, sd, na.rm = TRUE),
                                     n_na = rowSums(is.na(acc_esp_stat))),
    acc_archeo_espece   = data.frame(espece = especes,
                                     mean = rowMeans(acc_arch, na.rm = TRUE),
                                     sd   = apply(acc_arch, 1, sd, na.rm = TRUE),
                                     n_na = rowSums(is.na(acc_arch))),
    conf_externe_cumulee = conf_ext_sum,
    conf_archeo_cumulee  = conf_arch_sum,
    acc_globale_vec      = acc_glob)
}

agregats <- list(
  Test3     = agreger("Test3"),
  Test3bisD = agreger("Test3bisD"),
  Test3bisW = agreger("Test3bisW"))


# ---------------------------------------------------------------------
# 7. AFFICHAGE RECAPITULATIF
# ---------------------------------------------------------------------
cat("\n\n########################################################\n")
cat("RESUME SUR", N_REPS, "REPETITIONS\n")
cat("########################################################\n")

cat("\n== Accuracy externe globale (dom + wild confondus) ==\n")
tab_glob <- data.frame(
  modele = names(agregats),
  mean   = sapply(agregats, function(a) a$acc_globale[["mean"]]),
  sd     = sapply(agregats, function(a) a$acc_globale[["sd"]]),
  row.names = NULL)
print(tab_glob, row.names = FALSE, digits = 3)

cat("\n== Accuracy externe par statut (moyenne sur", N_REPS, "repetitions) ==\n")
for (m in names(agregats)) {
  cat("\n  Modele", m, ":\n")
  print(agregats[[m]]$acc_par_statut, row.names = FALSE, digits = 3)
}

cat("\n== Accuracy externe par espece ==\n")
for (m in names(agregats)) {
  cat("\n  Modele", m, ":\n")
  print(agregats[[m]]$acc_par_espece, row.names = FALSE, digits = 3)
}

cat("\n== Accuracy externe par espece x statut ==\n")
for (m in names(agregats)) {
  cat("\n  Modele", m, ":\n")
  print(agregats[[m]]$acc_espece_statut, row.names = FALSE, digits = 3)
}

cat("\n== Accuracy archeo par espece (moyenne sur", N_REPS, "repetitions) ==\n")
for (m in names(agregats)) {
  cat("\n  Modele", m, ":\n")
  print(agregats[[m]]$acc_archeo_espece, row.names = FALSE, digits = 3)
}

cat("\n== Matrices 9x3 cumulees sur les", N_REPS, "repetitions ==\n")
for (m in names(agregats)) {
  cat("\n  Modele", m, ":\n")
  print(agregats[[m]]$conf_archeo_cumulee)
}


# ---------------------------------------------------------------------
# 8. SAUVEGARDE FINALE
# ---------------------------------------------------------------------
resultats_final <- list(
  resultats_par_rep = resultats_par_rep,
  agregats          = agregats,
  archeo_3          = archeo_3,
  config = list(N_REPS = N_REPS, seeds_dom = seeds_dom,
                seeds_wild = seeds_wild, epochs = epochs,
                batch_size = batch_size, input_shape = input_shape,
                N_dom_model = N_dom_model, N_dom_ext = N_dom_ext,
                N_wild_model = N_wild_model, N_wild_ext = N_wild_ext,
                prop_train = prop_train,
                duree = Sys.time() - begin_time))

fichier_final <- file.path(dossier_sortie, sprintf(
  "Multi_seed_resultats_Nrep%d_ep%d_%s.rda",
  N_REPS, epochs, format(Sys.time(), "%Y%m%d_%H%M")))
save(resultats_final, file = fichier_final)

cat("\nResultats finaux ecrits dans :",
    normalizePath(fichier_final), "\n")
cat("Modeles .h5 et .rda dans      :",
    normalizePath(dossier_modeles), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")


# ---------------------------------------------------------------------
# 9. TEST5 -- A FAIRE PLUS TARD (ne pas implementer ici)
# ---------------------------------------------------------------------
# Schema prevu pour Test5.R :
#   3 modeles entraines separement (Sauvage / Dom / Dom+Sauv) puis
#   evalues sur Sauvage + Dom + Archeo + Autre, avec une classe de
#   rejet "autre" (seuil de confiance sur p_max). La structure de ce
#   script peut etre reutilisee en remplacant evaluer_archeo() par
#   une evaluation avec seuil (cf. Eval_modele.R, section 7.1) et
#   en considerant l'AUC OOD / F1 "autre".
# =====================================================================