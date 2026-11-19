# =====================================================================
# Multi_seed_Test5.R
# Campagne de repetitions pour Test5 (VGG19 gele + rejet par seuil).
# 
# Pour chaque repetition (5 graines differentes) :
#   - Construction de la partition (identique a Test3/Test3_bis)
#   - Entrainement de 3 modeles :
#       * Test5     : domestiques + sauvages
#       * Test5bisD : domestiques seuls
#       * Test5bisW : sauvages seuls
#   - Calibration d'un SEUIL DE REJET sur la partition de validation
#     (quantile 5% des p_max) -> AUCUNE FUITE.
#   - Evaluation sur :
#       1) External (dom + wild)
#       2) Archeo (3 especes)
#       3) OOD ("autres" especes modernes)
#     avec application du rejet ("autre" si p_max < seuil).
#
# Sorties : .h5, .rda par modele/repetition + .rda agrege.
# Commentaires sans accents.
# =====================================================================

# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine            <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db        <- "rda/db"
dossier_rda_ajout <- "rda/db_ajout"
dossier_sortie    <- "rda/results/multi_seed_Test5"
dossier_modeles   <- file.path(dossier_sortie, "modeles")

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")
col_image <- "VD_deep"
statut_domestique <- "c"
statut_sauvage    <- "w"

# Effectifs (identiques a Test3 pour comparabilite)
N_dom_model   <- 530
N_dom_ext     <- 100
N_wild_model  <- 530
N_wild_ext    <- 100
prop_train    <- 4/5

# Parametres d'entrainement et de campagne
N_REPS      <- 5
epochs      <- 100
batch_size  <- 32
input_shape <- c(120, 90, 3)

seeds_dom  <- 2329 + 2 * (0:(N_REPS - 1))
seeds_wild <- seeds_dom + 1

# Seuil de rejet cible (5% de faux rejets en validation)
quantile_seuil <- 0.05

rda_archeo <- file.path(dossier_rda_ajout, "archeo_complet_db.rda")
reecrire_modeles <- TRUE


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
dir.create(dossier_modeles, recursive = TRUE, showWarnings = FALSE)


# ---------------------------------------------------------------------
# 2. CHARGEMENT DES DONNEES MODERNES
# ---------------------------------------------------------------------
charger_taxon <- function(taxon, dossier_img) {
  fichiers <- list.files(dossier_db, pattern = taxon,
                         full.names = TRUE, ignore.case = TRUE)
  if (length(fichiers) == 0) stop("Aucun .rda pour ", taxon)
  prefere <- fichiers[tolower(basename(fichiers)) == paste0(taxon, "_db2.rda")]
  fichier <- if (length(prefere) > 0) prefere[1] else fichiers[1]

  env_tmp <- new.env()
  noms    <- load(fichier, envir = env_tmp)
  db      <- get(noms[1], envir = env_tmp)
  db <- db[as.character(db$status) %in% c(statut_domestique, statut_sauvage), ]

  chemins <- as.character(db[[col_image]])
  if (!all(file.exists(chemins))) {
    tous  <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    idx <- match(basename(chemins), basename(tous))
    chemins <- tous[idx]
  }
  tibble(taxon = taxon, statut = as.character(db$status), path = chemins)
}

tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]]))
)
classes_modele <- sort(unique(tous_les_taxons$taxon))
classes_pred   <- c(classes_modele, "autre") # Niveaux pour les facteurs


# ---------------------------------------------------------------------
# 3. CHARGEMENT ARCHEO & OOD
# ---------------------------------------------------------------------
if (!file.exists(rda_archeo)) stop("Fichier absent : ", rda_archeo)
env_a <- new.env()
load(rda_archeo, envir = env_a)
obj <- get(ls(env_a)[1], envir = env_a)

# Correction : recuperation du data.frame interne
master_archeo <- if(is.data.frame(obj)) obj else obj$master_df

archeo_3 <- master_archeo %>%
  filter(class %in% classes_modele) %>%
  mutate(statut = "archeo") %>%
  select(path, class, statut)

archeo_o <- master_archeo %>%
  filter(!(class %in% classes_modele)) %>%
  select(path, class, statut)

# Verif chemins
archeo_3 <- archeo_3[file.exists(archeo_3$path), ]
if (nrow(archeo_o) > 0) archeo_o <- archeo_o[file.exists(archeo_o$path), ]


# ---------------------------------------------------------------------
# 4. FONCTIONS UTILES
# ---------------------------------------------------------------------

# Copie de la partition Test3
construire_partitions <- function(seed_dom, seed_wild) {
  N_train_d <- round(N_dom_model * prop_train)
  labs_dom  <- c(rep("train", N_train_d), rep("test", N_dom_model - N_train_d), rep("external", N_dom_ext))
  set.seed(seed_dom)
  shuffled_dom <- sample(labs_dom)
  df_dom <- tous_les_taxons %>% filter(statut == statut_domestique) %>%
    group_by(taxon) %>% slice_sample(n = length(labs_dom)) %>%
    mutate(partition = shuffled_dom) %>% ungroup()

  N_train_w <- round(N_wild_model * prop_train)
  labs_wild <- c(rep("train", N_train_w), rep("test", N_wild_model - N_train_w), rep("external", N_wild_ext))
  set.seed(seed_wild)
  shuffled_wild <- sample(labs_wild)
  df_wild <- tous_les_taxons %>% filter(statut == statut_sauvage) %>%
    group_by(taxon) %>% slice_sample(n = length(labs_wild)) %>%
    mutate(partition = shuffled_wild) %>% ungroup()

  lib_statut <- setNames(c("domestique", "sauvage"), c(statut_domestique, statut_sauvage))
  bind_rows(df_dom, df_wild) %>%
    transmute(partition, class = taxon, statut = unname(lib_statut[statut]), path)
}

# Entrainement
entrainer_modele <- function(train_df, val_df, seed, verbose = 1) {
  datagen <- image_data_generator(rescale = 1/255)
  flux <- function(x, shuffle = TRUE) {
    flow_images_from_dataframe(
      x, directory = "", x_col = "path", y_col = "class",
      generator = datagen, target_size = input_shape[1:2],
      class_mode = "categorical", batch_size = batch_size,
      shuffle = shuffle, seed = seed)
  }
  base_model <- application_vgg19(weights = "imagenet", include_top = FALSE, input_shape = input_shape)
  predictions <- base_model$output %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dense(units = 3,   activation = "softmax")
  
  model <- keras_model(inputs = base_model$input, outputs = predictions)
  freeze_weights(base_model)
  model %>% compile(optimizer = optimizer_rmsprop(learning_rate = 0.01),
                    loss = "categorical_crossentropy", metrics = c("accuracy"))
  
  cbs <- list(
    callback_early_stopping(monitor = "accuracy", patience = 20),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 10, min_lr = 1e-7))
  
  hist <- model %>% fit(
    flux(train_df), steps_per_epoch = ceiling(nrow(train_df) / batch_size),
    epochs = epochs, validation_data = flux(val_df),
    validation_steps = ceiling(nrow(val_df) / batch_size),
    callbacks = cbs, verbose = verbose)
  
  list(model = model, history = hist)
}

# Calibration et Evaluation avec rejet
calibrer_et_evaluer <- function(model, val_df, eval_dfs, seuil_q = 0.05) {
  datagen <- image_data_generator(rescale = 1/255)
  
  # 1. Calibration sur validation
  gen_val <- flow_images_from_dataframe(
    as.data.frame(val_df), directory = "", x_col = "path", y_col = "class",
    generator = datagen, target_size = input_shape[1:2],
    class_mode = NULL, batch_size = batch_size, shuffle = FALSE)
  p_val <- predict(model, gen_val, steps = ceiling(nrow(val_df)/batch_size), verbose = 0)
  p_val <- p_val[seq_len(nrow(val_df)), , drop=FALSE]
  pmax_val <- apply(p_val, 1, max)
  
  seuil <- as.numeric(quantile(pmax_val, seuil_q))
  
  # Fonction interne pour evaluer un jeu
  eval_un_jeu <- function(df, nom_jeu) {
    if (is.null(df) || nrow(df) == 0) return(NULL)
    gen <- flow_images_from_dataframe(
      as.data.frame(df), directory = "", x_col = "path", y_col = "class",
      generator = datagen, target_size = input_shape[1:2],
      class_mode = NULL, batch_size = batch_size, shuffle = FALSE)
    
    p <- predict(model, gen, steps = ceiling(nrow(df)/batch_size), verbose = 0)
    p <- p[seq_len(nrow(df)), , drop=FALSE]
    
    # Associer les predictions aux lignes (shuffle = FALSE)
    idx <- match(gen$filenames, df$path)
    if(anyNA(idx)) idx <- 1:nrow(df)
    p <- p[idx, , drop=FALSE]
    df_eval <- df[idx, ]
    
    pmax <- apply(p, 1, max)
    argmax <- classes_modele[max.col(p, ties.method = "first")]
    
    # Application du rejet
    pred <- ifelse(pmax < seuil, "autre", argmax)
    
    # Facteurs (avec niveaux fixes pour eviter le bug table())
    vraie_ligne <- if(nom_jeu == "external") paste0(df_eval$class, "|", df_eval$statut) else paste0(df_eval$class, "|", nom_jeu)
    niveaux_lignes <- if(nom_jeu == "external") {
       as.vector(t(outer(classes_modele, c("domestique", "sauvage"), paste, sep="|")))
    } else if (nom_jeu == "archeo") {
       paste0(classes_modele, "|archeo")
    } else { unique(vraie_ligne) }
    
    fact_vraie <- factor(vraie_ligne, levels = niveaux_lignes)
    fact_pred  <- factor(pred, levels = classes_pred)
    
    conf <- table(vraie = fact_vraie, predite = fact_pred)
    
    # Metriques cles
    correct_global <- mean(pred == df_eval$class) # "autre" predit sur une vraie classe = faux
    taux_rejet     <- mean(pred == "autre")
    
    list(conf = conf, correct_global = correct_global, taux_rejet = taux_rejet, 
         pmax = pmax, pred = pred, df = df_eval)
  }
  
  list(
    seuil = seuil,
    ext   = eval_un_jeu(eval_dfs$external, "external"),
    arch  = eval_un_jeu(eval_dfs$archeo, "archeo"),
    ood   = eval_un_jeu(eval_dfs$ood, "ood")
  )
}


# ---------------------------------------------------------------------
# 5. BOUCLE PRINCIPALE (MULTI-SEED)
# ---------------------------------------------------------------------
resultats_par_rep <- list()

for (r in seq_len(N_REPS)) {
  seed_dom  <- seeds_dom[r]
  seed_wild <- seeds_wild[r]
  cat("\n=======================================================\n")
  cat("REPETITION", r, "/", N_REPS, "(seed_dom:", seed_dom, ")\n")
  cat("=======================================================\n")

  master_df <- construire_partitions(seed_dom, seed_wild)
  ext_df    <- master_df %>% filter(partition == "external")
  
  eval_dfs <- list(external = ext_df, archeo = archeo_3, ood = archeo_o)
  
  res_rep <- list()

  for (modele_type in c("Test5", "Test5bisD", "Test5bisW")) {
    cat("\n--- Modele :", modele_type, "---\n")

    if (modele_type == "Test5") {
      train_df <- master_df %>% filter(partition == "train")
      val_df   <- master_df %>% filter(partition == "test")
    } else if (modele_type == "Test5bisD") {
      train_df <- master_df %>% filter(partition == "train", statut == "domestique")
      val_df   <- master_df %>% filter(partition == "test",  statut == "domestique")
    } else {
      train_df <- master_df %>% filter(partition == "train", statut == "sauvage")
      val_df   <- master_df %>% filter(partition == "test",  statut == "sauvage")
    }

    nom_base   <- sprintf("%s_rep%d_seed%d", modele_type, r, seed_dom)
    fichier_h5 <- file.path(dossier_modeles, paste0(nom_base, ".h5"))
    
    if (file.exists(fichier_h5) && !reecrire_modeles) {
      model <- load_model_hdf5(fichier_h5, compile = FALSE)
      hist <- NULL
    } else {
      k_clear_session()
      try(tensorflow::set_random_seed(seed_dom), silent = TRUE)
      ent <- entrainer_modele(train_df, val_df, seed_dom)
      model <- ent$model
      hist <- as.data.frame(ent$history$metrics)
      tryCatch(save_model_hdf5(model, fichier_h5), error = function(e) NULL)
    }

    # Calibration + Evaluation (External / Archeo / OOD)
    ev <- calibrer_et_evaluer(model, val_df, eval_dfs, seuil_q = quantile_seuil)
    
    cat("  Seuil calibre :", round(ev$seuil, 3), "\n")
    if(!is.null(ev$ext))  cat("  [External] Acc (avec rejet) :", round(ev$ext$correct_global, 3), "| Rejets :", round(ev$ext$taux_rejet, 3), "\n")
    if(!is.null(ev$arch)) cat("  [Archeo]   Acc (avec rejet) :", round(ev$arch$correct_global, 3), "| Rejets :", round(ev$arch$taux_rejet, 3), "\n")
    if(!is.null(ev$ood))  cat("  [OOD]      Taux de rejet    :", round(ev$ood$taux_rejet, 3), "\n")

    res_rep[[modele_type]] <- list(
      hist = hist, seuil = ev$seuil, ext = ev$ext, arch = ev$arch, ood = ev$ood
    )
  }
  resultats_par_rep[[r]] <- res_rep
}


# ---------------------------------------------------------------------
# 6. AGREGATION DES RESULTATS
# ---------------------------------------------------------------------
agreger <- function(modele_type) {
  reps <- lapply(resultats_par_rep, `[[`, modele_type)
  
  seuils <- sapply(reps, `[[`, "seuil")
  
  # Fonction pour extraire une metrique sur les 3 jeux (avec gestion des NA)
  get_metric <- function(jeu, metric) {
    sapply(reps, function(x) if(is.null(x[[jeu]])) NA else x[[jeu]][[metric]])
  }
  
  # Matrices de confusion cumulees
  sum_conf <- function(jeu) {
    confs <- lapply(reps, function(x) if(is.null(x[[jeu]])) NULL else x[[jeu]]$conf)
    confs <- confs[!sapply(confs, is.null)]
    if(length(confs)==0) return(NULL)
    Reduce(`+`, confs)
  }

  list(
    seuil_mean = mean(seuils),
    seuil_sd   = sd(seuils),
    
    ext_acc_mean  = mean(get_metric("ext", "correct_global"), na.rm=TRUE),
    ext_acc_sd    = sd(get_metric("ext", "correct_global"), na.rm=TRUE),
    ext_rejet_mean= mean(get_metric("ext", "taux_rejet"), na.rm=TRUE),
    ext_conf      = sum_conf("ext"),
    
    arch_acc_mean = mean(get_metric("arch", "correct_global"), na.rm=TRUE),
    arch_acc_sd   = sd(get_metric("arch", "correct_global"), na.rm=TRUE),
    arch_rejet_mean=mean(get_metric("arch", "taux_rejet"), na.rm=TRUE),
    arch_conf     = sum_conf("arch"),
    
    ood_rejet_mean= mean(get_metric("ood", "taux_rejet"), na.rm=TRUE),
    ood_rejet_sd  = sd(get_metric("ood", "taux_rejet"), na.rm=TRUE),
    ood_conf      = sum_conf("ood")
  )
}

agregats <- list(
  Test5     = agreger("Test5"),
  Test5bisD = agreger("Test5bisD"),
  Test5bisW = agreger("Test5bisW")
)


# ---------------------------------------------------------------------
# 7. AFFICHAGE DES RESULTATS
# ---------------------------------------------------------------------
cat("\n\n########################################################\n")
cat("RESUME SUR", N_REPS, "REPETITIONS (Moyenne +/- Ecart-type)\n")
cat("########################################################\n")

cat("\n== Seuil calibre moyen ==\n")
print(sapply(agregats, function(a) sprintf("%.3f (%.3f)", a$seuil_mean, a$seuil_sd)))

cat("\n== Performance EXTERNE (Domestiques + Sauvages) ==\n")
for(m in names(agregats)) {
  cat("\n  ", m, ":\n")
  cat("    Accuracy globale (avec rejet) :", sprintf("%.3f (%.3f)", agregats[[m]]$ext_acc_mean, agregats[[m]]$ext_acc_sd), "\n")
  cat("    Taux de rejet moyen           :", sprintf("%.3f", agregats[[m]]$ext_rejet_mean), "\n")
}

cat("\n== Performance ARCHEO ==\n")
for(m in names(agregats)) {
  cat("\n  ", m, ":\n")
  cat("    Accuracy globale (avec rejet) :", sprintf("%.3f (%.3f)", agregats[[m]]$arch_acc_mean, agregats[[m]]$arch_acc_sd), "\n")
  cat("    Taux de rejet moyen           :", sprintf("%.3f", agregats[[m]]$arch_rejet_mean), "\n")
}

cat("\n== Capacite de detection OOD (Autres especes) ==\n")
cat("(Objectif : taux de rejet eleve pour les especes inconnues)\n")
for(m in names(agregats)) {
  cat("\n  ", m, ":\n")
  cat("    Taux de rejet OOD moyen       :", sprintf("%.3f (%.3f)", agregats[[m]]$ood_rejet_mean, agregats[[m]]$ood_rejet_sd), "\n")
}

cat("\n== Matrices de confusion cumulees sur", N_REPS, "repetitions ==\n")
for(m in names(agregats)) {
  cat("\n--- Modele", m, "---\n")
  cat("Externe (lignes : espece|statut ; colonnes : prediction + 'autre')\n")
  print(agregats[[m]]$ext_conf)
  cat("\nArcheo (lignes : espece|archeo ; colonnes : prediction + 'autre')\n")
  print(agregats[[m]]$arch_conf)
  cat("\nOOD (lignes : classe reelle ; colonnes : prediction + 'autre')\n")
  if(!is.null(agregats[[m]]$ood_conf)) print(agregats[[m]]$ood_conf)
}

# ---------------------------------------------------------------------
# 8. SAUVEGARDE
# ---------------------------------------------------------------------
resultats_final <- list(
  resultats_par_rep = resultats_par_rep,
  agregats          = agregats,
  config = list(N_REPS = N_REPS, seeds_dom = seeds_dom,
                quantile_seuil = quantile_seuil, epochs = epochs)
)

fichier_final <- file.path(dossier_sortie, sprintf("Multi_seed_Test5_Nrep%d_%s.rda", N_REPS, format(Sys.time(), "%Y%m%d_%H%M")))
save(resultats_final, file = fichier_final)

cat("\nResultats sauvegardes dans :", normalizePath(fichier_final), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")