# Ce script aborde la reconnaissance des 3 espèces (Olea, Phoenix, Vitis) avec
# un VGG19 gelé, entraîné sur domestiques ("c") et sauvages ("w"), avec la
# même partition que Test3.R. La nouveauté de Test5 est l'introduction d'une
# classe de rejet "autre" obtenue par un SEUIL SUR LA PROBABILITÉ MAXIMALE
# p_max. Le modèle conserve 3 sorties ; le rejet est appliqué à la prédiction,
# il n'est pas appris.
#
# Différences avec Test3.R :
#   - calcul d'un seuil de rejet sur la partition "test" (=validation) du
#     modèle (pas de fuite : external + archéo + autres ne servent jamais à
#     choisir le seuil)
#   - évaluation sur external (dom + sauv) + archéo (3 espèces) + "autres"
#     (autres espèces modernes), avec une matrice 10x4 (lignes = espèce|statut ;
#     colonnes = 3 espèces + "autre")
#   - calcul de la matrice sans rejet en parallèle, pour comparaison


# Paramètres généraux
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db  <- "rda/db"
dossier_ajout <- "rda/db_ajout"
dossier_res <- "rda/results/test5"

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")

col_image <- "VD_deep"
statut_domestique <- "c"
statut_sauvage    <- "w"

# Effectifs identiques à Test3.R (comparabilité des jeux external)
N_dom_model   <- 530
N_dom_ext     <- 100
N_wild_model  <- 530
N_wild_ext    <- 100

prop_train  <- 4/5
epochs      <- 100
seed        <- 2329
seed_wild   <- seed + 1
batch_size  <- 32
input_shape <- c(120, 90, 3)

# Seuil : on garde 95 % des images de validation in-distribution
# (taux de faux rejets cible = 5 %). Voir plus bas pour la justification.
quantile_seuil <- 0.05
# Seuils secondaires pour analyse de sensibilité
quantiles_sensibilite <- c(0.05, 0.10)
seuil_fixe_sensibilite <- 0.5


# Chargement des packages et préparation du dossier de travail
library(tidyverse)
library(keras)
py_require_legacy_keras()
library(tensorflow)

setwd(racine)
cat("Dossier de travail :", getwd(), "\n")
begin_time <- Sys.time()


# Chargement d'un taxon (identique à Test3.R)
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
  db <- db[as.character(db$status) %in% c(statut_domestique, statut_sauvage), ]

  chemins <- as.character(db[[col_image]])
  existe <- file.exists(chemins)
  if (!all(existe)) {
    tous <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    noms_disque <- basename(tous)
    if (anyDuplicated(noms_disque) > 0) stop("Noms dupliques dans ", dossier_img)
    idx <- match(basename(chemins), noms_disque)
    if (any(is.na(idx))) stop(sum(is.na(idx)), " images introuvables")
    chemins <- tous[idx]
  }
  tibble(taxon = taxon, statut = as.character(db$status), path = chemins)
}

tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]]))
)
cat("\nGraines disponibles (taxon x statut) :\n")
print(table(tous_les_taxons$taxon, tous_les_taxons$statut))


# Tirage et partition (identique à Test3.R)
N_train_d <- round(N_dom_model * prop_train)
N_test_d  <- N_dom_model - N_train_d
labs_dom  <- c(rep("train", N_train_d), rep("test", N_test_d),
               rep("external", N_dom_ext))
set.seed(seed)
shuffled_dom <- sample(labs_dom)

dom <- tous_les_taxons %>% filter(statut == statut_domestique)
df_dom <- dom %>%
  group_by(taxon) %>%
  sample_n(length(labs_dom)) %>%
  mutate(partition = shuffled_dom) %>%
  ungroup()

N_train_w <- round(N_wild_model * prop_train)
N_test_w  <- N_wild_model - N_train_w
labs_wild <- c(rep("train", N_train_w), rep("test", N_test_w),
               rep("external", N_wild_ext))
set.seed(seed_wild)
shuffled_wild <- sample(labs_wild)

wild <- tous_les_taxons %>% filter(statut == statut_sauvage)
df_wild <- wild %>%
  group_by(taxon) %>%
  sample_n(length(labs_wild)) %>%
  mutate(partition = shuffled_wild) %>%
  ungroup()

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


# Générateurs (mode catégoriel pour l'entraînement, NULL pour l'évaluation)
all_datagen <- image_data_generator(rescale = 1/255)

flux <- function(x, melanger = TRUE, mode = "categorical") {
  flow_images_from_dataframe(
    x, directory = "", x_col = "path", y_col = "class",
    generator = all_datagen, target_size = input_shape[1:2],
    class_mode = mode, batch_size = batch_size,
    shuffle = melanger, seed = seed)
}

train_generator <- flux(dfs$train)
test_generator  <- flux(dfs$test)
cat("\nCodage des classes (ordre alphabetique) :\n")
print(train_generator$class_indices)


# Modèle (identique à Test3.R)
try(tensorflow::set_random_seed(seed), silent = TRUE)
base_model <- application_vgg19(weights = "imagenet", include_top = FALSE,
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


# Entraînement du modèle
callbacks_list <- list(
  callback_early_stopping(monitor = "accuracy", patience = 20),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 1e-7)
)
history <- model %>% fit(
  train_generator,
  steps_per_epoch  = ceiling(nrow(dfs$train) / batch_size),
  epochs           = epochs,
  validation_data  = test_generator,
  validation_steps = ceiling(nrow(dfs$test)  / batch_size),
  callbacks        = callbacks_list
)
plot(history)


# Calibration du seuil de rejet (sur validation, sans fuite).
# Le seuil est déterminé sur dfs$test (=validation) uniquement :
#   - ce jeu a servi à l'early stopping, mais n'est PAS utilisé pour
#     l'évaluation finale (qui se fait sur external + archéo + autres)
#   - on rejette les 5 % de validation les moins confiants ; cela revient
#     à fixer un taux cible de faux rejets de 5 % en in-distribution
# Alternative non retenue : calibrer sur archéo/autres = fuite.

val_gen  <- flux(dfs$test, melanger = FALSE)
n_val    <- nrow(dfs$test)
proba_v  <- predict(model, val_gen, steps = ceiling(n_val / batch_size),
                    verbose = 0)
proba_v  <- proba_v[seq_len(n_val), , drop = FALSE]
pmax_v   <- apply(proba_v, 1, max)

seuil      <- as.numeric(quantile(pmax_v, quantile_seuil))
seuil_q10  <- as.numeric(quantile(pmax_v, 0.10))
seuil_fixe <- seuil_fixe_sensibilite

cat("\n[Calibration du seuil de rejet sur validation]\n")
cat("n_validation =", n_val, "\n")
cat("p_max : min=", round(min(pmax_v), 3),
    " median=", round(median(pmax_v), 3),
    " max=", round(max(pmax_v), 3), "\n")
cat("Seuil retenu (q", round(100*quantile_seuil, 0), ") :", round(seuil, 4), "\n")
cat("Seuil q10 (sensibilite) :", round(seuil_q10, 4), "\n")
cat("Seuil fixe 0.5 (sensibilite) :", seuil_fixe, "\n")


# Chargement des données archéo + autres.
# archeo_complet_db.rda regroupe : archéo des 3 espèces (statut="archeo")
# + autres espèces modernes (class="autre", statut="autre_espece").
# Effectifs très limités côté archéo (~23 images) : taux descriptifs.
fichier_archeo <- file.path(dossier_ajout, "archeo_complet_db.rda")
if (!file.exists(fichier_archeo)) stop("Fichier absent : ", fichier_archeo)

env_a <- new.env()
noms_a <- load(fichier_archeo, envir = env_a)
archeo_df <- get(noms_a[1], envir = env_a)
necessaires <- c("partition", "class", "statut", "path")
if (!all(necessaires %in% names(archeo_df))) {
  stop("Colonnes manquantes dans archeo_complet_db : ",
       paste(setdiff(necessaires, names(archeo_df)), collapse = ", "))
}
archeo_df <- as_tibble(archeo_df[, necessaires])

# Correction des chemins : d'abord on essaie tel quel, sinon par basename.
if (!all(file.exists(archeo_df$path))) {
  cat("Correction des chemins archeo par nom de fichier dans DATA/...\n")
  tous  <- list.files("DATA", recursive = TRUE, full.names = TRUE)
  noms_disque <- basename(tous)
  if (anyDuplicated(noms_disque) > 0) {
    stop("Noms de fichiers dupliques dans DATA/ (archeo)")
  }
  idx <- match(basename(archeo_df$path), noms_disque)
  if (anyNA(idx)) {
    print(head(basename(archeo_df$path)[is.na(idx)]))
    stop(sum(is.na(idx)), " images archeo/autres introuvables")
  }
  archeo_df$path <- tous[idx]
}
archeo_df$partition <- as.character(archeo_df$partition)

cat("\nRepartition archeo + autres (class x statut) :\n")
print(table(archeo_df$class, archeo_df$statut, useNA = "ifany"))


# Évaluation externe (external + archéo + autres) avec rejet
ext_df   <- dfs$external[, c("partition", "class", "statut", "path")]
eval_df  <- bind_rows(ext_df, archeo_df)
n_eval   <- nrow(eval_df)

cat("\nEffectif du jeu d'evaluation :", n_eval,
    "(dont external =", nrow(ext_df), ", archeo + autres =",
    nrow(archeo_df), ")\n")

# Prédictions (class_mode = NULL car "autre" ne fait pas partie des 3 classes)
eval_gen <- flux(eval_df, melanger = FALSE, mode = NULL)
steps_e  <- ceiling(n_eval / batch_size)
proba_e  <- predict(model, eval_gen, steps = steps_e, verbose = 0)
proba_e  <- proba_e[seq_len(n_eval), , drop = FALSE]

pmax_e       <- apply(proba_e, 1, max)
noms_classes <- names(sort(unlist(train_generator$class_indices)))  # 3 especes
classe_argmax <- noms_classes[max.col(proba_e)]

# Association ligne <-> fichier (pas de classes dans un générateur mode NULL)
fichiers_lus <- eval_gen$filenames
if (length(fichiers_lus) < n_eval) {
  # Keras renvoie n_eval*batch_size noms si pas de classes ; on tronque.
  fichiers_lus <- fichiers_lus[seq_len(n_eval)]
}
idx <- match(fichiers_lus, eval_df$path)
if (anyNA(idx)) {
  warning("Association par nom impossible : ordre des lignes utilise.")
  idx <- seq_len(n_eval)
}
vraie_class  <- eval_df$class[idx]
vraie_statut <- eval_df$statut[idx]

# Cohérence des espèces sur les lignes non-"autre"
especes_connues <- c(noms_classes, "autre")
if (!all(vraie_class %in% especes_connues)) {
  stop("Classes inattendues dans le jeu d'evaluation : ",
       paste(unique(setdiff(vraie_class, especes_connues)), collapse = ", "))
}

# Application du rejet
predire_avec_rejet <- function(seuil_val, proba = proba_e,
                               pmax = pmax_e, argmax = classe_argmax) {
  ifelse(pmax < seuil_val, "autre", argmax)
}

# Matrice principale (seuil q05)
pred_final <- predire_avec_rejet(seuil)

niveaux_lignes <- c(
  as.vector(sapply(noms_classes, function(e) paste0(e, "|",
     c("domestique", "sauvage", "archeo")))),
  "autre|autre_espece"
)
niveaux_cols <- c(noms_classes, "autre")

ligne_f <- factor(paste0(vraie_class, "|", vraie_statut),
                  levels = niveaux_lignes)
pred_f  <- factor(pred_final, levels = niveaux_cols)
conf_rejet <- table(vraie = ligne_f, predite = pred_f)

cat("\n[Tableau principal - seuil q05 =", round(seuil, 3), "]\n")
cat("Lignes = espece vraie | statut ; colonnes = espece predite OU 'autre' (rejete)\n")
print(conf_rejet)

# Matrice sans rejet (référence : argmax toujours appliqué)
pred_f0   <- factor(classe_argmax, levels = niveaux_cols)
conf_sans <- table(vraie = ligne_f, predite = pred_f0)
cat("\n[Tableau sans rejet (argmax seul)]\n")
print(conf_sans)

# Indicateurs par ligne : rejet, accuracy parmi non-rejetés
calculer_stats <- function(conf, lignes, cols) {
  res <- tibble::tibble(
    ligne = character(), n = integer(),
    taux_rejet = numeric(), accuracy_si_non_rejete = numeric(),
    accuracy_avec_rejet = numeric()
  )
  for (l in lignes) {
    v <- as.numeric(conf[l, ])
    n_l <- sum(v)
    if (n_l == 0) next
    rej <- v[match("autre", cols)]
    # bonne classe = espece vraie (prefixe avant "|")
    espece <- strsplit(l, "|", fixed = TRUE)[[1]][1]
    bon <- if (espece == "autre") v[match("autre", cols)] else v[match(espece, cols)]
    non_rej <- n_l - rej
    res <- dplyr::bind_rows(res, tibble::tibble(
      ligne = l, n = n_l,
      taux_rejet = rej / n_l,
      accuracy_si_non_rejete = if (non_rej > 0) bon / non_rej else NA_real_,
      accuracy_avec_rejet = bon / n_l
    ))
  }
  res
}

stats_rejet <- calculer_stats(conf_rejet, niveaux_lignes, niveaux_cols)
cat("\n[Statistiques par ligne - seuil q05]\n")
print(as.data.frame(stats_rejet), digits = 3)

# Sensibilité au seuil
sensibilite <- lapply(
  c(q05 = seuil, q10 = seuil_q10, fixe_0.5 = seuil_fixe),
  function(s) {
    pf <- factor(predire_avec_rejet(s), levels = niveaux_cols)
    tb <- table(vraie = ligne_f, predite = pf)
    # Taux de rejet globaux par "grand groupe"
    rej_par_groupe <- sapply(niveaux_lignes, function(l) {
      n_l <- sum(tb[l, ]); if (n_l == 0) NA_real_
      else as.numeric(tb[l, "autre"]) / n_l
    })
    list(conf = tb, rejets = rej_par_groupe)
  }
)

cat("\n[Sensibilite au seuil - taux de rejet par ligne]\n")
tab_sens <- sapply(sensibilite, `[[`, "rejets")
print(round(tab_sens, 3))

# Indicateurs globaux (accuracy "espèce correcte") selon le mode
bonne_espece <- (pred_final == vraie_class) & (vraie_class != "autre")
acc_globale  <- mean(bonne_espece)
cat("\nAccuracy espece (external seulement, sans les archeo/autres) :",
    round(mean(pred_final[vraie_statut %in% c("domestique","sauvage")] ==
               vraie_class[vraie_statut %in% c("domestique","sauvage")]), 3), "\n")
cat("Taux de rejet global :", round(mean(pred_final == "autre"), 3), "\n")


# Sauvegarde des résultats
dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
nom_base <- sprintf("Test5_3especes_rejet_Nd%d_Nw%d_ep%d_seed%d_q%.2f",
                    N_dom_model, N_wild_model, epochs, seed, quantile_seuil)

resultats <- list(
  master_df         = master_df,
  archeo_df         = archeo_df,
  historique        = as.data.frame(history$metrics),
  seuil             = seuil,
  seuil_q10         = seuil_q10,
  seuil_fixe        = seuil_fixe,
  pmax_validation   = pmax_v,
  conf_rejet        = conf_rejet,
  conf_sans_rejet   = conf_sans,
  stats_rejet       = stats_rejet,
  sensibilite       = sensibilite,
  probas_eval       = proba_e,
  pred_final        = pred_final,
  vraie_class_eval  = vraie_class,
  vraie_statut_eval = vraie_statut,
  chemin_eval       = eval_df$path[idx],
  codage_classes    = unlist(train_generator$class_indices),
  config = list(taxons = taxons, N_dom_model = N_dom_model,
                N_dom_ext = N_dom_ext, N_wild_model = N_wild_model,
                N_wild_ext = N_wild_ext, epochs = epochs,
                seed = seed, seed_wild = seed_wild,
                col_image = col_image, quantile_seuil = quantile_seuil),
  duree = Sys.time() - begin_time
)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))

tryCatch(save_model_hdf5(model, file.path(dossier_res, paste0(nom_base, ".h5"))),
         error = function(e) message("Modele non sauvegarde : ", conditionMessage(e)))

cat("\nResultats ecrits dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")