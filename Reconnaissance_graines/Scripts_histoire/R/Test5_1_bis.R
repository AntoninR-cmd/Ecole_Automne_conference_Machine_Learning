# =====================================================================
# Test5.R
# Reconnaissance de 3 ESPECES (Olea, Phoenix, Vitis) avec une classe de
# REJET "autre" obtenue par SEUIL DE CONFIANCE.
#
# PRINCIPE
#   Le modele est le meme que dans Test2/Test3 : VGG19 gele + tete a
#   3 sorties softmax. Il n'a JAMAIS vu d'image "autre" : il ne connait
#   que 3 classes. Apres l'entrainement, on applique une regle de decision :
#
#       confiance = plus grande des 3 probabilites softmax
#       si confiance >= seuil  ->  l'espece la plus probable
#       si confiance <  seuil  ->  "autre"
#
#   ATTENTION : avec 3 sorties softmax, la plus grande probabilite est
#   TOUJOURS >= 1/3. Un seuil inferieur a 1/3 ne rejette jamais rien.
#
# DONNEES (vue dorsale VD uniquement)
#   - especes connues : les 3 .rda de l'article (domestiques "c" et
#     sauvages "w"), partitionnees en 4 : train / test / seuil / external
#       train    : entraine les poids de la tete
#       test     : validation (pilote le learning rate), comme avant
#       seuil    : sert a CHOISIR le seuil (avec des images "autre")
#       external : evaluation finale (avec d'autres images "autre")
#   - images "autre" : rda/db_ajout/autres_especes_db.rda (cree par
#     Creer_rda.R), reparties en "seuil" et "external". Elles ne servent
#     JAMAIS a l'entrainement.
#
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db  <- "rda/db"
rda_autres  <- "rda/db_ajout/autres_especes_db.rda"
dossier_res <- "rda/results/rejet_seuil"

taxons <- c(olea    = "DATA/OLEA",
            phoenix = "DATA/PHOENIX",
            vitis   = "DATA/VITIS")

col_image        <- "VD_deep"   # vue dorsale (rda de l'article)
col_image_autres <- "img_deep"  # colonne d'image du rda "autres especes"

statut_domestique <- "c"
statut_sauvage    <- "w"

# Statuts utilises pour ENTRAINER le modele.
#   c("c", "w") : domestiques + sauvages (comme Test3)
#   "c"         : domestiques seuls (comme Test2)
# Le seuil et l'evaluation finale utilisent TOUJOURS les deux statuts.
statuts_entrainement <- c("c", "w")

# Effectifs PAR ESPECE et PAR STATUT pour les images connues
effectifs <- c(train = 40, test = 10, seuil = 50, external = 100)

# Images "autre" (nombre de specimens si la colonne specimen est remplie,
# sinon nombre d'images) pour choisir le seuil et pour le test final
n_autres_seuil <- 100
n_autres_ext   <- 100

# CHOIX DU SEUIL (trois possibilites, par ordre de priorite) :
#  1. seuil_impose : valeur fixee a la main, par ex. 0.7
#  2. rejet_connu_max : on prend le seuil qui rejette le plus d'images
#     "autre" sans rejeter plus de cette part d'images connues
#     (par ex. 0.05 = 5 % de fausses alertes au maximum)
#  3. sinon : le seuil qui maximise  (taux d'"autre" rejetes) -
#     (taux d'images connues rejetees)   [indice de Youden]
seuil_impose    <- NULL
rejet_connu_max <- NULL
grille_seuils   <- seq(0.20, 0.99, by = 0.01)

epochs      <- 10
seed        <- 2329
seed_wild   <- seed + 1
seed_autres <- seed + 2
batch_size  <- 16
input_shape <- c(120, 90, 3)   # (hauteur, largeur, canaux)


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


# ---------------------------------------------------------------------
# 2. IMAGES DES ESPECES CONNUES (rda de l'article, statuts "c" et "w")
# ---------------------------------------------------------------------
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
  for (s in c(statut_domestique, statut_sauvage)) {
    if (!(s %in% as.character(db$status))) {
      stop("Aucune ligne avec status == '", s, "' pour ", taxon)
    }
  }
  db <- db[as.character(db$status) %in% c(statut_domestique, statut_sauvage), ]

  chemins <- as.character(db[[col_image]])
  existe  <- file.exists(chemins)
  cat("Fichiers trouves tels quels :", sum(existe), "/", length(existe), "\n")
  if (!all(existe)) {
    cat("Recherche par nom de fichier dans", dossier_img, "...\n")
    tous <- list.files(dossier_img, recursive = TRUE, full.names = TRUE)
    noms_disque <- basename(tous)
    if (anyDuplicated(noms_disque) > 0) stop("Noms dupliques dans ", dossier_img)
    idx <- match(basename(chemins), noms_disque)
    if (any(is.na(idx))) stop(sum(is.na(idx)), " images introuvables dans ", dossier_img)
    chemins <- tous[idx]
  }
  tibble(taxon = taxon, statut = as.character(db$status), path = chemins)
}

tous_les_taxons <- bind_rows(
  lapply(names(taxons), function(t) charger_taxon(t, taxons[[t]]))
)
cat("\nGraines connues disponibles (taxon x statut) :\n")
print(table(tous_les_taxons$taxon, tous_les_taxons$statut))


# ---------------------------------------------------------------------
# 3. IMAGES "AUTRE" (rda cree par Creer_rda.R)
# ---------------------------------------------------------------------
env_tmp <- new.env()
nom_obj <- load(rda_autres, envir = env_tmp)
autres  <- get(nom_obj[1], envir = env_tmp)
if (!(col_image_autres %in% names(autres))) {
  stop("Colonne '", col_image_autres, "' absente de ", rda_autres)
}
if ("lisible" %in% names(autres)) autres <- autres %>% filter(is.na(lisible) | lisible)
if (!all(file.exists(autres[[col_image_autres]]))) {
  stop("Certains chemins du .rda 'autres especes' n'existent pas : ",
       "verifier la racine du projet.")
}
cat("\nImages 'autre' disponibles :", nrow(autres), "\n")

# Cle de decoupage : le specimen s'il est renseigne, sinon l'image.
# (deux images d'un meme specimen ne doivent pas etre separees)
if (!("specimen" %in% names(autres))) autres$specimen <- NA_character_
autres <- autres %>%
  mutate(cle = ifelse(is.na(specimen), paste(folder, id, sep = "/"), specimen))

set.seed(seed_autres)
cles <- sample(unique(autres$cle))
if (length(cles) < 4) stop("Pas assez d'images 'autre' pour les separer en deux jeux.")
n_val      <- min(n_autres_seuil, floor(length(cles) / 2))
cles_seuil <- cles[seq_len(n_val)]
reste      <- cles[-seq_len(n_val)]
cles_ext   <- reste[seq_len(min(n_autres_ext, length(reste)))]

master_autres <- autres %>%
  mutate(partition = case_when(cle %in% cles_seuil ~ "seuil",
                               cle %in% cles_ext   ~ "external",
                               TRUE                ~ NA_character_)) %>%
  filter(!is.na(partition)) %>%
  transmute(partition,
            class  = "autre",
            statut = NA_character_,
            path   = as.character(.data[[col_image_autres]]))

cat("Images 'autre' retenues par partition :\n"); print(table(master_autres$partition))


# ---------------------------------------------------------------------
# 4. PARTITION DES IMAGES CONNUES (train / test / seuil / external)
# ---------------------------------------------------------------------
# Un tirage par statut (graines aleatoires distinctes), meme effectif
# pour chaque espece.
labs <- rep(names(effectifs), times = effectifs)

tirer <- function(df, graine) {
  if (any(table(df$taxon) < length(labs))) {
    print(table(df$taxon))
    stop("Pas assez d'images dans un taxon (besoin de ", length(labs), ").")
  }
  set.seed(graine)
  etiquettes <- sample(labs)
  df %>%
    group_by(taxon) %>%
    sample_n(length(labs)) %>%
    mutate(partition = etiquettes) %>%
    ungroup()
}

lib_statut <- setNames(c("domestique", "sauvage"),
                       c(statut_domestique, statut_sauvage))

df_dom  <- tirer(tous_les_taxons %>% filter(statut == statut_domestique), seed)
df_wild <- tirer(tous_les_taxons %>% filter(statut == statut_sauvage),    seed_wild)

master_connus <- bind_rows(df_dom, df_wild) %>%
  transmute(partition, class = taxon, statut = unname(lib_statut[statut]), path)

master_df <- bind_rows(master_connus, master_autres)

cat("\nPartition x classe x statut (images connues et 'autre') :\n")
print(ftable(table(partition = master_df$partition,
                   classe    = master_df$class,
                   statut    = master_df$statut, useNA = "ifany")))

statuts_train <- unname(lib_statut[statuts_entrainement])
train_df <- master_connus %>% filter(partition == "train", statut %in% statuts_train)
val_df   <- master_connus %>% filter(partition == "test",  statut %in% statuts_train)
cat("\nEntrainement :", nrow(train_df), "images | validation :", nrow(val_df), "\n")


# ---------------------------------------------------------------------
# 5. MODELE : VGG19 gele + tete a 3 sorties softmax (3 especes)
# ---------------------------------------------------------------------
datagen <- image_data_generator(rescale = 1/255)

flux_train <- function(x) {
  flow_images_from_dataframe(
    as.data.frame(x), directory = "", x_col = "path", y_col = "class",
    generator = datagen, target_size = input_shape[1:2],
    class_mode = "categorical", batch_size = batch_size,
    shuffle = TRUE, seed = seed)
}
train_generator <- flux_train(train_df)
val_generator   <- flux_train(val_df)

classes <- names(sort(unlist(train_generator$class_indices)))
cat("Classes du modele (ordre des sorties) :", paste(classes, collapse = " / "), "\n")

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

callbacks_list <- list(
  callback_early_stopping(monitor = "accuracy", patience = 20),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 1e-7)
)

history <- model %>% fit(
  train_generator,
  steps_per_epoch  = ceiling(nrow(train_df) / batch_size),
  epochs           = epochs,
  validation_data  = val_generator,
  validation_steps = ceiling(nrow(val_df) / batch_size),
  callbacks        = callbacks_list
)
plot(history)

dir.create(dossier_res, recursive = TRUE, showWarnings = FALSE)
nom_base <- sprintf("Test5_rejet_seuil_ep%d_seed%d", epochs, seed)
tryCatch(save_model_hdf5(model, file.path(dossier_res, paste0(nom_base, ".h5"))),
         error = function(e) message("Modele non sauvegarde : ", conditionMessage(e)))


# ---------------------------------------------------------------------
# 6. PROBABILITES ET CONFIANCE POUR CHAQUE IMAGE
# ---------------------------------------------------------------------
# Pour chaque image on obtient 3 probabilites (une par espece).
#   conf             = la plus grande des 3 (la "confiance")
#   espece_predite   = l'espece correspondante
# Ces deux colonnes servent ensuite a la regle de rejet.
predire <- function(df) {
  df  <- as.data.frame(df)
  gen <- flow_images_from_dataframe(
    df, directory = "", x_col = "path", generator = datagen,
    target_size = input_shape[1:2], class_mode = NULL,   # pas d'etiquettes : on predit seulement
    batch_size = batch_size, shuffle = FALSE)
  proba <- predict(model, gen, steps = ceiling(nrow(df) / batch_size), verbose = 0)

  idx <- match(gen$filenames, df$path)
  if (anyNA(idx)) {
    warning("Association par nom de fichier impossible : ordre des lignes utilise.")
    idx <- seq_len(nrow(df))
  }
  res <- as_tibble(df[idx, ])
  colnames(proba) <- classes
  res$conf           <- apply(proba, 1, max)
  res$espece_predite <- classes[max.col(proba, ties.method = "first")]
  bind_cols(res, as_tibble(proba))
}

cat("\nPredictions sur le jeu 'seuil'...\n")
pred_seuil <- predire(master_df %>% filter(partition == "seuil"))
cat("Predictions sur le jeu 'external'...\n")
pred_ext   <- predire(master_df %>% filter(partition == "external"))


# ---------------------------------------------------------------------
# 7. CHOIX DU SEUIL (sur le jeu "seuil", jamais sur le jeu final)
# ---------------------------------------------------------------------
# Pour chaque seuil s :
#   rejet_autre = part des images "autre" dont la confiance est < s
#                 (= rejetees a raison)
#   rejet_connu = part des images connues dont la confiance est < s
#                 (= rejetees a tort)
courbe_seuils <- function(pred, grille) {
  est_autre <- pred$class == "autre"
  bind_rows(lapply(grille, function(s) {
    tibble(seuil       = s,
           rejet_autre = mean(pred$conf[est_autre]  < s),
           rejet_connu = mean(pred$conf[!est_autre] < s),
           accuracy_4  = mean(ifelse(est_autre,
                                     pred$conf < s,
                                     pred$conf >= s & pred$espece_predite == pred$class)))
  })) %>% mutate(J = rejet_autre - rejet_connu)
}

courbe <- courbe_seuils(pred_seuil, grille_seuils)

cat("\n[Jeu 'seuil'] taux de rejet selon le seuil (un point sur dix) :\n")
print(courbe %>% filter(round(seuil * 100) %% 10 == 0) %>%
        mutate(across(-seuil, ~ round(.x, 3))))

if (!is.null(seuil_impose)) {
  seuil <- seuil_impose
  cat("\nSeuil impose :", seuil, "\n")
} else if (!is.null(rejet_connu_max)) {
  ok <- courbe %>% filter(rejet_connu <= rejet_connu_max)
  if (nrow(ok) == 0) stop("Aucun seuil ne respecte rejet_connu_max = ", rejet_connu_max)
  seuil <- ok$seuil[which.max(ok$rejet_autre)]
  cat("\nSeuil choisi (rejet maximal avec <=", rejet_connu_max,
      "d'images connues rejetees) :", seuil, "\n")
} else {
  seuil <- courbe$seuil[which.max(courbe$J)]
  cat("\nSeuil choisi (maximise J = rejet_autre - rejet_connu) :", seuil, "\n")
}
print(courbe %>% filter(abs(seuil - !!seuil) < 1e-9) %>%
        mutate(across(-seuil, ~ round(.x, 3))))

cat("\nRappel : avec 3 sorties softmax, la confiance est toujours >= 1/3 (",
    round(1/3, 3), "). Part des images du jeu 'seuil' sous 0.34 :",
    round(mean(pred_seuil$conf < 0.34), 3), "\n")


# ---------------------------------------------------------------------
# 8. EVALUATION FINALE SUR LE JEU "EXTERNAL" AVEC LE SEUIL CHOISI
# ---------------------------------------------------------------------
est_autre <- pred_ext$class == "autre"

# --- 8.1 mesure independante du seuil : AUROC ---------------------------
# Probabilite qu'une image connue tiree au hasard ait une confiance plus
# elevee qu'une image "autre" tiree au hasard (0.5 = hasard, 1 = parfait).
r  <- rank(pred_ext$conf)
n1 <- sum(!est_autre); n0 <- sum(est_autre)
auroc <- (sum(r[!est_autre]) - n1 * (n1 + 1) / 2) / (n1 * n0)
cat("\n[External] AUROC (confiance : connues vs 'autre') :", round(auroc, 3), "\n")

# --- 8.2 decision a 4 classes avec le seuil -----------------------------
pred4 <- ifelse(pred_ext$conf >= seuil, pred_ext$espece_predite, "autre")
vrai4 <- pred_ext$class
niv4  <- c(classes, "autre")

conf4 <- table(vraie   = factor(vrai4, levels = niv4),
               predite = factor(pred4, levels = niv4))
cat("\n[External] Tableau 1 : classe vraie x classe predite (seuil =", seuil, ")\n")
print(conf4)
cat("Accuracy globale a 4 classes :", round(mean(pred4 == vrai4), 3), "\n")

cat("\nRejet des images 'autre' (a raison) :", round(mean(pred4[est_autre] == "autre"), 3), "\n")
cat("Rejet des images connues (a tort)  :", round(mean(pred4[!est_autre] == "autre"), 3), "\n")
acceptees <- !est_autre & pred4 != "autre"
cat("Parmi les images connues acceptees, part bien classee :",
    round(mean(pred4[acceptees] == vrai4[acceptees]), 3), "\n")

# --- 8.3 second tableau : espece vraie ET statut -------------------------
ligne_vraie <- ifelse(est_autre, "autre", paste(pred_ext$class, pred_ext$statut, sep = " | "))
niveaux_lignes <- c(unlist(lapply(classes, function(e) paste(e, lib_statut, sep = " | "))),
                    "autre")
conf_statut <- table(vraie_et_statut = factor(ligne_vraie, levels = niveaux_lignes),
                     predite         = factor(pred4, levels = niv4))
cat("\n[External] Tableau 2 : classe vraie ET statut x classe predite\n")
print(conf_statut)

taux_ligne <- sapply(niveaux_lignes, function(l) {
  attendu <- if (l == "autre") "autre" else strsplit(l, " | ", fixed = TRUE)[[1]][1]
  conf_statut[l, attendu] / sum(conf_statut[l, ])
})
cat("\nTaux de bonne decision par ligne (autre : rejetee ; espece : acceptee ET bien classee) :\n")
print(round(taux_ligne, 3))


# ---------------------------------------------------------------------
# 9. GRAPHIQUES
# ---------------------------------------------------------------------
groupe_plot <- function(d) {
  case_when(d$class == "autre"       ~ "autre (a rejeter)",
            d$statut == "domestique" ~ "connue - domestique",
            TRUE                     ~ "connue - sauvage")
}

g_conf <- bind_rows(pred_seuil %>% mutate(jeu = "seuil"),
                    pred_ext   %>% mutate(jeu = "external")) %>%
  mutate(groupe = groupe_plot(.)) %>%
  ggplot(aes(conf, fill = groupe)) +
  geom_density(alpha = 0.4) +
  geom_vline(xintercept = seuil, linetype = "dashed") +
  facet_wrap(~ jeu) +
  labs(x = "confiance = plus grande probabilite softmax", y = "densite",
       title = "Distribution de la confiance du modele",
       subtitle = paste("trait pointille : seuil =", seuil)) +
  theme_minimal()
print(g_conf)
ggsave(file.path(dossier_res, paste0(nom_base, "_confiance.png")), g_conf,
       width = 9, height = 4, dpi = 150)

g_courbe <- courbe %>%
  pivot_longer(c(rejet_autre, rejet_connu), names_to = "taux", values_to = "valeur") %>%
  ggplot(aes(seuil, valeur, colour = taux)) +
  geom_line() +
  geom_vline(xintercept = seuil, linetype = "dashed") +
  labs(x = "seuil de confiance", y = "part d'images rejetees",
       title = "Compromis du seuil (jeu 'seuil')") +
  theme_minimal()
print(g_courbe)
ggsave(file.path(dossier_res, paste0(nom_base, "_courbe_seuil.png")), g_courbe,
       width = 7, height = 4, dpi = 150)


# ---------------------------------------------------------------------
# 10. SAUVEGARDE DES RESULTATS
# ---------------------------------------------------------------------
resultats <- list(
  master_df         = master_df,
  historique        = as.data.frame(history$metrics),
  courbe_seuils     = courbe,
  seuil             = seuil,
  auroc             = auroc,
  confusion_4       = conf4,
  confusion_statut  = conf_statut,
  taux_par_ligne    = taux_ligne,
  predictions_seuil = pred_seuil,
  predictions_ext   = pred_ext,
  classes           = classes,
  config = list(taxons = taxons, statuts_entrainement = statuts_entrainement,
                effectifs = effectifs, n_autres_seuil = n_autres_seuil,
                n_autres_ext = n_autres_ext, seuil_impose = seuil_impose,
                rejet_connu_max = rejet_connu_max, epochs = epochs, seed = seed,
                col_image = col_image, rda_autres = rda_autres),
  duree = Sys.time() - begin_time
)
save(resultats, file = file.path(dossier_res, paste0(nom_base, ".rda")))

cat("\nResultats ecrits dans :", normalizePath(dossier_res), "\n")
cat("Duree totale :", format(Sys.time() - begin_time), "\n")
