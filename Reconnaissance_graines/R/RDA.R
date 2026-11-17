# =====================================================================
# Creer_rda_archeo.R
# Construit des fichiers .rda a partir de dossiers d'images :
#   - OLEA_ARCHEO / PHOENIX_ARCHEO / VITIS_ARCHEO (3 especes archeo)
#   - AUTRES (autres especes modernes = classe de rejet)
#
# Structure produite (compatible Eval_modele.R modifie) :
#   resultats <- list(
#     master_df = tibble(partition, class, statut, path, <metadata>),
#     config    = list(...)
#   )
#
# - partition = "external" (pret a l'evaluation)
# - class     = "olea" / "phoenix" / "vitis" / "autre"
# - statut    = "archeo" pour les 3 especes, "autre_espece" pour AUTRES
# - path      = chemin relatif a la racine du projet
#
# Colonnes de metadonnees conservees pour coherence avec Creer_rda.R :
#   id, folder, img_deep, taxon, origine, collection, specimen,
#   status, vue, largeur, hauteur, mode, lisible
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine         <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_sortie <- "rda/db_ajout"

# Taches : un dossier source -> un .rda. Adapte les chemins si besoin.
taches <- list(
  list(dossier = "DATA/OLEA_ARCHEO",    taxon = "olea",
       statut = "archeo",       origine = "archeo",
       nom_objet = "archeo_olea_db",    collection = NA_character_),
  list(dossier = "DATA/PHOENIX_ARCHEO", taxon = "phoenix",
       statut = "archeo",       origine = "archeo",
       nom_objet = "archeo_phoenix_db", collection = NA_character_),
  list(dossier = "DATA/VITIS_ARCHEO",   taxon = "vitis",
       statut = "archeo",       origine = "archeo",
       nom_objet = "archeo_vitis_db",   collection = NA_character_),
  list(dossier = "DATA/AUTRES",         taxon = "autre",
       statut = "autre_espece", origine = "autre_espece",
       nom_objet = "autres_especes_db", collection = NA_character_)
)

# Nom du .rda combinant les 4 (pour l'evaluation finale).
nom_objet_combine <- "archeo_complet_db"

extensions <- "\\.(jpe?g|png|tiff?|bmp)$"
lire_infos <- TRUE    # lire largeur/hauteur/mode (necessite Pillow)

# Extraction optionnelle d'une cle de specimen depuis le nom de fichier
# (utile pour regrouper les vues d'un meme specimen). NULL = pas d'extraction.
regex_specimen <- NULL

# Extraction optionnelle de la vue depuis le nom de fichier
# (ex. "_VD_" ou "_VL_"). NULL = colonne vue laissee a NA.
regex_vue <- NULL     # ex. "(VD|VL)"

ecraser <- TRUE


# ---------------------------------------------------------------------
# 1. PACKAGES ET DOSSIER DE TRAVAIL
# ---------------------------------------------------------------------
library(tidyverse)

setwd(racine)
racine_norm <- normalizePath(racine, winslash = "/")
cat("Dossier de travail :", getwd(), "\n")


# ---------------------------------------------------------------------
# 2. FONCTIONS OUTILS (reprises de Creer_rda.R)
# ---------------------------------------------------------------------
chemin_relatif <- function(chemins_abs) {
  prefixe <- paste0(racine_norm, "/")
  ifelse(startsWith(chemins_abs, prefixe),
         substring(chemins_abs, nchar(prefixe) + 1),
         chemins_abs)
}

infos_image <- function(chemins) {
  vide <- tibble(largeur = rep(NA_integer_, length(chemins)),
                 hauteur = rep(NA_integer_, length(chemins)),
                 mode    = rep(NA_character_, length(chemins)))
  tryCatch({
    reticulate::py_require("pillow")
    PIL <- reticulate::import("PIL.Image")
    lignes <- lapply(seq_along(chemins), function(i) {
      if (i %% 200 == 0) cat("  images lues :", i, "/", length(chemins), "\n")
      tryCatch({
        im <- PIL$open(chemins[i])
        taille <- unlist(im$size)
        m <- im$mode
        im$close()
        list(largeur = as.integer(taille[1]),
             hauteur = as.integer(taille[2]),
             mode = m)
      }, error = function(e) list(largeur = NA_integer_,
                                  hauteur = NA_integer_,
                                  mode = NA_character_))
    })
    bind_rows(lignes)
  }, error = function(e) {
    message("Lecture des dimensions impossible (", conditionMessage(e),
            ") : colonnes largeur/hauteur/mode a NA.")
    vide
  })
}


# ---------------------------------------------------------------------
# 3. FONCTION PRINCIPALE : UN DOSSIER -> UN master_df
# ---------------------------------------------------------------------
creer_master_df <- function(dossier, taxon, statut, origine,
                            collection = NA_character_) {
  cat("\n=====================================================\n")
  cat("Dossier :", dossier, "\n")
  cat("  taxon :", taxon, " | statut :", statut,
      " | origine :", origine, "\n")
  cat("=====================================================\n")

  if (!dir.exists(dossier)) {
    warning("Dossier introuvable : ", dossier, " (ignore).")
    return(NULL)
  }
  fichiers <- list.files(normalizePath(dossier, winslash = "/"),
                         pattern = extensions, recursive = TRUE,
                         full.names = TRUE, ignore.case = TRUE)
  if (length(fichiers) == 0) {
    warning("Aucune image dans ", dossier, " (ignore).")
    return(NULL)
  }
  fichiers <- sort(normalizePath(fichiers, winslash = "/"))
  cat("Images trouvees :", length(fichiers), "\n")

  ids     <- tools::file_path_sans_ext(basename(fichiers))
  folders <- basename(dirname(fichiers))
  rel     <- chemin_relatif(fichiers)

  # choix de la vue (optionnel)
  vue_vec <- if (is.null(regex_vue)) rep(NA_character_, length(fichiers))
             else str_extract(ids, regex_vue)

  db <- tibble(
    # --- colonnes attendues par Eval_modele.R
    partition = "external",
    class     = taxon,
    statut    = statut,
    path      = rel,
    # --- metadonnees (coherence avec Creer_rda.R)
    id        = ids,
    folder    = folders,
    img_deep  = rel,
    taxon     = taxon,
    origine   = origine,
    collection = collection,
    specimen  = if (is.null(regex_specimen)) NA_character_
                else str_extract(ids, regex_specimen),
    status    = statut,   # duplique de statut (ancien nom)
    vue       = vue_vec
  )

  if (lire_infos) {
    cat("Lecture des dimensions et du mode des images...\n")
    db <- bind_cols(db, infos_image(fichiers))
    db$lisible <- !is.na(db$largeur)
  } else {
    db <- db %>% mutate(largeur = NA_integer_, hauteur = NA_integer_,
                        mode = NA_character_, lisible = NA)
  }

  # resume
  cat("\nImages par dossier parent (folder) :\n")
  print(head(sort(table(db$folder), decreasing = TRUE), 15))
  if (lire_infos && any(!is.na(db$mode))) {
    cat("\nMode des images (L = gris, RGB = couleur) :\n")
    print(table(db$mode, useNA = "ifany"))
    if (any(!db$lisible))
      warning(sum(!db$lisible), " image(s) illisible(s).")
  }

  db
}


# ---------------------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------------------
dir.create(dossier_sortie, recursive = TRUE, showWarnings = FALSE)

liste_dfs <- list()

for (t in taches) {
  df <- creer_master_df(t$dossier, t$taxon, t$statut, t$origine, t$collection)
  if (is.null(df)) next
  liste_dfs[[t$nom_objet]] <- df

  fichier_rda <- file.path(dossier_sortie, paste0(t$nom_objet, ".rda"))
  if (file.exists(fichier_rda) && !ecraser)
    stop("Le fichier existe deja : ", fichier_rda)

  resultats <- list(
    master_df = df,
    config = list(dossier = t$dossier, taxon = t$taxon,
                  statut = t$statut, origine = t$origine,
                  n_images = nrow(df), date = Sys.time())
  )
  assign(t$nom_objet, resultats)
  save(list = t$nom_objet, file = fichier_rda, envir = environment())
  write.csv(df, file.path(dossier_sortie, paste0(t$nom_objet, ".csv")),
            row.names = FALSE, fileEncoding = "UTF-8")

  cat("  -> ", fichier_rda, " (", nrow(df), " lignes)\n", sep = "")
}


# ---------------------------------------------------------------------
# 5. FICHIER COMBINE (pour l'evaluation finale)
# ---------------------------------------------------------------------
if (length(liste_dfs) > 0) {
  cat("\n=====================================================\n")
  cat("Construction du .rda combine :", nom_objet_combine, "\n")
  cat("=====================================================\n")

  # aligner les colonnes (certains dossiers peuvent etre absents)
  cols_ref <- names(liste_dfs[[1]])
  liste_dfs <- lapply(liste_dfs, function(d) {
    manquants <- setdiff(cols_ref, names(d))
    for (m in manquants) d[[m]] <- NA
    d[, cols_ref, drop = FALSE]
  })
  combine <- bind_rows(liste_dfs)

  cat("\nRepartition finale (class x statut) :\n")
  print(table(combine$class, combine$statut, useNA = "ifany"))

  resultats_combine <- list(
    master_df = combine,
    config = list(
      sources = sapply(taches, `[[`, "dossier"),
      n_total = nrow(combine),
      n_par_classe = as.list(table(combine$class)),
      date = Sys.time()
    )
  )
  fichier_combine <- file.path(dossier_sortie, paste0(nom_objet_combine, ".rda"))
  assign(nom_objet_combine, resultats_combine)
  save(list = nom_objet_combine, file = fichier_combine, envir = environment())
  write.csv(combine, file.path(dossier_sortie, paste0(nom_objet_combine, ".csv")),
            row.names = FALSE, fileEncoding = "UTF-8")
  cat("\n -> ", fichier_combine, " (", nrow(combine), " lignes)\n", sep = "")
}

cat("\nFichiers presents dans", dossier_sortie, ":\n")
print(list.files(dossier_sortie))