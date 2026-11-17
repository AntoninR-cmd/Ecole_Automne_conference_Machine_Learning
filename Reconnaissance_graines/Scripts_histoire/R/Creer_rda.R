# =====================================================================
# Creer_rda.R
# Construit des fichiers .rda (tableaux de donnees) a partir d'un dossier
# d'images, pour :
#   - les graines ARCHEOBOTANIQUES des 3 especes (olea, phoenix, vitis)
#   - les graines d'AUTRES ESPECES (classe de rejet, "autre")
#
# Contrairement aux .rda de l'article, pour ces nouvelles images on ne
# sait reconnaitre ni la vue (VD / VL) ni le statut (sauvage / domestique).
# Le tableau ne contient donc :
#   - AUCUNE colonne VD_deep / VL_deep : une seule colonne "img_deep"
#     (chemin de l'image, vue inconnue)
#   - une colonne "status" presente mais VIDE (NA)
#   - une colonne "vue" presente mais VIDE (NA)
#
# Colonnes produites :
#   id          nom du fichier sans extension
#   folder      nom du dossier parent immediat (site, contexte, espece...)
#   img_deep    chemin de l'image, relatif a la racine du projet
#   taxon       classe a predire : "olea", "phoenix", "vitis" ou "autre"
#   origine     "archeo" ou "autre_espece"
#   collection  libelle libre (site, fouille...) - facultatif
#   specimen    cle de specimen extraite de l'id (si regex_specimen renseigne)
#   status      NA (inconnu)
#   vue         NA (inconnue)
#   largeur, hauteur, mode, lisible : informations lues dans l'image
#               (mode "L" = niveaux de gris, "RGB" = couleur)
#
# Les fichiers sont ecrits dans rda/db_ajout (et NON dans rda/db), pour ne
# pas melanger ces donnees avec celles de l'article.
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine         <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_sortie <- "rda/db_ajout"

# "choisir" : on choisit UN dossier a la main (fenetre Windows), puis le
#             type de donnees dans un menu
# "tout"    : on traite toutes les taches de la liste ci-dessous
mode <- "choisir"

# Taches du mode "tout". ATTENTION : les chemins de dossiers sont des
# exemples a ADAPTER a ton arborescence.
taches <- list(
  list(dossier = "DATA/ARCHEO/OLEA",    taxon = "olea",    origine = "archeo",
       nom_objet = "archeo_olea_db",    collection = NA_character_),
  list(dossier = "DATA/ARCHEO/PHOENIX", taxon = "phoenix", origine = "archeo",
       nom_objet = "archeo_phoenix_db", collection = NA_character_),
  list(dossier = "DATA/ARCHEO/VITIS",   taxon = "vitis",   origine = "archeo",
       nom_objet = "archeo_vitis_db",   collection = NA_character_),
  list(dossier = "DATA/AUTRES",         taxon = "autre",   origine = "autre_espece",
       nom_objet = "autres_especes_db", collection = NA_character_)
)

extensions <- "\\.(jpe?g|png|tiff?|bmp)$"   # formats d'image pris en compte

# Lire largeur / hauteur / mode de chaque image (necessite Python + Pillow,
# installes automatiquement par reticulate ; sinon colonnes a NA).
lire_infos <- TRUE

# Expression reguliere pour extraire une cle de specimen depuis l'id
# (utile si deux photos d'un meme specimen ont des noms proches, par
# exemple une partie commune). NULL = colonne "specimen" a NA.
# Exemple : "^[^_]+" garde ce qui precede le premier "_".
regex_specimen <- NULL

ecraser <- TRUE   # ecraser un .rda existant du meme nom


# ---------------------------------------------------------------------
# 1. PACKAGES ET DOSSIER DE TRAVAIL
# ---------------------------------------------------------------------
library(tidyverse)

setwd(racine)
racine_norm <- normalizePath(racine, winslash = "/")
cat("Dossier de travail :", getwd(), "\n")


# ---------------------------------------------------------------------
# 2. FONCTIONS OUTILS
# ---------------------------------------------------------------------

# chemin relatif a la racine du projet (barres "/"), comme dans les .rda de l'article
chemin_relatif <- function(chemins_abs) {
  prefixe <- paste0(racine_norm, "/")
  ifelse(startsWith(chemins_abs, prefixe),
         substring(chemins_abs, nchar(prefixe) + 1),
         chemins_abs)
}

# largeur, hauteur et mode de chaque image (lecture de l'en-tete seulement)
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
        list(largeur = as.integer(taille[1]), hauteur = as.integer(taille[2]), mode = m)
      }, error = function(e) list(largeur = NA_integer_, hauteur = NA_integer_,
                                  mode = NA_character_))
    })
    bind_rows(lignes)
  }, error = function(e) {
    message("Lecture des dimensions impossible (", conditionMessage(e),
            ") : colonnes largeur / hauteur / mode laissees a NA.")
    vide
  })
}


# ---------------------------------------------------------------------
# 3. FONCTION PRINCIPALE : UN DOSSIER -> UN .rda
# ---------------------------------------------------------------------
creer_rda_dossier <- function(dossier, taxon, origine, nom_objet,
                              collection = NA_character_) {
  cat("\n=====================================================\n")
  cat("Dossier :", dossier, "\n")
  cat("Classe (taxon) :", taxon, "| origine :", origine, "| objet :", nom_objet, "\n")
  cat("=====================================================\n")

  if (!dir.exists(dossier)) stop("Dossier introuvable : ", dossier)
  fichiers <- list.files(normalizePath(dossier, winslash = "/"),
                         pattern = extensions, recursive = TRUE,
                         full.names = TRUE, ignore.case = TRUE)
  if (length(fichiers) == 0) stop("Aucune image dans ", dossier)
  fichiers <- sort(normalizePath(fichiers, winslash = "/"))
  cat("Images trouvees :", length(fichiers), "\n")

  db <- tibble(
    id         = tools::file_path_sans_ext(basename(fichiers)),
    folder     = basename(dirname(fichiers)),
    img_deep   = chemin_relatif(fichiers),
    taxon      = taxon,
    origine    = origine,
    collection = collection,
    specimen   = if (is.null(regex_specimen)) NA_character_
                 else str_extract(tools::file_path_sans_ext(basename(fichiers)),
                                  regex_specimen),
    status     = NA_character_,   # inconnu pour ces images
    vue        = NA_character_    # inconnue pour ces images
  )

  if (lire_infos) {
    cat("Lecture des dimensions et du mode des images...\n")
    db <- bind_cols(db, infos_image(fichiers))
    db$lisible <- !is.na(db$largeur)
  } else {
    db <- db %>% mutate(largeur = NA_integer_, hauteur = NA_integer_,
                        mode = NA_character_, lisible = NA)
  }

  # ---- controles et resume --------------------------------------------
  if (anyDuplicated(db[, c("folder", "id")]) > 0) {
    warning("Des couples (folder, id) sont dupliques : verifier les noms de fichiers.")
  }
  cat("\nImages par dossier parent (folder) :\n")
  print(head(sort(table(db$folder), decreasing = TRUE), 15))
  if (lire_infos && any(!is.na(db$mode))) {
    cat("\nMode des images (L = niveaux de gris, RGB = couleur) :\n")
    print(table(db$mode, useNA = "ifany"))
    cat("\nDimensions : largeur", paste(range(db$largeur, na.rm = TRUE), collapse = " - "),
        "| hauteur", paste(range(db$hauteur, na.rm = TRUE), collapse = " - "), "\n")
    if (any(!db$lisible)) {
      warning(sum(!db$lisible), " image(s) illisible(s) : colonne 'lisible' == FALSE.")
    }
  }

  # ---- sauvegarde ------------------------------------------------------
  dir.create(dossier_sortie, recursive = TRUE, showWarnings = FALSE)
  fichier_rda <- file.path(dossier_sortie, paste0(nom_objet, ".rda"))
  if (file.exists(fichier_rda) && !ecraser) {
    stop("Le fichier existe deja : ", fichier_rda, " (ecraser = FALSE)")
  }
  assign(nom_objet, db)
  save(list = nom_objet, file = fichier_rda, envir = environment())
  write.csv(db, file.path(dossier_sortie, paste0(nom_objet, ".csv")),
            row.names = FALSE, fileEncoding = "UTF-8")

  cat("\n.rda ecrit :", normalizePath(fichier_rda, winslash = "/"), "\n")
  cat("Objet contenu :", nom_objet, "(", nrow(db), "lignes,", ncol(db), "colonnes )\n")
  invisible(db)
}


# ---------------------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------------------
if (mode == "choisir") {

  if (!interactive()) stop("Le mode 'choisir' demande une session interactive.")
  if (!exists("choose.dir")) stop("choose.dir() n'existe que sous Windows.")

  d <- choose.dir(caption = "Dossier d'images a traiter")
  if (is.na(d)) stop("Aucun dossier choisi.")

  types <- c("Graines archeo - olea", "Graines archeo - phoenix",
             "Graines archeo - vitis", "Autres especes (classe de rejet)")
  rep <- menu(types, title = "Quel type de donnees contient ce dossier ?")
  if (rep == 0) stop("Aucun type choisi.")

  params <- list(
    list(taxon = "olea",    origine = "archeo",       nom_objet = "archeo_olea_db"),
    list(taxon = "phoenix", origine = "archeo",       nom_objet = "archeo_phoenix_db"),
    list(taxon = "vitis",   origine = "archeo",       nom_objet = "archeo_vitis_db"),
    list(taxon = "autre",   origine = "autre_espece", nom_objet = "autres_especes_db"))[[rep]]

  coll <- trimws(readline("Collection / site (facultatif, Entree pour passer) : "))
  coll <- if (nchar(coll) == 0) NA_character_ else coll

  creer_rda_dossier(dossier = d, taxon = params$taxon, origine = params$origine,
                    nom_objet = params$nom_objet, collection = coll)

} else if (mode == "tout") {

  for (t in taches) {
    if (!dir.exists(t$dossier)) {
      message("Dossier absent, tache ignoree : ", t$dossier)
      next
    }
    creer_rda_dossier(t$dossier, t$taxon, t$origine, t$nom_objet, t$collection)
  }

} else {
  stop("'mode' doit valoir \"choisir\" ou \"tout\".")
}

cat("\nFichiers presents dans", dossier_sortie, ":\n")
print(list.files(dossier_sortie))
