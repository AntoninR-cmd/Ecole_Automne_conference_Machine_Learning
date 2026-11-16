# =====================================================================
# Creer_rda_archeo.R
# Fabrique de nouveaux tableaux (.rda) pour les images ajoutees au projet.
#
# PARTIE A : pour chaque espece (olea, phoenix, vitis), copie du .rda de
#            base + ajout des graines ARCHEOBOTANIQUES (autre dossier,
#            sans donnees de contour pour la methode EFT).
#            -> rda/db/<taxon>_db2_archeo.rda
#
# PARTIE B : un tableau separe pour les images BRUITEES/MODIFIEES et les
#            images SANS LIEN, avec comme etiquette (colonne status)
#            l'espece (olea / phoenix / vitis) au lieu de c / w.
#            -> rda/db/autres_images_db.rda
#
# Les .rda d'origine ne sont JAMAIS modifies : on ecrit de nouveaux fichiers.
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES (a adapter a ton arborescence reelle)
# ---------------------------------------------------------------------
racine     <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db <- "rda/db"
taxons     <- c("olea", "phoenix", "vitis")

# --- Partie A : graines archeobotaniques (un dossier par espece)
dossiers_archeo <- c(olea    = "DATA/ARCHEO/OLEA",
                     phoenix = "DATA/ARCHEO/PHOENIX",
                     vitis   = "DATA/ARCHEO/VITIS")

# Valeur de "status" donnee aux graines archeo. On evite volontairement "c"
# et "w" pour que le filtre status == "c" de Test2.R ne les prenne pas.
status_archeo <- "a"

# --- Partie B : images bruitees/modifiees (un dossier par espece)
dossiers_bruitees <- c(olea    = "DATA/BRUITEES/OLEA",
                       phoenix = "DATA/BRUITEES/PHOENIX",
                       vitis   = "DATA/BRUITEES/VITIS")

# Expression reguliere retiree de l'identifiant d'une image bruitee pour
# retrouver l'image d'origine. Exemple : si "0001_1_bruit.jpg" est la
# version bruitee de "0001_1_VD.jpg"... adapter. NULL = pas de lien enregistre.
motif_source_bruitees <- NULL     # ex : "_(bruit|sale|noise)[0-9]*$"

# --- Partie B : images sans lien avec les 3 especes
dossier_sans_lien <- "DATA/SANS_LIEN"
classe_sans_lien  <- "autre"      # etiquette (status) donnee a ces images

# Vue utilisee quand le nom de fichier ne finit pas par _VD, _VL ou _VV
vue_defaut <- "VD"

# Securite : ne pas ecraser un fichier existant
ecraser <- FALSE


# ---------------------------------------------------------------------
# 1. PACKAGES, DOSSIER DE TRAVAIL
# ---------------------------------------------------------------------
library(tidyverse)
setwd(racine)
cat("Dossier de travail :", getwd(), "\n")


# ---------------------------------------------------------------------
# 2. FONCTIONS
# ---------------------------------------------------------------------

# Construit un tableau "une ligne = une graine" a partir d'un dossier.
# Les fichiers sont supposes nommes <id>_VD.jpg, <id>_VL.jpg, comme dans
# l'archive : les deux vues d'une meme graine sont regroupees sur une ligne
# (colonnes VD_deep et VL_deep). Sans suffixe de vue, vue_defaut est utilisee.
# Les chemins sont stockes RELATIFS a la racine du projet.
tableau_depuis_dossier <- function(dossier, prefixe_id = "",
                                   motif_source = NULL) {
  if (!dir.exists(dossier)) stop("Dossier introuvable : ", dossier)
  fichiers <- list.files(dossier, pattern = "\\.(jpe?g|png)$",
                         recursive = TRUE, full.names = TRUE,
                         ignore.case = TRUE)
  if (length(fichiers) == 0) stop("Aucune image .jpg/.png dans ", dossier)

  nom   <- tools::file_path_sans_ext(basename(fichiers))
  a_vue <- grepl("_(VD|VL|VV)$", nom)
  vue   <- ifelse(a_vue, sub("^.*_(VD|VL|VV)$", "\\1", nom), vue_defaut)
  id_brut <- ifelse(a_vue, sub("_(VD|VL|VV)$", "", nom), nom)

  # id_source : identifiant de l'image d'origine (utile pour eviter qu'une
  # image et sa version bruitee se retrouvent l'une en train, l'autre en test)
  id_source <- if (is.null(motif_source)) id_brut else sub(motif_source, "", id_brut)

  long <- tibble(id        = paste0(prefixe_id, id_brut),
                 folder    = basename(dirname(fichiers)),
                 id_source = id_source,
                 vue       = vue,
                 path      = fichiers)

  doublons <- long %>% count(id, folder, vue) %>% filter(n > 1)
  if (nrow(doublons) > 0) {
    warning(nrow(doublons), " couple(s) id/vue en double : seul le premier est garde.")
    long <- long %>% distinct(id, folder, vue, .keep_all = TRUE)
  }

  large <- long %>%
    pivot_wider(names_from = vue, values_from = path, names_glue = "{vue}_deep")
  for (col in c("VD_deep", "VL_deep")) {
    if (!col %in% names(large)) large[[col]] <- NA_character_
  }
  large
}

# Sauvegarde sans ecraser (sauf si ecraser <- TRUE)
sauver_rda <- function(objet, nom_objet, fichier) {
  if (file.exists(fichier) && !ecraser) {
    stop("Le fichier existe deja : ", fichier,
         " (mettre ecraser <- TRUE pour le remplacer)")
  }
  env <- new.env()
  assign(nom_objet, objet, envir = env)
  save(list = nom_objet, envir = env, file = fichier)
  cat("Ecrit :", fichier, "\n")
}

# Verifie que tous les chemins non vides existent
verifier_chemins <- function(df) {
  for (col in intersect(c("VD_deep", "VL_deep"), names(df))) {
    ch <- df[[col]][!is.na(df[[col]])]
    manque <- ch[!file.exists(ch)]
    if (length(manque) > 0) {
      print(head(manque))
      stop(length(manque), " fichiers introuvables dans la colonne ", col)
    }
  }
  invisible(TRUE)
}


# ---------------------------------------------------------------------
# 3. PARTIE A : .rda de chaque espece + graines archeobotaniques
# ---------------------------------------------------------------------
for (taxon in taxons) {
  cat("\n===== PARTIE A :", toupper(taxon), "=====\n")

  # --- 3.1 .rda de base (on prefere <taxon>_db2.rda)
  fichiers <- list.files(dossier_db, pattern = taxon, full.names = TRUE,
                         ignore.case = TRUE)
  prefere  <- fichiers[tolower(basename(fichiers)) == paste0(taxon, "_db2.rda")]
  if (length(prefere) == 0) stop("Fichier ", taxon, "_db2.rda introuvable dans ", dossier_db)
  env_tmp  <- new.env()
  nom_objet <- load(prefere[1], envir = env_tmp)[1]
  db_orig  <- get(nom_objet, envir = env_tmp)
  cat("Base :", prefere[1], "- objet", nom_objet, "-", nrow(db_orig), "lignes\n")
  if (!all(c("id", "status") %in% names(db_orig))) {
    print(names(db_orig)); stop("Colonnes 'id' et/ou 'status' absentes.")
  }

  # --- 3.2 nouvelles lignes archeo
  if (!dir.exists(dossiers_archeo[[taxon]])) {
    message("Dossier archeo absent pour ", taxon, " : espece ignoree.")
    next
  }
  archeo <- tableau_depuis_dossier(dossiers_archeo[[taxon]],
                                   prefixe_id = "archeo_") %>%
    mutate(status = status_archeo, origine = "archeo")
  cat("Graines archeo :", nrow(archeo), "\n")
  verifier_chemins(archeo)

  # --- 3.3 mise en forme de la base d'origine (types compatibles)
  orig <- db_orig %>%
    mutate(across(any_of(c("id", "folder", "status")), as.character),
           origine   = "moderne",
           id_source = id)

  cat("Colonnes de la base absentes des lignes archeo (remplies par NA) :\n")
  print(setdiff(names(orig), names(archeo)))   # typiquement les contours *_coo

  # --- 3.4 fusion et controles
  db_new <- bind_rows(orig, archeo)
  if (anyDuplicated(db_new$id) > 0) warning("Identifiants en double dans ", taxon)

  cat("Repartition origine x status :\n")
  print(table(db_new$origine, db_new$status))

  sauver_rda(db_new, nom_objet,
             file.path(dossier_db, paste0(taxon, "_db2_archeo.rda")))
}


# ---------------------------------------------------------------------
# 4. PARTIE B : images bruitees/modifiees + images sans lien
# ---------------------------------------------------------------------
cat("\n===== PARTIE B : images bruitees et sans lien =====\n")
morceaux <- list()

for (taxon in taxons) {
  if (!dir.exists(dossiers_bruitees[[taxon]])) {
    message("Dossier de bruitees absent pour ", taxon, " : ignore.")
    next
  }
  morceaux[[paste0("bruitee_", taxon)]] <-
    tableau_depuis_dossier(dossiers_bruitees[[taxon]],
                           prefixe_id   = "bruit_",
                           motif_source = motif_source_bruitees) %>%
    mutate(status = taxon, origine = "bruitee")
}

if (dir.exists(dossier_sans_lien)) {
  morceaux[["sans_lien"]] <-
    tableau_depuis_dossier(dossier_sans_lien, prefixe_id = "sanslien_") %>%
    mutate(status = classe_sans_lien, origine = "sans_lien")
} else {
  message("Dossier sans lien absent : ignore.")
}

if (length(morceaux) > 0) {
  autres_images_db <- bind_rows(morceaux) %>%
    select(id, folder, status, origine, id_source, VD_deep, VL_deep, everything())
  verifier_chemins(autres_images_db)

  cat("Repartition origine x status :\n")
  print(table(autres_images_db$origine, autres_images_db$status))

  sauver_rda(autres_images_db, "autres_images_db",
             file.path(dossier_db, "autres_images_db.rda"))
}

cat("\nTermine.\n")
