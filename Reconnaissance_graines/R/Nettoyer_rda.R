# =====================================================================
# Nettoyer_rda.R
# Retire d'un .rda les lignes dont les images ont ete supprimees du
# dossier de photos, ecrase l'ancien .rda et affiche les effectifs par
# statut avant / apres.
#
# Principe : une ligne est consideree "a supprimer" si l'image d'AU MOINS
# UNE des colonnes d'images (VD_deep, VL_deep) n'existe plus dans le
# dossier. La comparaison se fait sur le NOM DU FICHIER (sans le chemin),
# comme dans les scripts Test : elle marche meme si les chemins stockes
# dans le .rda ne correspondent pas a ton arborescence.
#
# Securites :
#   - le nombre de lignes trouvees doit etre egal a n_attendu (126),
#     sinon le script s'arrete SANS rien ecrire
#   - une copie de l'ancien .rda est faite avant d'ecraser
# Commentaires sans accents (encodage Windows).
# =====================================================================


# ---------------------------------------------------------------------
# 0. PARAMETRES
# ---------------------------------------------------------------------
racine      <- "C:/Users/fanos/Documents/Cours/Projet ML"
dossier_db  <- "rda/db"
taxon       <- "hordeum"
dossier_img <- "DATA/HORDEUM"

# colonnes d'images du .rda : une ligne est supprimee si une de ces
# images manque (mettre "VD_deep" seul pour ne tester que la vue dorsale)
colonnes_images <- c("VD_deep")

n_attendu <- 30      # nombre de lignes qu'on s'attend a supprimer (NULL = pas de controle)
ecrire    <- TRUE     # FALSE = simulation : affiche tout, n'ecrit rien

faire_sauvegarde   <- TRUE
dossier_sauvegarde <- "rda/db_sauvegardes"   # hors de rda/db (evite toute confusion)


# ---------------------------------------------------------------------
# 1. PACKAGES ET DOSSIER DE TRAVAIL
# ---------------------------------------------------------------------
library(tidyverse)
setwd(racine)
cat("Dossier de travail :", getwd(), "\n")


# ---------------------------------------------------------------------
# 2. CHARGEMENT DU .RDA
# ---------------------------------------------------------------------
fichiers <- list.files(dossier_db, pattern = taxon, full.names = TRUE, ignore.case = TRUE)
if (length(fichiers) == 0) stop("Aucun .rda '", taxon, "' dans ", dossier_db)
prefere <- fichiers[tolower(basename(fichiers)) == paste0(taxon, "_db2.rda")]
fichier_rda <- if (length(prefere) > 0) prefere[1] else fichiers[1]
cat("Fichier .rda :", fichier_rda, "\n")

env_rda <- new.env()
noms    <- load(fichier_rda, envir = env_rda)
nom_obj <- noms[1]
db      <- get(nom_obj, envir = env_rda)
cat("Objet :", nom_obj, "-", nrow(db), "lignes\n")

if (!all(c("status", colonnes_images) %in% names(db))) {
  print(names(db))
  stop("Colonnes manquantes dans le .rda.")
}

cat("\nGraines disponibles (par statut) - AVANT nettoyage :\n")
avant <- table(db$status)
print(avant)


# ---------------------------------------------------------------------
# 3. REPERAGE DES LIGNES DONT L'IMAGE A ETE SUPPRIMEE
# ---------------------------------------------------------------------
if (!dir.exists(dossier_img)) stop("Dossier introuvable : ", dossier_img)
noms_disque <- basename(list.files(dossier_img, recursive = TRUE))
cat("\nImages presentes dans", dossier_img, ":", length(noms_disque), "\n")

manquant <- sapply(colonnes_images, function(col) {
  !(basename(as.character(db[[col]])) %in% noms_disque)
})
manquant <- matrix(manquant, nrow = nrow(db), dimnames = list(NULL, colonnes_images))

cat("\nImages introuvables, par colonne :\n")
print(colSums(manquant))

a_supprimer <- apply(manquant, 1, any)
cat("Lignes a supprimer (au moins une image manquante) :", sum(a_supprimer), "\n")

cat("\nExemple de lignes reperees :\n")
print(head(db[a_supprimer, ] %>% select(any_of(c("id", "folder", "status", colonnes_images))), 10))

cat("\nLignes reperees par statut :\n")
print(table(db$status[a_supprimer]))

# controle du nombre attendu
if (!is.null(n_attendu) && sum(a_supprimer) != n_attendu) {
  stop("Le script a trouve ", sum(a_supprimer), " lignes a supprimer au lieu de ",
       n_attendu, " : rien n'a ete modifie. Verifier 'colonnes_images' (une seule ",
       "vue ?) et le contenu de ", dossier_img, ".")
}


# ---------------------------------------------------------------------
# 4. SUPPRESSION ET SAUVEGARDE (ecrase l'ancien .rda)
# ---------------------------------------------------------------------
db_net <- db[!a_supprimer, ]
cat("\nLignes avant :", nrow(db), "| apres :", nrow(db_net), "\n")

if (ecrire) {
  if (faire_sauvegarde) {
    dir.create(dossier_sauvegarde, recursive = TRUE, showWarnings = FALSE)
    copie <- file.path(dossier_sauvegarde,
                       paste0(tools::file_path_sans_ext(basename(fichier_rda)),
                              "_avant_nettoyage_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".rda"))
    file.copy(fichier_rda, copie)
    cat("Copie de l'ancien .rda :", copie, "\n")
  }
  assign(nom_obj, db_net, envir = env_rda)          # remplace l'objet nettoye
  save(list = noms, file = fichier_rda, envir = env_rda)  # ecrase l'ancien .rda
  cat(".rda ecrase :", fichier_rda, "\n")
} else {
  cat("SIMULATION (ecrire = FALSE) : rien n'a ete ecrit.\n")
}


# ---------------------------------------------------------------------
# 5. VERIFICATION : RELECTURE ET EFFECTIFS PAR STATUT
# ---------------------------------------------------------------------
if (ecrire) {
  env_verif <- new.env()
  noms_v    <- load(fichier_rda, envir = env_verif)
  db_verif  <- get(noms_v[1], envir = env_verif)
} else {
  db_verif <- db_net
}

apres <- table(db_verif$status)
apres_v <- setNames(as.integer(apres[names(avant)]), names(avant))
apres_v[is.na(apres_v)] <- 0L

cat("\nGraines disponibles (par statut) - APRES nettoyage :\n")
print(apres)

cat("\nRecapitulatif :\n")
print(rbind(avant      = as.integer(avant),
            apres      = apres_v,
            supprimees = as.integer(avant) - apres_v) %>% `colnames<-`(names(avant)))
