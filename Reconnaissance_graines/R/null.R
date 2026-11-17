# =====================================================================
# VERIFICATION DES FUITES DE DONNEES (TRAIN vs VAL)
# =====================================================================

verifier_fuite <- function(df_train, df_val, nom_jeu) {
  cat("\n=====================================================\n")
  cat("VERIFICATION DES FUITES :", nom_jeu, "\n")
  cat("=====================================================\n")
  
  # 1. Extraction des noms de fichiers complets
  noms_train <- basename(df_train$path)
  noms_val   <- basename(df_val$path)
  
  # 2. Extraction des "identifiants" de base (sans extension)
  # Cela permet de reperer si "graine01_dorsal.jpg" et "graine01_dorsal.png"
  # sont presents par exemple.
  ids_train <- tools::file_path_sans_ext(noms_train)
  ids_val   <- tools::file_path_sans_ext(noms_val)
  
  # 3. Recherche des doublons exacts (meme nom de fichier complet)
  doublons_exacts <- intersect(noms_train, noms_val)
  
  # 4. Recherche des collisions sur l'identifiant de base
  # (utile si les noms sont tres proches mais pas identiques)
  collisions_ids <- intersect(ids_train, ids_val)
  
  cat("Nombre d'images dans TRAIN :", length(noms_train), "\n")
  cat("Nombre d'images dans VAL   :", length(noms_val), "\n\n")
  
  # --- Resultats doublons exacts
  if (length(doublons_exacts) > 0) {
    cat("[!] ALERTE :", length(doublons_exacts), "noms de fichiers EXACTS se retrouvent dans TRAIN et VAL.\n")
    cat("    Exemples :\n")
    print(head(doublons_exacts, 5))
    cat("    -> Il y a une fuite directe de donnees.\n")
  } else {
    cat("[OK] Aucun nom de fichier exact n'est partage entre TRAIN et VAL.\n")
  }
  
  # --- Resultats collisions d'identifiants
  if (length(collisions_ids) > 0) {
    cat("\n[!] ATTENTION :", length(collisions_ids), "identifiants de base (sans extension) sont partages.\n")
    cat("    Exemples :\n")
    print(head(collisions_ids, 5))
    cat("    -> Si ces identifiants correspondent a la MEME graine biologique (par ex. une autre vue),\n")
    cat("       c'est une fuite de donnees.\n")
  } else {
    cat("\n[OK] Aucun identifiant de base partage.\n")
  }
  
  # --- Verification supplementaire : les doublons sont-ils de la meme classe ?
  if (length(doublons_exacts) > 0) {
    # On joint pour voir si la classe est coherente
    df_train_sub <- df_train %>% 
      mutate(nom = basename(path)) %>% 
      filter(nom %in% doublons_exacts) %>% 
      select(nom, class, statut, path)
    
    df_val_sub <- df_val %>% 
      mutate(nom = basename(path)) %>% 
      filter(nom %in% doublons_exacts) %>% 
      select(nom, class, statut, path)
    
    fusion <- inner_join(df_train_sub, df_val_sub, by = "nom", suffix = c("_train", "_val"))
    
    if (nrow(fusion) > 0) {
      cat("\nComparaison des classes pour les doublons :\n")
      print(table(fusion$class_train, fusion$class_val))
      if (all(fusion$class_train == fusion$class_val)) {
        cat("-> Les doublons ont la MEME classe (fuite de donnee 'propre', le modele apprend par coeur).\n")
      } else {
        cat("-> ATTENTION : certains doublons ont des classes DIFFERENTES (bruit d'etiquetage ou erreur).\n")
      }
    }
  }
  
  cat("=====================================================\n\n")
}

# Appel de la fonction sur les jeux sauvages
verifier_fuite(
  df_train = jeux$sauvages$train, 
  df_val   = jeux$sauvages$val, 
  nom_jeu  = "Sauvages (Train vs Validation)"
)

# Optionnel : verification sur les domestiques aussi
verifier_fuite(
  df_train = jeux$domestiques$train, 
  df_val   = jeux$domestiques$val, 
  nom_jeu  = "Domestiques (Train vs Validation)"
)