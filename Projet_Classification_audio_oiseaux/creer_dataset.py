"""
Création du jeu de données de descripteurs audio (une ligne par fichier .wav).

Pour chaque enregistrement, on calcule les descripteurs vus dans
Test_transformation_1.py (courbes trame par trame), puis on résume chaque
courbe par 4 statistiques : moyenne, écart-type, min, max.

Colonnes du CSV final :
  - itemid  : identifiant du fichier (à exclure des variables explicatives)
  - dataset : "ff" (ff1010bird) ou "war" (warblrb10k)
  - hasbird : 1 = oiseau présent, 0 = absent (variable à prédire)
  - 5 descripteurs x 4 stats      = 20 colonnes
  - 13 MFCC        x 4 stats      = 52 colonnes
  -> 72 colonnes de descripteurs

Utilisation :
  python creer_dataset.py          # tous les fichiers (~15 700)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

# --- Chemins (relatifs à l'emplacement de ce script) -------------------------
DOSSIER = Path(__file__).resolve().parent   # dossier qui contient ce script
DATA = DOSSIER / "data"                     # "/" assemble les chemins
JEUX = [
    # (nom court, dossier des .wav, fichier de métadonnées)
    ("ff",  DATA / "wav_ff",  DATA / "ff1010bird_metadata_2018.csv"), # Chemin d'accès au enregistrement audio et au étiquettes pour "ff"
    ("war", DATA / "wav_war", DATA / "warblrb10k_public_metadata_2018.csv"), # Chemin d'accès au enregistrement audio et au étiquettes pour "war"
]
SORTIE = DOSSIER / "features_oiseaux.csv" # Nom du fichier final

# --- Paramètres d'analyse (identiques à Test_transformation_1.py) ------------
SR = 22050          # tous les fichiers sont rééchantillonnés à 22050 Hz
FRAME_SIZE = 1024   # Taille d'une trame (valeurs simplifiant les transformations de fourrier pour les MFCC)
HOP_LENGTH = 512    # Taille d'une trame : 1024 valeurs ≈ 46 ms (puissance de 2 : transformée de Fourier plus rapide)
N_MFCC = 13         # Nombre de coefficients MFCC 



def amplitude_envelope(signal, frame_size, hop_length):
    """Amplitude max dans chaque trame (même définition que le notebook)."""
    return np.array([signal[i:i + frame_size].max()
                     for i in range(0, len(signal), hop_length)]) 
# i = début de chaque trame : 0, 512, 1024, ... (pas = hop_length)
# signal[i:i + frame_size] = les 1024 valeurs de la trame 
# On récupère le vecteur des maximums sur les trames.

def resumer(nom, courbe):
    """Résume une courbe (valeur par trame) en moyenne / écart-type / min / max."""
    return {
        f"{nom}_mean": float(np.mean(courbe)),
        f"{nom}_std":  float(np.std(courbe)),
        f"{nom}_min":  float(np.min(courbe)),
        f"{nom}_max":  float(np.max(courbe)),
    }
# Résume une courbe (une valeur par trame) en 4 statistiques 

def extraire_features(chemin_wav):
    """Charge un fichier audio et renvoie un dictionnaire {nom_colonne: valeur}."""
    y, sr = librosa.load(chemin_wav, sr=SR)

    feats = {}

    # Domaine temporel
    feats |= resumer("ae", amplitude_envelope(y, FRAME_SIZE, HOP_LENGTH)) # Extraction de l'amplitude 
    feats |= resumer("rms", librosa.feature.rms(
        y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]) # Extraction RMS (volume moyen)
    feats |= resumer("zcr", librosa.feature.zero_crossing_rate(
        y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]) # Extraction de zcr, proportion de fois ou le signal change de de signe.

    # Domaine fréquentiel
    feats |= resumer("centroid", librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0])  # Fréquence moyenne du spectre pondéré par l'énergie (Hz).
    feats |= resumer("bandwidth", librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0])  # l’étalement du spectre autour du centroïde, en Hz

    # MFCC : matrice (13, n_trames) -> chaque ligne résumée comme les autres
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                n_fft=FRAME_SIZE, hop_length=HOP_LENGTH) 
    for k in range(N_MFCC):
        feats |= resumer(f"mfcc{k + 1}", mfcc[k]) # Treize valeurs MFCC par trames, puis résumer par "resumer"

    return feats


def traiter_un_fichier(nom_jeu, dossier_wav, itemid, hasbird):
    """Extrait les features d'un fichier ; renvoie None si le fichier pose problème."""
     # Lien étiquette <-> audio : l'itemid du CSV de métadonnées donne le nom du .wav
    chemin = dossier_wav / f"{itemid}.wav"

    # Un fichier manquant ou corrompu ne doit pas arrêter le traitement des
    # environ 15 700 fichiers : on signale l'erreur et on passe au suivant.
    try:
        feats = extraire_features(chemin)
    except Exception as e:
        print(f"  [ignoré] {chemin.name} : {e}")
        return None
    # Ligne finale : 3 colonnes d'identification + 72 descripteurs
    # (**feats recopie toutes les paires clé/valeur du dictionnaire feats)
    return {"itemid": itemid, "dataset": nom_jeu, "hasbird": int(hasbird), **feats}


def main():
# Liste des tâches : un quadruplet par fichier audio
    taches = []
    for nom_jeu, dossier_wav, csv_meta in JEUX:
        # itemid lu comme du texte : il doit rester exactement le nom du fichier
        # (sinon pandas pourrait transformer "00123" en 123)
        meta = pd.read_csv(csv_meta, dtype={"itemid": str})
        print(f"{nom_jeu} : {len(meta)} fichiers")
        # Chaque quadruplet = les 4 arguments de traiter_un_fichier
        taches += [(nom_jeu, dossier_wav, r.itemid, r.hasbird)
                   for r in meta.itertuples()]

    debut = time.time()
    # Les fichiers sont indépendants : on les répartit sur tous les cœurs (n_jobs=-1).
    # delayed(...) prépare l'appel sans l'exécuter ; *t déballe le quadruplet.
    # Le résultat garde l'ordre de "taches" ; verbose=5 affiche la progression.
    lignes = Parallel(n_jobs=-1, verbose=5)(
        delayed(traiter_un_fichier)(*t) for t in taches)

    # Les fichiers en erreur ont renvoyé None : on les retire
    lignes = [l for l in lignes if l is not None]

# Liste de dictionnaires -> DataFrame : une ligne par dict, une colonne par clé
    df = pd.DataFrame(lignes)
    df.to_csv(SORTIE, index=False) # index=False : pas de colonne de numéros de ligne

# Bilan : taille du tableau, durée, fichiers ignorés
    print(f"\n{len(df)} lignes x {df.shape[1]} colonnes "
          f"enregistrées dans {SORTIE.name} ({time.time() - debut:.0f} s)")
    if len(df) < len(taches):
        print(f"{len(taches) - len(df)} fichier(s) ignoré(s)")

# main() ne s'exécute que si on lance ce fichier, pas si on l'importe.
# Indispensable avec joblib : chaque processus parallèle réimporte ce script,
# et sans cette condition, chacun relancerait main() à son tour.
if __name__ == "__main__":
    main()
