------------------------------------------------------------------------

editor_options: markdown: wrap: 72 ---

# Détection de chants d'oiseaux par apprentissage automatique

## Présentation du projet

Notre projet porte sur la **classification supervisée d'enregistrements audio**, appliquée à la détection d'oiseaux dans l'environnement à partir de leurs chants.

L'enjeu avec ce type de données est d'en extraire des **descripteurs audio** : enveloppe d'amplitude, RMS, taux de passage par zéro (*zero crossing rate*), centroïde spectral, largeur de bande spectrale et coefficients MFCC. Ces descripteurs servent de variables explicatives dans les modèles de machine learning que nous utilisons pour prédire si un enregistrement contient un chant d'oiseau ou non.

Nous mettons en œuvre plusieurs modèles : régression logistique (simple et régularisée), SVM (linéaire et à noyau gaussien), arbre de décision, forêt aléatoire, XGBoost et un réseau de neurones (MLP). Nous comparons leurs performances pour retenir le meilleur modèle prédictif.

## Contenu du dossier

| Fichier | Rôle |
|------------------------------------|------------------------------------|
| `creer_dataset.py` | Extrait les descripteurs audio de chaque enregistrement et construit `features_oiseaux.csv` |
| `features_oiseaux.csv` | Jeu de données prêt à l'emploi : une ligne par enregistrement, 72 descripteurs |
| `ML.ipynb` | Analyse : exploration, entraînement, réglage et comparaison des modèles |
| `README.md` | Ce fichier |

Les enregistrements audio (environ 15 Go) ne sont pas inclus. Le fichier `features_oiseaux.csv` suffit pour exécuter le notebook. Les fichiers audio ne sont nécessaires que pour recréer ce CSV.

## Les données

Les enregistrements proviennent du **challenge DCASE 2018, tâche 3 « Bird Audio Detection »**. Nous utilisons les deux jeux de développement, étiquetés :

| Jeu | Enregistrements | Origine |
|------------------------|------------------------|------------------------|
| **freefield1010** (`ff`) | 7 690 | enregistrements de terrain variés |
| **Warblr** (`war`) | 8 000 | enregistrements faits avec des smartphones par le public au Royaume-Uni |

Chaque enregistrement dure environ 10 secondes et porte une étiquette `hasbird` : 1 s'il contient un chant d'oiseau, 0 sinon.

### Téléchargement

- Page du challenge : <https://dcase.community/challenge2018/task-bird-audio-detection>
- Audio freefield1010 : <https://archive.org/download/ff1010bird/ff1010bird_wav.zip>
- Audio Warblr : <https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip>
- Étiquettes (fichiers `*_metadata_2018.csv`) : <https://figshare.com/articles/Bird_Audio_Detection_public_data_for_DCASE_Challenge_2018_Task_3/6026471>

### Organisation attendue

Pour que `creer_dataset.py` trouve les fichiers, il faut les ranger dans un dossier `data/`, placé à côté du script :

```         
projet/
├── creer_dataset.py
├── features_oiseaux.csv
├── ML.ipynb
├── README.md
└── data/
    ├── wav_ff/                              ← fichiers .wav de freefield1010
    ├── wav_war/                             ← fichiers .wav de Warblr
    ├── ff1010bird_metadata_2018.csv
    └── warblrb10k_public_metadata_2018.csv
```

Les archives audio se décompressent dans un dossier `wav/` : il faut le renommer en `wav_ff/` ou `wav_war/` selon le jeu.

## Utilisation

### 1. Créer le jeu de données (facultatif)

Cette étape n'est utile que pour recréer `features_oiseaux.csv` à partir des enregistrements audio. Elle nécessite Python 3.9 ou plus récent, ainsi que les bibliothèques `librosa` et `joblib`.

``` bash
python creer_dataset.py
```

Pour chaque enregistrement, le script : 1. charge le fichier audio et le rééchantillonne à 22 050 Hz ; 2. le découpe en trames de 1 024 valeurs (environ 46 ms) qui se chevauchent de moitié ; 3. calcule, trame par trame, 18 descripteurs : enveloppe d'amplitude, RMS, taux de passage par zéro, centroïde spectral, largeur de bande et 13 coefficients MFCC ; 4. résume chaque courbe obtenue par 4 statistiques : moyenne, écart-type, minimum et maximum.

On obtient 18 × 4 = **72 descripteurs** par enregistrement. Les fichiers sont traités en parallèle sur tous les cœurs du processeur, et un fichier illisible est ignoré sans arrêter le traitement.

Colonnes du CSV : - `itemid` : identifiant de l'enregistrement ; - `dataset` : jeu d'origine (`ff` ou `war`) ; - `hasbird` : variable à prédire (1 = oiseau, 0 = pas d'oiseau) ; - 72 colonnes de descripteurs, nommées `<descripteur>_<statistique>`, par exemple `zcr_max` ou `mfcc3_mean`.

`itemid` et `dataset` ne sont pas utilisés comme variables explicatives.

### 2. Analyse (`ML.ipynb`)

Le notebook lit `features_oiseaux.csv` dans le même dossier. Sa première cellule installe les bibliothèques nécessaires.

Il suit les étapes suivantes :

1.  **Exploration** : vérification de l'import et du type des variables, équilibre des classes (environ 51 % d'enregistrements avec oiseau) et distribution des descripteurs. Les boîtes à moustaches montrent des différences d'échelle importantes entre les MFCC et les autres descripteurs.

2.  **Découpage des données** : 80 % pour l'entraînement, 20 % pour le test, en conservant la proportion d'oiseaux dans les deux jeux. Les variables sont ensuite standardisées à partir du jeu d'entraînement seulement, pour corriger les différences d'échelle.

3.  **Modélisation** : comme les classes sont équilibrées, la métrique retenue est l'**accuracy** (taux de bien classés). Chaque modèle suit la même procédure :

    - une validation croisée stratifiée à 5 plis, avec les hyperparamètres par défaut ;
    - un réglage des hyperparamètres par recherche aléatoire (`RandomizedSearchCV`), en validation croisée stratifiée à 5 plis sur le jeu d'entraînement ;
    - le calcul de l'accuracy sur le jeu de test, pour le modèle par défaut et pour le modèle réglé ;
    - la matrice de confusion et le rapport de classification du modèle réglé.

    La régression logistique simple sert de point de référence. Les modèles comparés sont : régression logistique régularisée, arbre de décision, forêt aléatoire, SVM linéaire, SVM à noyau gaussien et XGBoost.

4.  **Réseau de neurones** : un MLP à deux couches cachées (16 puis 8 neurones), entraîné avec Keras pendant 20 épochs, avec 20 % du jeu d'entraînement réservés à la validation.

5.  **Tableau récapitulatif** : accuracy d'entraînement et de test de chaque modèle, avec les hyperparamètres par défaut et avec les hyperparamètres réglés.

6.  **Conclusion**.

Le jeu de test n'intervient jamais dans le choix des hyperparamètres : il sert uniquement à mesurer la performance finale de chaque modèle.

## Principaux résultats

Les descripteurs audio permettent de bien prédire la présence de chants d'oiseaux : la plupart des modèles atteignent une accuracy de test d'environ 82 à 83 %. Le SVM à noyau gaussien et le MLP à deux couches cachées obtiennent les meilleurs résultats, avec plus de 84 % d'accuracy sur le jeu de test et peu de surapprentissage.

## Documentation

- librosa (extraction des descripteurs audio) : <https://librosa.org/doc/latest/index.html>
- Vidéo YouTube : <https://www.youtube.com/watch?v=m3XbqfIij_Y>

## Auteurs

Tao LEVRAY, Quentin ROUSSELIN, Clément JOURDREN.
