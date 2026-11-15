# Détection et localisation de vaisseaux sur des élastogrammes

Projet Machine Learning / Deep Learning (M2), en partenariat avec Echosens.
Camille Hodé et Satine. Présentation le 20 novembre 2026.

## Objectif

Sur une image d'élastographie impulsionnelle (119 ou 129 lignes de profondeur
sur 481 colonnes de temps), deux questions :

1. **Détection** : un vaisseau (`vessel` ou `supervessel`) est-il présent ?
2. **Localisation** : si oui, à quelle profondeur (quelle ligne de l'image) ?

## Modèle retenu

Un U-Net dont l'encodeur ResNet34 est initialisé avec les poids ImageNet, entraîné
20 époques. Le même réseau a deux sorties :

- une **tête de classification** qui donne la probabilité de présence ;
- une **carte de segmentation** moyennée sur la largeur. On obtient un score par
  ligne, transformé en distribution de probabilité sur la profondeur. La profondeur
  prédite est la médiane de cette distribution autour de son pic.

Perte : entropie croisée binaire pour la présence, plus une entropie croisée entre
la bande annotée et la distribution prédite pour la profondeur (calculée seulement
sur les images avec vaisseau). Les classes sont équilibrées à l'intérieur de chaque
sonde pendant l'entraînement. Augmentations : recadrage horizontal et légère
variation de contraste. Le modèle final est l'ensemble de 3 graines (moyenne de
leurs prédictions).

Ce modèle a été choisi parmi 6 configurations : poids ImageNet ou encodeur
pré-entraîné sur les ondes de cisaillement, à 20, 40 ou 60 époques. Aucune autre
configuration n'a fait mieux de façon démontrée. Nous avons donc gardé la plus
simple et la plus courte à entraîner, qui était la configuration de référence
fixée avant les résultats.

## Protocole d'évaluation

- **963 images, 94 examens**, dont 623 images avec vaisseau. Les images au label
  incertain ou incohérent sont exclues.
- **Validation croisée en 5 plis groupés par examen** : chaque image est prédite
  par un modèle qui n'a jamais vu son examen.
- **Calibration interne** : dans chaque pli, environ 1/6 des examens d'entraînement
  sont mis de côté. Ils n'entraînent pas le réseau et servent à fixer le seuil de
  présence de chaque sonde, visé à 90 % de spécificité.
- **Nombre d'époques fixe**, sans sélection d'époque sur les données de test.
- **Référence pour la localisation** : une « ligne fixe », c'est-à-dire la même
  profondeur pour toutes les images d'une sonde (la médiane du train).
- **Intervalles de confiance à 95 %** par bootstrap sur les examens.

## Résultats (ensemble des 3 graines)

| | Sonde 7 | Sonde 9 |
|---|---|---|
| Images (dont avec vaisseau) | 724 (553) | 239 (70) |
| **AUC de détection** | 0,735 [0,690 – 0,782] | 0,733 [0,648 – 0,816] |
| Sensibilité au seuil calibré | 46,1 % | 35,7 % |
| Spécificité obtenue (visée : 90 %) | 87,1 % | 88,8 % |
| **Profondeur dans la bande annotée** | 82,3 % | 75,7 % |
| Ligne fixe (référence) | 62,2 % | 74,3 % |
| Écart au modèle [IC95] | +20,1 pts [+15,6 ; +24,6] | +1,4 pts [−7,4 ; +10,9] |
| Bout en bout (détecté ET bien localisé) | 39,8 % | 27,1 % |

Pour lire le tableau :

- La **détection** est modérée : AUC autour de 0,73. Au seuil qui garde environ
  10 % de fausses alarmes, le modèle trouve un peu moins d'un vaisseau sur deux.
- La **localisation** est le point fort sur la sonde 7 : le modèle place la
  profondeur dans la bande annotée nettement plus souvent que la ligne fixe.
  Sur la sonde 9, il n'y a que 70 images avec vaisseau, et aucune différence
  n'est démontrée.
- Les courbes d'apprentissage montrent que la loss de validation stagne dès la
  4e époque environ, alors que la loss d'entraînement continue de baisser. Cela
  explique pourquoi entraîner plus longtemps n'a pas amélioré les résultats.

Le détail est dans [`sorties/final/bilan_final.txt`](sorties/final/bilan_final.txt).

## Contenu du dépôt

| Fichier | Contenu |
|---|---|
| `modele_final.py` | Script complet : données, modèle, entraînement, évaluation, figures |
| `sorties/final/bilan_final.txt` | Bilan rédigé : détection, localisation, bout en bout |
| `sorties/final/resume_final.csv` | Chiffres principaux par graine et pour l'ensemble |
| `sorties/final/profondeur_final.png` | Profondeur prédite vs annotée, modèle et ligne fixe |
| `sorties/final/courbes_final.png` | Loss d'entraînement et de validation, learning rate |
| `sorties/final/seuils_final.json` | Seuils de présence calibrés, par pli et par sonde |

## Données

Les élastogrammes et leurs annotations appartiennent à Echosens et **ne sont pas
fournis**. Le script attend la structure suivante :

```
Annotations/imgtag.annotations.json
Elastos/*.bmp et Elastos/*.txt
sorties/final/partitions.json      (partitions figées, fournies avec les données)
```

## Utilisation

Avec Python 3 (testé avec 3.14) et `torch`, `torchvision`, `segmentation-models-pytorch`,
`scikit-learn`, `numpy`, `pandas`, `Pillow` et `matplotlib` installés :

```bash
python modele_final.py --test-rapide   # 1 époque, 1 pli : vérifie l'installation
python modele_final.py --entrainer     # 3 graines x 5 plis, puis évaluation (~70 min/pli sur CPU, bien moins sur GPU)
python modele_final.py --evaluer       # recalcule bilan et figures à partir des prédictions sauvegardées
```

Versions utilisées : entraînement avec torch 2.14 sur GPU NVIDIA et scikit-learn 1.9 ;
évaluation vérifiée avec torch 2.15, scikit-learn 1.9.1 et segmentation-models-pytorch 0.6. L'entraînement sur GPU n'est pas parfaitement
déterministe : un nouvel entraînement donne des chiffres très proches, mais pas
identiques au dernier chiffre.
