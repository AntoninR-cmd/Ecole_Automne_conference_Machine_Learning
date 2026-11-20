# Ecole d'Automne - 8 conférences sur l'Intelligence Artificielle et le Machine Learning dans la Science du vivant.

Le 20 novembre 2026 aura lieu un après-midi de conférences sur la réalisation de projets de Machine Learning dans les sciences du vivant, en salle D4, bâtiment 24.
Les 8 conférences de 20 min suivantes auront lieu de 13h30 à 16h30 :

- Identifier des molécules inconnues à partir de leur spectre de masse : une approche par apprentissage automatique (Fanny, Antonin et Inès)
- Identification des blessures du genou les plus fréquentes chez le coureur (Youri, Lilou et Charles)
- Prédiction de la résistance aux antibiotiques (Baptiste, Alexandre et Benjamin)
- Comment croiser des données hétérogènes pour anticiper les départs de feu ? (Jules et Pierre)
- Détection et localisation automatiques des fractures sur clichés radiographies par approche YOLO (Chiara, Anatole et Augustin C.)
- Reconnaissance d'images de graines : jusqu'où un modèle peut-il généraliser ? (Anton, Edgar et Augustin B.)
- Détection automatique de chants d’oiseaux dans des enregistrements audio (Tao, Quentin et Clement)
- LIVER WARS : à la recherche des vaisseaux perdus (Hodé, Satine et Camille)

Ces conférences sont ouvertes au public. Vous pouvez retrouver plus d'informations sur le programme, les conférenciers, la date, le lieu et l'heure sur le site web suivant :
https://ecole-automne-data-science.netlify.app/

## Règles pour les collaborateurs du projet GitHub

- Travaillez sur votre propre branche, nommée d'après votre nom ou votre projet.
- Créez à la racine un dossier au nom de votre projet et importez-y votre projet de Machine Learning.
- Ne modifiez que votre dossier projet, ainsi que vos propres fichiers dans `Auto-évaluation/` et `Evaluation par les pairs/`.
- Déposez votre auto-évaluation dans `Auto-évaluation/`, sous le nom `auto-eval_[projet]_[participants]`
  (ex. `auto-eval_CASMI_Fanny-Antonin-Ines.md`).
- Déposez l'évaluation du projet qui vous est attribué dans `Evaluation par les pairs/`, sous le nom
  `eval_pairs_[votre projet]_de_[projet évalué]` (ex. `eval_pairs_CASMI_de_running-injury-kinematics.md`).
- `[projet]` est le nom court donné dans le tableau ci-dessous ; les participants sont séparés par des tirets, sans accents.
- Ne fusionnez rien dans `main` : les pull requests sont gérées par Antonin.
- Date limite : 20 novembre 2026.

## Déposer vos évaluations depuis votre branche

Votre branche ne contient que ce README et votre dossier : les dossiers `Auto-évaluation/` et `Evaluation par les pairs/` n'y existent pas. **Inutile de récupérer `main`** : créez à la racine de votre branche des dossiers portant **exactement** le même nom, et mettez-y uniquement vos fichiers. Lors de la fusion dans `main`, Git range vos fichiers dans les dossiers existants, à côté de ceux des autres groupes. Comme chaque nom de fichier est unique, il n'y a pas de conflit.

```bash
# depuis la racine du dépôt, sur votre branche
mkdir -p "Auto-évaluation" "Evaluation par les pairs"
cp chemin/vers/votre-auto-eval.md "Auto-évaluation/auto-eval_[projet]_[participants].md"
cp chemin/vers/votre-evaluation.md "Evaluation par les pairs/eval_pairs_[votre projet]_de_[projet évalué].md"
git add "Auto-évaluation" "Evaluation par les pairs"
git commit -m "Auto-évaluation et évaluation croisée"
git push
```

Puis ouvrez une pull request vers `main`. Sur le site GitHub, vous pouvez aussi faire « Add file > Create new file » et taper le chemin complet, par exemple `Auto-évaluation/auto-eval_CASMI_Fanny-Antonin-Ines.md` : le `/` crée le dossier.

Points d'attention :

- Le nom des dossiers doit être identique au caractère près (accent de « é », espaces, majuscules), sinon un second dossier est créé.
- Ne modifiez pas ce README sur votre branche : il est mis à jour sur `main`, et toute modification de votre côté crée un conflit.
- Les évaluations déjà déposées dans un dossier projet ont été copiées et renommées dans ces deux dossiers : inutile de les redéposer.

## Évaluation croisée

| Groupe | Projet (nom court) | Évalue | Projet évalué |
|---|---|---|---|
| Fanny, Antonin, Inès | CASMI | Youri, Lilou, Charles | running-injury-kinematics |
| Youri, Lilou, Charles | running-injury-kinematics | Baptiste, Alexandre, Benjamin | Prediction-resistance-antibiotique |
| Baptiste, Alexandre, Benjamin | Prediction-resistance-antibiotique | Jules, Pierre | feux-mediterraneens |
| Jules, Pierre | feux-mediterraneens | Chiara, Anatole, Augustin C | Detection-fractures-YOLO |
| Chiara, Anatole, Augustin C | Detection-fractures-YOLO | Anton, Edgar, Augustin B | Reconnaissance-graines |
| Anton, Edgar, Augustin B | Reconnaissance-graines | Tao, Quentin, Clement | Classification-audio-oiseaux |
| Tao, Quentin, Clement | Classification-audio-oiseaux | Hodé, Satine, Camille | LIVER-WARS |
| Hodé, Satine, Camille | LIVER-WARS | Fanny, Antonin, Inès | CASMI |
