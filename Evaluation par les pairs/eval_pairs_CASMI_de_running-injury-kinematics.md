---
title: Evaluation par les pairs

---

### Grille d’évaluation croisée

:::info
À remplir par chaque groupe pour chaque autre groupe, sur une échelle de 1 à 5, avec 5 = excellent). A remplir pour le **19 novembre** impérativement.
:::

==Groupe évaluateur== : CASMI (Fanny, Antonin et Inès)
==Groupe évalué== : Détection des blessures du genou chez le coureur, running-injury-kinematics (Youri, Lilou et Charles)

| **Critère** | **Sous-critères** | **Note (1-5)** | **Commentaires** |
|---|---|---|---|
| **1. Pertinence du thème** | Le thème choisi est-il original et complémentaire aux enseignements existants ? | 5 | Sujet bien choisi : déséquilibre de classes, explicabilité, facteur de confusion. Même jeu de données que Fuentes-Jiménez et al. (2026), apport propre avec la vérification du biais d'époque. |
| | L’ancrage en sciences du vivant est-il justifié et bien illustré ? | 5 | Oui, cas clinique de médecine, donc bien ancré en biologie. |
| **2. Structure et clarté** | La présentation suit-elle une logique pédagogique (méthodologie → technique → application) ? | 4 | Démarche facile à suivre. Partie deep learning moins aboutie que le reste. |
| | Le niveau de détail permet-il une bonne compréhension du sujet ? | 4 | Démarche claire, mais positions relatives des capteurs parfois difficiles à comprendre. |
| **3. Qualité méthodologique** | Les concepts clés sont-ils expliqués de manière accessible mais rigoureuse ? | 4 | Exactitude équilibrée bien amenée avec le piège du « toujours sain » (61,5 % de bonnes réponses). Différence entre détecter et prédire explicitée. SHAP, AUC et odds ratios pas définis sur les slides. |
| | Les choix méthodologiques sont-ils argumentés ? | 4 | Choix sur les données argumentés (bibliographie et données). Choix des modèles moins justifié, hyperparamètres par défaut. |
| **4. Rigueur technique** | Le code (GitHub) est-il fonctionnel et commenté | 3 | Notebooks R commentés presque ligne à ligne, scripts de préparation documentés. requirements.txt incomplet : ni torch, ni scikit-learn, ni matplotlib, ni paquets R. Chemins absolus propres à un poste dans les scripts du deep learning. |
| | Les résultats sont-ils reproductibles (par ex. y a t-il des graines pour générer les échantillons ?) | 4 | Graine fixée dans tous les notebooks R (set.seed(2026)). Plis de validation croisée figés dans plis.csv, communs à R et Python. Aucune graine dans le code Python du deep learning. |
| | Les données sont-elles bien documentées et adaptées à l’application ? | 5 | Très bien documentées : source, DOI, licence CC BY 4.0. Passage de 1 832 séances à 587 coureurs justifié étape par étape. Dictionnaire des variables avec unités. Pièges (fuite d'information, effet d'époque) signalés avant la modélisation. |
| | Y a-t-il des erreurs techniques ou des approximations ? | 3 | Partie classique propre : une séance par coureur, prétraitements appris dans chaque pli, donc pas de fuite. Écarts entre modèles dans la variabilité entre plis. Deep learning : un seul découpage 70/15/15 au lieu des plis communs annoncés, évalué au dernier epoch et pas au meilleur. |
| **5. Ancrage applicatif** | L’application en sciences du vivant est-elle pertinente et bien illustrée ? | 4 | Fil rouge d'Audrey, résultats confrontés à la physiopathologie : surtout foulée, pied et bassin, hanche moins importante que prévu. Application réelle (capteurs portables) encore hypothétique. |
| | Les limites de la méthode/thème sont-elles mentionnées ?? | 4 | Limites principales citées : coureurs déjà blessés (détection, pas prédiction), époque confondue avec le diagnostic. Manque la performance faible (exactitude équilibrée autour de 0,60), insuffisante pour un usage clinique. |
| **6. Qualité des supports** | Les slides sont-ils synthétiques et d'une forme adaptée ? | 4 | Slides claires et faciles à lire, palette de couleurs bien choisie. Quelques slides chargées (tableaux de résultats). |
| | Le dépôt GitHub est-il organisé (README, structure claire, données accessibles) ? | 3 | Dépôt fait tard et incomplet : requirements.txt partiel, deux versions différentes de charte_graphique.py, scripts du deep learning non documentés. |
| | Les documents partagés en amont (code, données, slides) sont-ils utilisables et complets pour une réutilisation ? | 4 | coureurs.csv, plis.csv et dictionnaire des variables réutilisables tels quels. Dépendances pas toutes déclarées. |
| | Les sources sont-elles citées et fiables ? | 4 | Une vingtaine de références, surtout à comité de lecture. Sources lues seulement en résumé signalées honnêtement. Quelques slides sans source. |
