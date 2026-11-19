---
title: Evaluation par les pairs

---

### Grille d’évaluation croisée

:::info
À remplir par chaque groupe pour chaque autre groupe, sur une échelle de 1 à 5, avec 5 = excellent). A remplir pour le **19 novembre** impérativement.
:::

==Groupe évaluateur== : CASMI (Fanny, Antonin et Inès)
==Groupe évalué== : Détection des blessures du genou chez le coureur, running-injury-kinematics (Youri, Lilou et Charles)

   **Critère**                          | **Sous-critères**                                                                 | **Note (1-5)** | **Commentaires**                     |
 |--------------------------------------|-----------------------------------------------------------------------------------|----------------|--------------------------------------|
 | **1. Pertinence du thème**           | Le thème choisi est-il original et complémentaire aux enseignements existants ?       | 5 | Sujet bien choisi, qui mobilise déséquilibre de classes, explicabilité et facteur de confusion. Apport propre par rapport à l'article de Fuentes-Jiménez et al. (2026), qui traite le même jeu de données, avec la vérification du biais d'époque. |
 |           |  L’ancrage en sciences du vivant est-il justifié et bien illustré ?      | 5 | Oui : un cas clinique de médecine, donc bien ancré en biologie. |
 | **2. Structure et clarté**           | La présentation suit-elle une logique pédagogique (méthodologie → technique → application) ?      |                |                                      |
  |           | Le niveau de détail permet-il une bonne compréhension du sujet ?      |                |                                      |
 | **3. Qualité méthodologique**        | Les concepts clés sont-ils expliqués de manière accessible mais rigoureuse ?        | 5 | Concepts expliqués de façon accessible et rigoureuse : l'exactitude équilibrée est introduite par le piège du « toujours sain » (61,5 % de bonnes réponses), et la différence entre détecter et prédire est explicitée. Seule réserve : SHAP n'est défini qu'en annexe. |
  |        | Les choix méthodologiques sont-ils argumentés ?         |                |                                      |
 | **4. Rigueur technique**             | Le code (GitHub) est-il fonctionnel et commenté  | 3 | Notebooks R commentés presque ligne à ligne et scripts de préparation documentés. En revanche, requirements.txt ne déclare ni torch, ni scikit-learn, ni matplotlib, ni les paquets R, et les scripts du deep learning contiennent des chemins absolus propres à un poste. |
  |             | Les résultats sont-ils reproductibles (par ex. y a t-il des graines pour générer les échantillons ?) | 5 | Graine fixée dans tous les notebooks R (set.seed(2026)) et plis de validation croisée figés dans plis.csv, partagés entre R et Python. Les résultats sont versionnés dans resultats/. |
  |             | Les données sont-elles bien documentées et adaptées à l’application ?  | 5 | Données très bien documentées : source, DOI et licence CC BY 4.0, passage de 1 832 séances à 587 coureurs justifié étape par étape, dictionnaire des variables avec unités. Les pièges (fuite d'information, effet d'époque) sont signalés avant la modélisation. |
   |            | Y a-t-il des erreurs techniques ou des approximations ?  | 4 | Partie classique rigoureuse : une seule séance par coureur et des prétraitements appris dans chaque pli, ce qui évite les fuites. Les approximations concernent surtout le deep learning : un seul découpage 70/15/15 au lieu des plis communs annoncés, et une évaluation au dernier epoch plutôt qu'au meilleur. |
| **5. Ancrage applicatif**            | L’application en sciences du vivant est-elle pertinente et bien illustrée ?  |                |                                      |
|        | Les limites de la méthode/thème sont-elles mentionnées ??  | 4 | Les limites essentielles sont citées : coureurs déjà blessés (on détecte sans prédire) et époque confondue avec le diagnostic. Il manque la faiblesse de la performance (exactitude équilibrée autour de 0,60), insuffisante pour un usage clinique. |
 | **6. Qualité des supports**          | Les slides sont-ils synthétiques et d'une forme adaptée ? | |
  |          | Le dépôt GitHub est-il organisé (README, structure claire, données accessibles) ? | |
   |           | Les documents partagés en amont (code, données, slides) sont-ils utilisables et complets pour une réutilisation ? | |
|          | Les sources sont-elles citées et fiables ? | 