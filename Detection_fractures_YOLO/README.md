# Détection et localisation automatiques des fractures sur radiographies par approche YOLO

## Contexte
Détection de fractures du poignet pédiatrique sur radiographies, avec le dataset GRAZPEDWRI-DX.

## Équipe
- Chiara Thomé
- Anatole Levrel
- Augustin Charbonnier

## Méthode
- Modèles : YOLOv8, YOLO11, YOLO26 (versions Nano)
- Données : échantillon rééquilibré([10 % / 90 %) entre images négatives et positives, mêmes proportions en train, val et test
- Entraînement avec Ultralytics :

| Essai | Modèle YOLO | Paramètres clés |
|---|---|---|
| Baseline | yolo11n.pt | epochs=200, imgsz=640, batch=16 |
| Run 2 | yolo11n.pt | epochs=200, imgsz=864, batch=16 |
| Run 3 | yolo11n.pt | epochs=200, imgsz=1024, batch=16 |
| Run 4 | yolo11n.pt | epochs=200, imgsz=1256, batch=16 |
| Run 5 | yolo11s.pt | epochs=200, imgsz=640, batch=16 |
| Run 6 | yolo26n.pt | epochs=200, imgsz=640, batch=16 |
| Run 7 | yolov8n.pt | epochs=200, imgsz=640, batch=16 |



## Structure du dossier
```
Detection_fractures_YOLO/
├── README.md
├── requirements.txt
├── data.yaml
├── src/            # scripts de préparation, entraînement, évaluation
├── results/        # figures et métriques
└── docs/           # slides, auto-évaluation, évaluation croisée
```

## Utilisation
```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```
Le dataset n'est pas inclus : https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193.

## Résultats

À date : 
Tous les essais : `epochs=200`, `batch=16`.

| Essai | Modèle YOLO | Paramètres clés | Résultats / Commentaires |
|---|---|---|---|
| Baseline | yolo11n.pt | imgsz=640 | best.pt après 72 époques, 1h30, mAP50 de 0,89, mAP50-95 de 0,47, précision 0,90, rappel 0,82 : détection fiable, mais rappel et localisation fine des boîtes à améliorer. |
| Run 2 | yolo11n.pt | imgsz=864 | upcoming |
| Run 3 | yolo11n.pt | imgsz=1024 | best.pt après 74 époques, performances équivalentes. |
| Run 4 | yolo11n.pt | imgsz=1256 | best.pt à 54 époques, 2h30, performances équivalentes. |

| Modèle | mAP50 | mAP50-95 | Précision | Rappel |
|---|---|---|---|---|
| YOLOv8n | | | | |
| YOLO11n | | | | |
| YOLO26n | | | | |

## Références
- Dataset GRAZPEDWRI-DX : https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193
- Ultralytics YOLO : https://docs.ultralytics.com
