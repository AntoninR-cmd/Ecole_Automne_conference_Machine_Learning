
# ============================================================
# RECADRAGE AUTOMATIQUE DES IMAGES DE GRAINES
# Objectif : obtenir des images de 968 x 648 pixels
# ============================================================

from pathlib import Path
import shutil
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\fanos\Documents\Cours\Projet ML\Python"
)

DATA_DIR = PROJECT_DIR / "DATA"
OUTPUT_DIR = PROJECT_DIR / "TRONC_DATA"

SPECIES = ["OLEA_ARCHEO","PHOENIX_ARCHEO","VITIS_ARCHEO"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# Dimensions finales obligatoires
TARGET_WIDTH = 968
TARGET_HEIGHT = 648

# Rapport largeur / hauteur
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT

# Détection du fond blanc
WHITE_THRESHOLD = 20

# Fermeture morphologique
MORPH_KERNEL_SIZE = 7

# Marges autour de la graine
PADDING_X = 0.15
PADDING_Y = 0.15
MIN_PADDING = 15

# Contrôle de détection
MIN_OBJECT_RATIO = 0.001
MAX_OBJECT_RATIO = 0.60

# Aperçu visuel
PREVIEW_ENABLED = True
N_PREVIEW = 8
RANDOM_SEED = 42

# Rapport
CSV_REPORT = OUTPUT_DIR / "crop_report.csv"


# ============================================================
# 2. DÉTECTION DE LA GRAINE
# ============================================================

def detect_seed(image):
    """
    Détecte la graine sur le fond blanc.

    Retourne :
        bbox : (x_min, y_min, x_max, y_max)
        status : OK / INCERTAIN / ECHEC
        details : informations sur la détection
    """

    rgb = np.array(image.convert("RGB"))

    height, width = rgb.shape[:2]
    total_pixels = height * width

    # Distance euclidienne au blanc
    white = np.array([255, 255, 255])

    distance = np.sqrt(
        np.sum(
            (rgb.astype(np.float32) - white) ** 2,
            axis=2
        )
    )

    # Segmentation
    mask = (
        distance > WHITE_THRESHOLD
    ).astype(np.uint8) * 255

    # Fermeture morphologique
    kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Composantes connexes
    n_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if n_labels <= 1:
        return None, "ECHEC", {
            "reason": "Aucun objet détecté",
            "object_ratio": 0
        }

    # Plus grande composante
    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_label = np.argmax(areas) + 1

    largest_area = stats[
        largest_label, cv2.CC_STAT_AREA
    ]

    object_ratio = largest_area / total_pixels

    # Bounding box
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]

    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    bbox = (x, y, x + w, y + h)

    reasons = []

    if object_ratio < MIN_OBJECT_RATIO:
        reasons.append("Objet trop petit")

    if object_ratio > MAX_OBJECT_RATIO:
        reasons.append("Objet trop grand")

    # Vérification des composantes secondaires
    sorted_areas = np.sort(areas)[::-1]

    if len(sorted_areas) > 1:
        second_ratio = sorted_areas[1] / largest_area
    else:
        second_ratio = 0

    if second_ratio > 0.15:
        reasons.append("Composantes secondaires importantes")

    if (
        x == 0 or y == 0
        or x + w >= width
        or y + h >= height
    ):
        reasons.append("Objet touchant le bord")

    details = {
        "object_ratio": object_ratio,
        "second_component_ratio": second_ratio,
        "reason": "; ".join(reasons)
    }

    if reasons:
        return bbox, "INCERTAIN", details

    return bbox, "OK", details


# ============================================================
# 3. CALCUL DE LA FENÊTRE DE RECADRAGE
# ============================================================

def calculate_crop_window(bbox, image_size):
    """
    Calcule un rectangle de recadrage :

    - centré sur la graine ;
    - contenant toute la bounding box ;
    - avec une marge de sécurité ;
    - au rapport 968 / 648 ;
    - entièrement situé dans l'image.

    La fenêtre est aussi grande que nécessaire,
    sans dépasser les dimensions disponibles.
    """

    x_min, y_min, x_max, y_max = bbox

    image_width, image_height = image_size

    object_width = x_max - x_min
    object_height = y_max - y_min

    # Centre de la graine
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Marges souhaitées
    margin_x = max(
        MIN_PADDING,
        int(object_width * PADDING_X)
    )

    margin_y = max(
        MIN_PADDING,
        int(object_height * PADDING_Y)
    )

    # Dimensions nécessaires pour contenir la graine
    required_width = object_width + 2 * margin_x
    required_height = object_height + 2 * margin_y

    # Adapter au rapport cible
    crop_width = max(
        required_width,
        required_height * TARGET_RATIO
    )

    crop_height = crop_width / TARGET_RATIO

    # Dimensions maximales possibles dans l'image
    max_crop_width = min(
        image_width,
        image_height * TARGET_RATIO
    )

    max_crop_height = max_crop_width / TARGET_RATIO

    # Réduire les marges si nécessaire
    crop_width = min(crop_width, max_crop_width)
    crop_height = crop_width / TARGET_RATIO

    # Vérifier que la graine entière tient dans la fenêtre
    if (
        object_width > crop_width
        or object_height > crop_height
    ):
        return None

    # Conversion en dimensions entières
    crop_width = int(round(crop_width))
    crop_height = int(round(crop_height))

    # Correction du rapport après arrondi
    crop_height = int(round(crop_width / TARGET_RATIO))

    crop_width = min(crop_width, image_width)
    crop_height = min(crop_height, image_height)

    # Position initiale centrée sur la graine
    x0 = int(round(center_x - crop_width / 2))
    y0 = int(round(center_y - crop_height / 2))

    # Contraintes de position pour ne jamais couper la graine
    min_x0 = max(0, x_max - crop_width)
    max_x0 = min(image_width - crop_width, x_min)

    min_y0 = max(0, y_max - crop_height)
    max_y0 = min(image_height - crop_height, y_min)

    if min_x0 > max_x0 or min_y0 > max_y0:
        return None

    x0 = int(np.clip(x0, min_x0, max_x0))
    y0 = int(np.clip(y0, min_y0, max_y0))

    x1 = x0 + crop_width
    y1 = y0 + crop_height

    return (x0, y0, x1, y1)


# ============================================================
# 4. REDIMENSIONNEMENT AU FORMAT FINAL
# ============================================================

def resize_to_target(image):
    """
    Produit une image de 968 x 648 pixels
    sans déformation géométrique.

    Si nécessaire, ajoute du blanc autour de l'image
    pour respecter le rapport cible.
    """

    return ImageOps.pad(
        image.convert("RGB"),
        (TARGET_WIDTH, TARGET_HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5)
    )


# ============================================================
# 5. SAUVEGARDE
# ============================================================

def save_image(image, output_path):
    """
    Sauvegarde en conservant le format d'origine
    lorsque celui-ci est pris en charge.
    """

    suffix = output_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        image.save(
            output_path,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=True
        )

    elif suffix == ".png":
        image.save(
            output_path,
            format="PNG",
            optimize=True
        )

    elif suffix in {".tif", ".tiff"}:
        image.save(output_path, format="TIFF")

    elif suffix == ".bmp":
        image.save(output_path, format="BMP")

    else:
        image.save(output_path)


# ============================================================
# 6. TRAITEMENT D'UNE IMAGE
# ============================================================

def process_image(source_path, output_path):

    try:

        with Image.open(source_path) as original:

            # Correction de l'orientation EXIF
            image = ImageOps.exif_transpose(original).convert("RGB")

        original_width, original_height = image.size

        # Cas particulier : image déjà au format cible
        if image.size == (TARGET_WIDTH, TARGET_HEIGHT):

            # Copie exacte du fichier original
            shutil.copy2(source_path, output_path)

            return {
                "status": "DEJA_FORMAT_CIBLE",
                "bbox": None,
                "crop_bbox": (0, 0, original_width, original_height),
                "object_ratio": None,
                "reason": "Image déjà en 968x648",
                "original_width": original_width,
                "original_height": original_height
            }

        # Détection de la graine
        bbox, status, details = detect_seed(image)

        if status == "OK":

            # Calcul de la fenêtre de recadrage
            crop_bbox = calculate_crop_window(
                bbox,
                image.size
            )

            if crop_bbox is not None:

                # Recadrage
                cropped = image.crop(crop_bbox)

                # Redimensionnement final
                final_image = resize_to_target(cropped)

                final_status = "OK"

            else:

                # Impossible de construire une fenêtre
                # contenant toute la graine
                final_image = resize_to_target(image)

                final_status = "REPLI_IMAGE_ENTIERE"

                details["reason"] = (
                    "Fenêtre incompatible avec les dimensions "
                    "de l'image : image entière conservée"
                )

                crop_bbox = None

        else:

            # Détection incertaine ou impossible :
            # ne pas risquer de couper la graine
            final_image = resize_to_target(image)

            final_status = status
            crop_bbox = None

        # Vérification des dimensions
        assert final_image.size == (
            TARGET_WIDTH,
            TARGET_HEIGHT
        )

        save_image(final_image, output_path)

        return {
            "status": final_status,
            "bbox": bbox,
            "crop_bbox": crop_bbox,
            "object_ratio": details.get("object_ratio"),
            "reason": details.get("reason", ""),
            "original_width": original_width,
            "original_height": original_height
        }

    except Exception as error:

        return {
            "status": "ERREUR",
            "bbox": None,
            "crop_bbox": None,
            "object_ratio": None,
            "reason": str(error),
            "original_width": None,
            "original_height": None
        }


# ============================================================
# 7. TRAITEMENT DE TOUS LES DOSSIERS
# ============================================================

def process_all_images():

    results = []

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for species in SPECIES:

        source_folder = DATA_DIR / species
        output_folder = OUTPUT_DIR / species

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        if not source_folder.exists():
            print(f"Dossier absent : {source_folder}")
            continue

        files = sorted([
            p for p in source_folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        print(f"\n{'=' * 50}")
        print(f"ESPECE : {species}")
        print(f"Images : {len(files)}")
        print(f"{'=' * 50}")

        for source_path in files:

            relative_path = source_path.relative_to(source_folder)

            output_path = output_folder / relative_path

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            result = process_image(
                source_path,
                output_path
            )

            result["species"] = species
            result["file"] = str(source_path)
            result["output_file"] = str(output_path)

            results.append(result)

            print(
                f"{source_path.name} : {result['status']}"
            )

    return results


# ============================================================
# 8. RAPPORT CSV
# ============================================================

def save_report(results):

    if not results:
        print("Aucune image traitée.")
        return

    df = pd.DataFrame(results)

    # Coordonnées de la graine
    for i, name in enumerate(
        ["seed_xmin", "seed_ymin", "seed_xmax", "seed_ymax"]
    ):

        df[name] = df["bbox"].apply(
            lambda b: b[i] if isinstance(b, tuple) else None
        )

    # Coordonnées de la fenêtre de recadrage
    for i, name in enumerate(
        ["crop_xmin", "crop_ymin", "crop_xmax", "crop_ymax"]
    ):

        df[name] = df["crop_bbox"].apply(
            lambda b: b[i] if isinstance(b, tuple) else None
        )

    df = df.drop(columns=["bbox", "crop_bbox"])

    df.to_csv(
        CSV_REPORT,
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== BILAN ==========")

    print(f"Total : {len(df)}")

    for status, count in df["status"].value_counts().items():
        print(f"{status} : {count}")

    print(f"\nRapport : {CSV_REPORT}")

    # Vérification des fichiers problématiques
    problematic = df[
        ~df["status"].isin([
            "OK",
            "DEJA_FORMAT_CIBLE"
        ])
    ]

    if len(problematic) > 0:

        print("\nImages à vérifier :")

        for _, row in problematic.iterrows():
            print(
                f"- {row['file']} : {row['reason']}"
            )


# ============================================================
# 9. APERCU VISUEL
# ============================================================

def show_random_examples(results):

    if not PREVIEW_ENABLED:
        return

    valid_results = [
        r for r in results
        if r["status"] in ["OK", "DEJA_FORMAT_CIBLE"]
    ]

    if not valid_results:
        return

    random.seed(RANDOM_SEED)

    selected = random.sample(
        valid_results,
        min(N_PREVIEW, len(valid_results))
    )

    for result in selected:

        original = Image.open(
            result["file"]
        ).convert("RGB")

        processed = Image.open(
            result["output_file"]
        ).convert("RGB")

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 6)
        )

        axes[0].imshow(original)
        axes[0].set_title(
            f"Originale : {original.size}"
        )
        axes[0].axis("off")

        axes[1].imshow(processed)
        axes[1].set_title(
            f"Résultat : {processed.size}"
        )
        axes[1].axis("off")

        fig.suptitle(
            Path(result["file"]).name
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# 10. PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("\nRECADRAGE DES IMAGES DE GRAINES")
    print("Format final : 968 x 648 pixels")

    results = process_all_images()

    save_report(results)

    show_random_examples(results)

    print("\nTraitement terminé.")
