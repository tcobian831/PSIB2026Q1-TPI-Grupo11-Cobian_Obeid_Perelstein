"""
inspect_dev_subset.py

Inspeccion visual del subset fijo de desarrollo.

Entrada:
- outputs/tables/dev_subset.csv

Salida:
- outputs/figures/dev_subset/
- outputs/tables/dev_subset_inspection_summary.csv

No segmenta automaticamente.
Solo muestra imagen original, imagen preprocesada, mascara manual y contornos.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loading import read_grayscale_image
from preprocessing import (
    preprocess_ultrasound_image,
    estimate_speckle_index,
    high_frequency_energy_ratio,
)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convierte una mascara PNG en mascara booleana."""
    return mask > 0


def overlay_contour(
    image: np.ndarray,
    mask: np.ndarray,
    contour_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Superpone el contorno de la mascara sobre la imagen."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    contours, _ = cv2.findContours(
        (mask.astype(np.uint8)) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        image_rgb,
        contours,
        contourIdx=-1,
        color=contour_color,
        thickness=2,
    )

    return image_rgb


def compute_mask_summary(mask: np.ndarray, image_shape: Tuple[int, int]) -> dict:
    """Calcula area y bounding box de la mascara manual."""
    h, w = image_shape
    area_px = int(mask.sum())
    area_fraction = area_px / (h * w)

    if area_px == 0:
        return {
            "mask_area_px": 0,
            "mask_area_fraction": 0.0,
            "bbox_x_min": np.nan,
            "bbox_x_max": np.nan,
            "bbox_y_min": np.nan,
            "bbox_y_max": np.nan,
            "bbox_width": np.nan,
            "bbox_height": np.nan,
        }

    ys, xs = np.where(mask)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return {
        "mask_area_px": area_px,
        "mask_area_fraction": area_fraction,
        "bbox_x_min": x_min,
        "bbox_x_max": x_max,
        "bbox_y_min": y_min,
        "bbox_y_max": y_max,
        "bbox_width": x_max - x_min + 1,
        "bbox_height": y_max - y_min + 1,
    }


def save_dev_subset_figure(
    raw_image: np.ndarray,
    preprocessed_image: np.ndarray,
    manual_mask: np.ndarray,
    label: str,
    image_id: str,
    output_path: Path,
) -> None:
    """Guarda una figura compacta para inspeccion del subset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_overlay = overlay_contour(raw_image, manual_mask)
    pre_overlay = overlay_contour(preprocessed_image, manual_mask)

    raw_speckle = estimate_speckle_index(raw_image)
    pre_speckle = estimate_speckle_index(preprocessed_image)

    raw_hf = high_frequency_energy_ratio(raw_image)
    pre_hf = high_frequency_energy_ratio(preprocessed_image)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(preprocessed_image, cmap="gray")
    axes[1].set_title("Preprocesada")
    axes[1].axis("off")

    axes[2].imshow(manual_mask, cmap="gray")
    axes[2].set_title("Mascara manual")
    axes[2].axis("off")

    axes[3].imshow(raw_overlay)
    axes[3].set_title("Contorno original")
    axes[3].axis("off")

    axes[4].imshow(pre_overlay)
    axes[4].set_title("Contorno preprocesada")
    axes[4].axis("off")

    title = (
        f"{label} - {image_id} | "
        f"speckle {raw_speckle:.3f}->{pre_speckle:.3f} | "
        f"HF {raw_hf:.4f}->{pre_hf:.4f}"
    )

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    subset_path = Path("outputs/tables/dev_subset.csv")

    if not subset_path.exists():
        raise FileNotFoundError(
            "No existe outputs/tables/dev_subset.csv. "
            "Primero ejecuta: python src/create_dev_subset.py"
        )

    subset = pd.read_csv(subset_path)

    records = []

    for _, row in subset.iterrows():
        raw_image = read_grayscale_image(row["image_path"])
        preprocessed_image = preprocess_ultrasound_image(raw_image)

        manual_mask_raw = read_grayscale_image(row["mask_path"])
        manual_mask = binarize_mask(manual_mask_raw)

        image_id = row["image_id"]
        label = row["label"]

        safe_id = (
            image_id
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        output_path = (
            Path("outputs/figures/dev_subset")
            / label
            / f"{safe_id}_dev_subset.png"
        )

        save_dev_subset_figure(
            raw_image=raw_image,
            preprocessed_image=preprocessed_image,
            manual_mask=manual_mask,
            label=label,
            image_id=image_id,
            output_path=output_path,
        )

        record = {
            "label": label,
            "image_id": image_id,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "height": raw_image.shape[0],
            "width": raw_image.shape[1],
            "speckle_raw": estimate_speckle_index(raw_image),
            "speckle_preprocessed": estimate_speckle_index(preprocessed_image),
            "high_freq_raw": high_frequency_energy_ratio(raw_image),
            "high_freq_preprocessed": high_frequency_energy_ratio(preprocessed_image),
            **compute_mask_summary(manual_mask, raw_image.shape),
        }

        records.append(record)

    summary = pd.DataFrame(records)

    output_summary = Path("outputs/tables/dev_subset_inspection_summary.csv")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary, index=False)

    print("\nInspeccion del subset fijo terminada.")
    print("Figuras guardadas en: outputs/figures/dev_subset/")
    print("Tabla guardada en: outputs/tables/dev_subset_inspection_summary.csv")
    print("\nResumen:")
    print(summary[[
        "label",
        "image_id",
        "mask_area_fraction",
        "speckle_raw",
        "speckle_preprocessed",
        "high_freq_raw",
        "high_freq_preprocessed",
    ]])


if __name__ == "__main__":
    main()
