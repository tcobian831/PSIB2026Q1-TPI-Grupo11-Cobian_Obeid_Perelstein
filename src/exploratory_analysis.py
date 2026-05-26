"""
exploratory_analysis.py

Inspeccion visual y cuantitativa inicial del dataset BUSI.

Este modulo NO segmenta automaticamente.
Su objetivo es verificar que:
1. Las imagenes y mascaras se cargan correctamente.
2. La mascara manual coincide con la lesion visible.
3. El preprocesamiento conserva la lesion, los bordes y el patron ecografico.
4. La imagen preprocesada no introduce artefactos fuertes.

Entradas:
- data/raw/Dataset_BUSI_with_GT/
- src/data_loading.py
- src/preprocessing.py

Salidas:
- outputs/figures/exploratory/
- outputs/tables/exploratory_mask_summary.csv
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loading import (
    build_busi_metadata,
    read_grayscale_image,
    compute_log_fourier_spectrum,
)

from preprocessing import (
    preprocess_dataset,
    preprocess_ultrasound_image,
    estimate_speckle_index,
    high_frequency_energy_ratio,
)


LESION_CLASSES = ["benign", "malignant"]


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convierte una mascara manual PNG en mascara booleana.
    """

    return mask > 0


def overlay_contour(
    image: np.ndarray,
    mask: np.ndarray,
    contour_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Superpone el contorno de una mascara sobre una imagen en escala de grises.

    La salida es RGB para que el contorno sea visible.
    """

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


def compute_mask_basic_features(mask: np.ndarray, image_shape: Tuple[int, int]) -> dict:
    """
    Calcula estadisticas basicas de una mascara manual.

    Esto NO es extraccion final de caracteristicas.
    Es solo control de calidad exploratorio.
    """

    h, w = image_shape
    area = int(mask.sum())
    area_fraction = area / (h * w)

    if area == 0:
        return {
            "manual_mask_area_px": 0,
            "manual_mask_area_fraction": 0.0,
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
        "manual_mask_area_px": area,
        "manual_mask_area_fraction": area_fraction,
        "bbox_x_min": x_min,
        "bbox_x_max": x_max,
        "bbox_y_min": y_min,
        "bbox_y_max": y_max,
        "bbox_width": x_max - x_min + 1,
        "bbox_height": y_max - y_min + 1,
    }


def save_exploratory_figure(
    raw_image: np.ndarray,
    preprocessed_image: np.ndarray,
    manual_mask: np.ndarray,
    label: str,
    image_id: str,
    output_path: str | Path,
) -> None:
    """
    Guarda una figura de inspeccion.

    Paneles:
    1. Imagen original.
    2. Imagen preprocesada.
    3. Mascara manual.
    4. Contorno manual sobre original.
    5. Contorno manual sobre preprocesada.
    6. Histogramas original/preprocesada.
    7. Fourier original.
    8. Fourier preprocesada.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_overlay = overlay_contour(raw_image, manual_mask)
    pre_overlay = overlay_contour(preprocessed_image, manual_mask)

    raw_spectrum = compute_log_fourier_spectrum(raw_image)
    pre_spectrum = compute_log_fourier_spectrum(preprocessed_image)

    raw_speckle = estimate_speckle_index(raw_image)
    pre_speckle = estimate_speckle_index(preprocessed_image)

    raw_hf = high_frequency_energy_ratio(raw_image)
    pre_hf = high_frequency_energy_ratio(preprocessed_image)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    axes[0, 0].imshow(raw_image, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(preprocessed_image, cmap="gray")
    axes[0, 1].set_title("Preprocesada")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(manual_mask, cmap="gray")
    axes[0, 2].set_title("Mascara manual")
    axes[0, 2].axis("off")

    axes[0, 3].hist(raw_image.ravel(), bins=64, alpha=0.6, label="Original")
    axes[0, 3].hist(preprocessed_image.ravel(), bins=64, alpha=0.6, label="Preprocesada")
    axes[0, 3].set_title("Histogramas")
    axes[0, 3].set_xlabel("Intensidad")
    axes[0, 3].set_ylabel("Frecuencia")
    axes[0, 3].legend()

    axes[1, 0].imshow(raw_overlay)
    axes[1, 0].set_title("Contorno sobre original")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pre_overlay)
    axes[1, 1].set_title("Contorno sobre preprocesada")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(raw_spectrum, cmap="gray")
    axes[1, 2].set_title("Fourier original")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(pre_spectrum, cmap="gray")
    axes[1, 3].set_title("Fourier preprocesada")
    axes[1, 3].axis("off")

    title = (
        f"{label} - {image_id} | "
        f"speckle: {raw_speckle:.3f} -> {pre_speckle:.3f} | "
        f"HF: {raw_hf:.4f} -> {pre_hf:.4f}"
    )

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_exploratory_subset(
    metadata: pd.DataFrame,
    n_per_class: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Selecciona imagenes benignas y malignas para inspeccion visual.

    No usa imagenes normales porque el foco inmediato del proyecto es
    segmentar masas en benign/malignant.
    """

    lesion_metadata = metadata[
        metadata["label"].isin(LESION_CLASSES)
    ].copy()

    available_per_class = lesion_metadata.groupby("label").size()

    n_safe = min(n_per_class, int(available_per_class.min()))

    subset = (
        lesion_metadata
        .groupby("label", group_keys=False)
        .sample(n=n_safe, random_state=random_state)
        .reset_index(drop=True)
    )

    return subset


def run_exploratory_analysis(
    n_per_class: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Ejecuta la inspeccion exploratoria.

    Devuelve una tabla con estadisticas basicas de las mascaras manuales
    y de la imagen antes/despues del preprocesamiento.
    """

    metadata = build_busi_metadata("data/raw")

    subset = build_exploratory_subset(
        metadata,
        n_per_class=n_per_class,
        random_state=random_state,
    )

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
            Path("outputs/figures/exploratory")
            / label
            / f"{safe_id}_exploratory.png"
        )

        save_exploratory_figure(
            raw_image=raw_image,
            preprocessed_image=preprocessed_image,
            manual_mask=manual_mask,
            label=label,
            image_id=image_id,
            output_path=output_path,
        )

        mask_features = compute_mask_basic_features(
            manual_mask,
            image_shape=raw_image.shape,
        )

        record = {
            "image_id": image_id,
            "label": label,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "height": raw_image.shape[0],
            "width": raw_image.shape[1],
            "speckle_raw": estimate_speckle_index(raw_image),
            "speckle_preprocessed": estimate_speckle_index(preprocessed_image),
            "high_freq_raw": high_frequency_energy_ratio(raw_image),
            "high_freq_preprocessed": high_frequency_energy_ratio(preprocessed_image),
            **mask_features,
        }

        records.append(record)

    summary = pd.DataFrame(records)

    output_table = Path("outputs/tables/exploratory_mask_summary.csv")
    output_table.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_table, index=False)

    return summary


def main() -> None:
    """
    Ejecuta el analisis exploratorio.

    Correr desde la raiz del repo:

        python src/exploratory_analysis.py
    """

    print("\nEjecutando inspeccion exploratoria...")
    summary = run_exploratory_analysis(
        n_per_class=8,
        random_state=42,
    )

    print("\nResumen de mascaras manuales e imagenes inspeccionadas:\n")
    print(summary[[
        "label",
        "image_id",
        "height",
        "width",
        "manual_mask_area_fraction",
        "speckle_raw",
        "speckle_preprocessed",
        "high_freq_raw",
        "high_freq_preprocessed",
    ]])

    print("\nListo.")
    print("Figuras guardadas en: outputs/figures/exploratory/")
    print("Tabla guardada en: outputs/tables/exploratory_mask_summary.csv")


if __name__ == "__main__":
    main()
