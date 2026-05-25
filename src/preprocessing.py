"""
Bloque 2 - Preprocesamiento de ecografias mamarias BUSI.

Este modulo implementa el segundo bloque del pipeline:
1. Recibe el dataset organizado por data_loading.py.
2. Lee cada imagen ecografica cruda.
3. Evalua indicadores simples asociados a ruido speckle.
4. Aplica reduccion de ruido y realce local de contraste.
5. Guarda el dataset preprocesado.
6. Genera visualizaciones comparativas antes/despues.

No realiza segmentacion.
No extrae caracteristicas de la lesion.
No modifica las mascaras manuales.

Justificacion tecnica:
- La ecografia presenta ruido speckle, de naturaleza multiplicativa, que aparece como textura granular.
- Para reducirlo sin destruir totalmente bordes, se usa una combinacion conservadora:
  normalizacion robusta + filtro mediano + filtro bilateral + CLAHE.
- El filtro mediano reduce impulsos y granularidad local.
- El filtro bilateral suaviza regiones manteniendo bordes.
- CLAHE mejora contraste local, util en ecografias de bajo contraste.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loading import (
    VALID_CLASSES,
    build_busi_metadata,
    read_grayscale_image,
    compute_log_fourier_spectrum,
)


def robust_normalize_uint8(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """
    Normaliza intensidades usando percentiles.

    Esto evita que pixeles extremos dominen la escala de intensidades.

    Parametros
    ----------
    image:
        Imagen en escala de grises.
    lower_percentile:
        Percentil inferior usado como minimo robusto.
    upper_percentile:
        Percentil superior usado como maximo robusto.

    Retorna
    -------
    np.ndarray
        Imagen normalizada en uint8, rango [0, 255].
    """

    image_float = image.astype(np.float32)

    p_low = np.percentile(image_float, lower_percentile)
    p_high = np.percentile(image_float, upper_percentile)

    if p_high <= p_low:
        return image.astype(np.uint8)

    normalized = (image_float - p_low) / (p_high - p_low)
    normalized = np.clip(normalized, 0, 1)

    normalized_uint8 = (255 * normalized).astype(np.uint8)

    return normalized_uint8


def estimate_speckle_index(
    image: np.ndarray,
    window_size: int = 15,
    eps: float = 1e-6,
) -> float:
    """
    Estima un indice simple de granularidad compatible con speckle.

    El speckle en ecografia se manifiesta como variacion local de intensidad.
    Una forma practica de cuantificarlo es calcular el coeficiente de variacion local:

        CV_local = sigma_local / (mu_local + eps)

    Luego se toma la mediana de CV_local en toda la imagen.

    Este indice no es un diagnostico fisico perfecto del ruido, pero sirve para
    comparar si el preprocesamiento redujo la granularidad local.

    Parametros
    ----------
    image:
        Imagen en escala de grises.
    window_size:
        Tamano de ventana local.
    eps:
        Constante pequena para evitar division por cero.

    Retorna
    -------
    float
        Mediana del coeficiente de variacion local.
    """

    image_float = image.astype(np.float32)

    local_mean = cv2.blur(image_float, (window_size, window_size))
    local_mean_sq = cv2.blur(image_float ** 2, (window_size, window_size))

    local_variance = local_mean_sq - local_mean ** 2
    local_variance = np.maximum(local_variance, 0)

    local_std = np.sqrt(local_variance)
    local_cv = local_std / (local_mean + eps)

    # Se excluyen zonas casi negras para evitar cocientes artificialmente altos.
    valid_mask = local_mean > 10

    if np.sum(valid_mask) == 0:
        return float(np.median(local_cv))

    return float(np.median(local_cv[valid_mask]))


def high_frequency_energy_ratio(image: np.ndarray, radius_fraction: float = 0.25) -> float:
    """
    Calcula la proporcion de energia en altas frecuencias.

    El ruido granular tiende a aumentar el contenido de alta frecuencia.
    Este indicador sirve para comparar imagen cruda vs imagen preprocesada.

    Parametros
    ----------
    image:
        Imagen en escala de grises.
    radius_fraction:
        Fraccion del radio maximo usada para separar bajas y altas frecuencias.

    Retorna
    -------
    float
        Energia relativa en altas frecuencias.
    """

    image_float = image.astype(np.float32)

    fourier = np.fft.fft2(image_float)
    fourier_shifted = np.fft.fftshift(fourier)

    magnitude_sq = np.abs(fourier_shifted) ** 2

    height, width = image.shape
    cy, cx = height // 2, width // 2

    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    max_radius = np.sqrt(cy ** 2 + cx ** 2)
    cutoff_radius = radius_fraction * max_radius

    high_freq_mask = distance > cutoff_radius

    total_energy = np.sum(magnitude_sq)

    if total_energy == 0:
        return 0.0

    high_energy = np.sum(magnitude_sq[high_freq_mask])

    return float(high_energy / total_energy)

def preprocess_ultrasound_image(
    image: np.ndarray,
    median_kernel_size: int = 3,
    bilateral_diameter: int = 7,
    bilateral_sigma_color: float = 25,
    bilateral_sigma_space: float = 25,
) -> np.ndarray:
    """
    Preprocesa una ecografia mamaria de forma conservadora.

    Pipeline aplicado:
    1. Normalizacion robusta de intensidades.
    2. Filtro mediano suave.
    3. Filtro bilateral para reducir ruido preservando bordes.

    En esta version NO se aplica CLAHE al resultado principal porque,
    en las pruebas visuales, aumento artificialmente la granularidad local
    y elevo el indice de speckle. Esto puede sesgar la segmentacion y el
    analisis posterior de textura, borde e interfase lesion-tejido.

    Retorna
    -------
    np.ndarray
        Imagen preprocesada en uint8.
    """

    image_norm = robust_normalize_uint8(image)

    image_median = cv2.medianBlur(image_norm, median_kernel_size)

    image_bilateral = cv2.bilateralFilter(
        image_median,
        d=bilateral_diameter,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    return image_bilateral

def preprocess_dataset(
    metadata: pd.DataFrame,
    output_dir: str | Path = "data/processed/preprocessed",
) -> pd.DataFrame:
    """
    Preprocesa todas las imagenes listadas en metadata.

    Parametros
    ----------
    metadata:
        Tabla generada por build_busi_metadata().
    output_dir:
        Carpeta donde se guardaran las imagenes preprocesadas.

    Retorna
    -------
    pd.DataFrame
        Metadata original ampliada con:
        - preprocessed_image_path
        - speckle_index_raw
        - speckle_index_preprocessed
        - high_freq_ratio_raw
        - high_freq_ratio_preprocessed
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_records = []

    for _, row in metadata.iterrows():
        image = read_grayscale_image(row["image_path"])
        image_preprocessed = preprocess_ultrasound_image(image)

        label = row["label"]
        image_id = row["image_id"]

        class_output_dir = output_dir / label
        class_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = class_output_dir / f"{image_id}_preprocessed.png"

        success = cv2.imwrite(str(output_path), image_preprocessed)

        if not success:
            raise IOError(f"No se pudo guardar la imagen preprocesada: {output_path}")

        speckle_raw = estimate_speckle_index(image)
        speckle_pre = estimate_speckle_index(image_preprocessed)

        hf_raw = high_frequency_energy_ratio(image)
        hf_pre = high_frequency_energy_ratio(image_preprocessed)

        record = row.to_dict()
        record["preprocessed_image_path"] = str(output_path)
        record["speckle_index_raw"] = speckle_raw
        record["speckle_index_preprocessed"] = speckle_pre
        record["high_freq_ratio_raw"] = hf_raw
        record["high_freq_ratio_preprocessed"] = hf_pre

        processed_records.append(record)

    metadata_preprocessed = pd.DataFrame(processed_records)

    return metadata_preprocessed


def summarize_preprocessing(metadata_preprocessed: pd.DataFrame) -> pd.DataFrame:
    """
    Resume indicadores antes/despues por clase.

    Permite verificar si el preprocesamiento redujo la granularidad local
    y el contenido relativo de alta frecuencia.
    """

    summary = (
        metadata_preprocessed
        .groupby("label")
        .agg(
            n_images=("image_id", "count"),
            mean_speckle_raw=("speckle_index_raw", "mean"),
            mean_speckle_preprocessed=("speckle_index_preprocessed", "mean"),
            mean_high_freq_raw=("high_freq_ratio_raw", "mean"),
            mean_high_freq_preprocessed=("high_freq_ratio_preprocessed", "mean"),
        )
        .reset_index()
    )

    summary["speckle_reduction_percent"] = (
        100
        * (
            summary["mean_speckle_raw"]
            - summary["mean_speckle_preprocessed"]
        )
        / summary["mean_speckle_raw"]
    )

    summary["high_freq_reduction_percent"] = (
        100
        * (
            summary["mean_high_freq_raw"]
            - summary["mean_high_freq_preprocessed"]
        )
        / summary["mean_high_freq_raw"]
    )

    return summary


def plot_preprocessing_comparison_for_image(
    raw_image: np.ndarray,
    preprocessed_image: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Guarda una figura comparativa antes/despues.

    Para cada imagen muestra:
    1. Imagen cruda.
    2. Imagen preprocesada.
    3. Histograma crudo vs preprocesado.
    4. Fourier crudo.
    5. Fourier preprocesado.

    La figura se guarda en disco. No usa plt.show(), para no bloquear la terminal.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_spectrum = compute_log_fourier_spectrum(raw_image)
    pre_spectrum = compute_log_fourier_spectrum(preprocessed_image)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(preprocessed_image, cmap="gray")
    axes[1].set_title("Preprocesada")
    axes[1].axis("off")

    axes[2].hist(raw_image.ravel(), bins=64, alpha=0.6, label="Original")
    axes[2].hist(preprocessed_image.ravel(), bins=64, alpha=0.6, label="Preprocesada")
    axes[2].set_title("Histogramas")
    axes[2].set_xlabel("Intensidad")
    axes[2].set_ylabel("Frecuencia")
    axes[2].legend()

    axes[3].imshow(raw_spectrum, cmap="gray")
    axes[3].set_title("Fourier original")
    axes[3].axis("off")

    axes[4].imshow(pre_spectrum, cmap="gray")
    axes[4].set_title("Fourier preprocesada")
    axes[4].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_preprocessing_examples(
    metadata_preprocessed: pd.DataFrame,
    output_dir: str | Path = "outputs/figures/preprocessing",
    n_examples_per_class: int = 3,
    random_state: int = 42,
) -> None:
    """
    Guarda ejemplos comparativos de cada clase.

    Se generan figuras para normal, benign y malignant.
    Esto documenta visualmente el efecto del preprocesamiento.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in VALID_CLASSES:
        subset = metadata_preprocessed[
            metadata_preprocessed["label"] == label
        ].copy()

        if subset.empty:
            continue

        n_examples = min(n_examples_per_class, len(subset))
        subset = subset.sample(n=n_examples, random_state=random_state)

        for _, row in subset.iterrows():
            raw_image = read_grayscale_image(row["image_path"])
            pre_image = read_grayscale_image(row["preprocessed_image_path"])

            safe_image_id = (
                row["image_id"]
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )

            output_path = output_dir / label / f"{safe_image_id}_comparison.png"

            title = (
                f"{label} - {row['image_id']} | "
                f"Speckle raw={row['speckle_index_raw']:.3f}, "
                f"pre={row['speckle_index_preprocessed']:.3f}"
            )

            plot_preprocessing_comparison_for_image(
                raw_image=raw_image,
                preprocessed_image=pre_image,
                title=title,
                output_path=output_path,
            )


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda un DataFrame como CSV.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


def main() -> None:
    """
    Ejecuta el bloque completo de preprocesamiento.

    Debe ejecutarse desde la raiz del repo:

        python src/preprocessing.py
    """

    data_root = Path("data/raw")

    print("\nCargando metadata del dataset BUSI...")
    metadata = build_busi_metadata(data_root)

    print("\nPreprocesando dataset...")
    metadata_preprocessed = preprocess_dataset(
        metadata,
        output_dir="data/processed/preprocessed",
    )

    print("\nCalculando resumen del preprocesamiento...")
    preprocessing_summary = summarize_preprocessing(metadata_preprocessed)

    print("\nResumen del preprocesamiento por clase:\n")
    print(preprocessing_summary)

    save_dataframe(
        metadata_preprocessed,
        "outputs/tables/metadata_busi_preprocessed.csv",
    )

    save_dataframe(
        preprocessing_summary,
        "outputs/tables/preprocessing_summary.csv",
    )

    print("\nGuardando figuras comparativas antes/despues...")
    save_preprocessing_examples(
        metadata_preprocessed,
        output_dir="outputs/figures/preprocessing",
        n_examples_per_class=3,
        random_state=42,
    )

    print("\nPreprocesamiento finalizado.")
    print("Imagenes preprocesadas guardadas en: data/processed/preprocessed/")
    print("Metadata guardada en: outputs/tables/metadata_busi_preprocessed.csv")
    print("Resumen guardado en: outputs/tables/preprocessing_summary.csv")
    print("Figuras guardadas en: outputs/figures/preprocessing/")


if __name__ == "__main__":
    main()