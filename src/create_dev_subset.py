"""
Crea un subset fijo de desarrollo para pruebas de segmentacion.

No segmenta imagenes.
Solo selecciona imagenes benignas y malignas y guarda sus rutas en:

outputs/tables/dev_subset.csv
"""

from pathlib import Path

import pandas as pd

from data_loading import build_busi_metadata


LESION_CLASSES = ["benign", "malignant"]


def create_dev_subset(
    n_per_class: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Selecciona un subset fijo de imagenes benignas y malignas.

    Este subset se usara para comparar metodos de segmentacion.
    """

    metadata = build_busi_metadata("data/raw")

    lesion_metadata = metadata[
        metadata["label"].isin(LESION_CLASSES)
    ].copy()

    subset = (
        lesion_metadata
        .groupby("label", group_keys=False)
        .sample(n=n_per_class, random_state=random_state)
        .reset_index(drop=True)
    )

    return subset


def main() -> None:
    subset = create_dev_subset(
        n_per_class=10,
        random_state=42,
    )

    output_path = Path("outputs/tables/dev_subset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset.to_csv(output_path, index=False)

    print("\nSubset de desarrollo creado:\n")
    print(subset[["label", "image_id", "image_path", "mask_path"]])

    print(f"\nGuardado en: {output_path}")


if __name__ == "__main__":
    main()