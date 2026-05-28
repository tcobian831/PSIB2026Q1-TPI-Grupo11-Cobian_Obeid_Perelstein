"""
Interfaz para segmentacion semi-automatica de lesiones en ecografias mamarias BUSI.

Version 3 - correcciones Cobi:
1. El usuario carga una ecografia BUSI.
2. El usuario dibuja un ROI rectangular alrededor de la lesion.
3. A partir del ROI se elige automaticamente una semilla interna y se crea la mascara automatica.
4. El usuario puede cargar la mascara manual/ground truth de la base de datos.
5. La interfaz muestra: imagen con ROI y mascara automatica, mascara automatica, mascara manual
   y comparacion visual auto vs manual.
6. Calcula Dice, Jaccard, sensibilidad y precision entre mascara automatica y manual.
7. Permite guardar la mascara automatica.

Dependencias:
    pip install PySide6 opencv-python numpy

Ejecucion:
    python busi_region_growing_gui_v3_cobi.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# Rectangulo ROI en coordenadas de imagen: x0, y0, x1, y1, con x1/y1 incluidos.
RoiRect = Tuple[int, int, int, int]
Point = Tuple[int, int]


class ClickableImageLabel(QLabel):
    """QLabel que muestra una imagen escalada y devuelve coordenadas reales de imagen."""

    image_clicked = Signal(int, int)
    roi_selected = Signal(int, int, int, int)

    def __init__(self, text: str = "Cargue una imagen", minimum_size: Tuple[int, int] = (520, 330)) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(*minimum_size)
        self.setStyleSheet("border: 1px solid #888; background-color: #111; color: #ddd;")

        self._qimage: Optional[QImage] = None
        self._image_width: int = 0
        self._image_height: int = 0
        self._mode: str = "none"
        self._drag_start: Optional[Point] = None

    def set_interaction_mode(self, mode: str) -> None:
        if mode not in {"none", "seed", "roi"}:
            raise ValueError("Modo invalido. Use 'none', 'seed' o 'roi'.")
        self._mode = mode
        self._drag_start = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Recibe imagen RGB uint8 y la muestra manteniendo aspect ratio."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("set_image espera una imagen RGB con shape (H, W, 3).")

        image_rgb = np.ascontiguousarray(image_rgb.astype(np.uint8))
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width

        self._qimage = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        self._image_width = width
        self._image_height = height
        self._update_pixmap()

    def clear_image(self, text: str) -> None:
        self._qimage = None
        self._image_width = 0
        self._image_height = 0
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming convention
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self) -> None:
        if self._qimage is None:
            return

        pixmap = QPixmap.fromImage(self._qimage)
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming convention
        if self._mode == "none" or self.pixmap() is None or self._qimage is None:
            return
        if event.button() != Qt.LeftButton:
            return

        x_label, y_label = self._event_position(event)
        mapped = self._label_to_image_coordinates(x_label, y_label)
        if mapped is None:
            return

        if self._mode == "roi":
            self._drag_start = mapped
            return

        if self._mode == "seed":
            x_img, y_img = mapped
            self.image_clicked.emit(x_img, y_img)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming convention
        if self._mode != "roi" or self._drag_start is None:
            return
        if event.button() != Qt.LeftButton:
            return

        x_label, y_label = self._event_position(event)
        mapped = self._label_to_image_coordinates(x_label, y_label)
        if mapped is None:
            self._drag_start = None
            return

        x_start, y_start = self._drag_start
        x_end, y_end = mapped
        self._drag_start = None

        x0, x1 = sorted((x_start, x_end))
        y0, y1 = sorted((y_start, y_end))

        # Evita rectangulos accidentales demasiado chicos.
        if (x1 - x0) < 10 or (y1 - y0) < 10:
            return

        self.roi_selected.emit(x0, y0, x1, y1)

    @staticmethod
    def _event_position(event: QMouseEvent) -> Point:
        """Compatibilidad simple con Qt6."""
        pos = event.position()
        return int(pos.x()), int(pos.y())

    def _label_to_image_coordinates(self, x_label: int, y_label: int) -> Optional[Point]:
        """Convierte coordenadas del QLabel a coordenadas de la imagen original."""
        pixmap = self.pixmap()
        if pixmap is None or self._image_width <= 0 or self._image_height <= 0:
            return None

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        offset_x = (self.width() - pixmap_width) // 2
        offset_y = (self.height() - pixmap_height) // 2

        x_in_pixmap = x_label - offset_x
        y_in_pixmap = y_label - offset_y

        if not (0 <= x_in_pixmap < pixmap_width and 0 <= y_in_pixmap < pixmap_height):
            return None

        x_img = int(x_in_pixmap * self._image_width / pixmap_width)
        y_img = int(y_in_pixmap * self._image_height / pixmap_height)

        x_img = int(np.clip(x_img, 0, self._image_width - 1))
        y_img = int(np.clip(y_img, 0, self._image_height - 1))

        return x_img, y_img


# -----------------------------------------------------------------------------
# Lectura, preprocesamiento y segmentacion
# -----------------------------------------------------------------------------


def read_image_grayscale(path: str | Path) -> np.ndarray:
    """Lee una imagen como escala de grises uint8."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return image


def read_manual_mask(path: str | Path, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    """
    Lee una mascara manual como binaria booleana.

    Devuelve:
        mask_bool: mascara True/False.
        resized: True si tuvo que redimensionarse para coincidir con la imagen.
    """
    mask_gray = read_image_grayscale(path)
    resized = False

    if mask_gray.shape != target_shape:
        mask_gray = cv2.resize(
            mask_gray,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        resized = True

    # En BUSI las mascaras suelen estar en 0/255. El umbral 127 robustecce JPG/PNG.
    return mask_gray > 127, resized


def preprocess_for_region_growing(image_gray: np.ndarray, median_kernel_size: int = 3) -> np.ndarray:
    """
    Preprocesamiento conservador para ecografia.

    1. Asegura uint8.
    2. Aplica filtro mediano para reducir granularidad/speckle preservando bordes.
    """
    if image_gray.dtype != np.uint8:
        image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # OpenCV exige kernel impar y mayor a 1.
    if median_kernel_size % 2 == 0:
        median_kernel_size += 1
    median_kernel_size = max(3, median_kernel_size)

    return cv2.medianBlur(image_gray, median_kernel_size)


def make_allowed_mask(shape: Tuple[int, int], roi_rect: Optional[RoiRect]) -> np.ndarray:
    """Crea una mascara booleana que limita la zona donde puede crecer la region."""
    height, width = shape
    allowed = np.zeros((height, width), dtype=bool)

    if roi_rect is None:
        allowed[:, :] = True
        return allowed

    x0, y0, x1, y1 = roi_rect
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(y1, 0, height - 1))

    allowed[y0 : y1 + 1, x0 : x1 + 1] = True
    return allowed


def choose_seed_from_roi(image_gray: np.ndarray, roi_rect: RoiRect) -> Point:
    """
    Elige automaticamente una semilla dentro del ROI.

    Para lesiones hipoecoicas se buscan pixeles relativamente oscuros dentro del ROI.
    Para evitar caer en speckle aislado, se toma el componente oscuro mas grande y se usa
    su centroide. Si falla, se usa el centro geometrico del ROI.
    """
    height, width = image_gray.shape
    x0, y0, x1, y1 = roi_rect
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(y1, 0, height - 1))

    crop = image_gray[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return (x0 + x1) // 2, (y0 + y1) // 2

    threshold = float(np.percentile(crop, 30))
    dark = (crop <= threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if num_labels <= 1:
        return (x0 + x1) // 2, (y0 + y1) // 2

    # Ignora etiqueta 0, que es fondo. Elige el componente oscuro de mayor area.
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    selected_label = int(np.argmax(component_areas) + 1)
    cx, cy = centroids[selected_label]

    seed_x = int(np.clip(round(x0 + cx), x0, x1))
    seed_y = int(np.clip(round(y0 + cy), y0, y1))
    return seed_x, seed_y


def region_growing(
    image_gray: np.ndarray,
    seed_xy: Point,
    tolerance: int = 25,
    connectivity: int = 8,
    max_area_fraction: float = 0.90,
    roi_rect: Optional[RoiRect] = None,
    grow_dark_lesion: bool = True,
) -> np.ndarray:
    """
    Segmenta una region por crecimiento de regiones desde una semilla.

    - Puede limitar el crecimiento a un ROI rectangular.
    - En modo hipoecoico, acepta pixeles mas oscuros y pixeles hasta seed+tolerancia.
    - Aplica postprocesamiento y conserva el componente conectado de la semilla.
    """
    if image_gray.ndim != 2:
        raise ValueError("region_growing espera una imagen en escala de grises.")

    height, width = image_gray.shape
    seed_x, seed_y = seed_xy

    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError("La semilla esta fuera de la imagen.")

    allowed = make_allowed_mask(image_gray.shape, roi_rect)
    if not allowed[seed_y, seed_x]:
        raise ValueError("La semilla debe quedar dentro del ROI seleccionado.")

    # Valor robusto de semilla: mediana de una ventana local, menos sensible a speckle.
    r = 2
    y0, y1 = max(0, seed_y - r), min(height, seed_y + r + 1)
    x0, x1 = max(0, seed_x - r), min(width, seed_x + r + 1)
    seed_value = float(np.median(image_gray[y0:y1, x0:x1]))

    upper = min(255.0, seed_value + tolerance)
    lower = max(0.0, seed_value - tolerance)

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

    mask = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    queue: deque[Point] = deque([(seed_x, seed_y)])
    visited[seed_y, seed_x] = True

    allowed_area = int(allowed.sum())
    max_area = max(1, int(max_area_fraction * allowed_area))
    area = 0

    while queue:
        x, y = queue.popleft()
        pixel_value = float(image_gray[y, x])

        if grow_dark_lesion:
            accepted = pixel_value <= upper
        else:
            accepted = lower <= pixel_value <= upper

        if accepted:
            mask[y, x] = True
            area += 1

            if area >= max_area:
                break

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and allowed[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((nx, ny))

    return postprocess_mask(mask, seed_xy=seed_xy, allowed_mask=allowed)


def fill_binary_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Rellena huecos internos de una mascara binaria uint8 0/255."""
    if mask_u8.max() == 0:
        return mask_u8

    flood = mask_u8.copy()
    height, width = flood.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    # El fondo externo queda en blanco; lo no alcanzado por el flood fill son huecos.
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled


def postprocess_mask(
    mask: np.ndarray,
    seed_xy: Optional[Point] = None,
    allowed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Limpieza morfologica, relleno de huecos y seleccion del componente de la semilla."""
    mask_u8 = (mask.astype(np.uint8)) * 255

    if allowed_mask is not None:
        mask_u8[~allowed_mask] = 0

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask_u8 = fill_binary_holes(mask_u8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open, iterations=1)

    if allowed_mask is not None:
        mask_u8[~allowed_mask] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8 > 0

    selected_label: Optional[int] = None
    if seed_xy is not None:
        seed_x, seed_y = seed_xy
        if 0 <= seed_y < labels.shape[0] and 0 <= seed_x < labels.shape[1]:
            label_at_seed = int(labels[seed_y, seed_x])
            if label_at_seed > 0:
                selected_label = label_at_seed

    if selected_label is None:
        areas = stats[1:, cv2.CC_STAT_AREA]
        selected_label = int(np.argmax(areas) + 1)

    clean_mask = labels == selected_label

    if allowed_mask is not None:
        clean_mask &= allowed_mask

    return clean_mask


# -----------------------------------------------------------------------------
# Visualizacion y metricas
# -----------------------------------------------------------------------------


def make_overlay(
    image_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    seed_xy: Optional[Point] = None,
    roi_rect: Optional[RoiRect] = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """Devuelve imagen RGB con ROI, mascara roja semitransparente, contorno y semilla."""
    base_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    output = base_rgb.copy()

    if mask is not None and mask.any():
        overlay = base_rgb.copy()
        overlay[mask] = np.array([255, 0, 0], dtype=np.uint8)
        output = cv2.addWeighted(overlay, alpha, base_rgb, 1 - alpha, 0)

        contours, _ = cv2.findContours(
            (mask.astype(np.uint8)) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    if roi_rect is not None:
        x0, y0, x1, y1 = roi_rect
        cv2.rectangle(output, (x0, y0), (x1, y1), color=(0, 180, 255), thickness=2)

    if seed_xy is not None:
        cv2.drawMarker(
            output,
            seed_xy,
            color=(255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    return output


def mask_to_rgb(mask: Optional[np.ndarray], color: Tuple[int, int, int]) -> np.ndarray:
    """Convierte una mascara booleana en imagen RGB colorida sobre fondo negro."""
    if mask is None:
        return np.zeros((220, 320, 3), dtype=np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = np.array(color, dtype=np.uint8)
    return rgb


def make_comparison_overlay(
    image_gray: np.ndarray,
    auto_mask: Optional[np.ndarray],
    manual_mask: Optional[np.ndarray],
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay comparativo:
    - Verde: verdadero positivo, auto y manual coinciden.
    - Rojo: falso positivo, la mascara automatica agrego de mas.
    - Azul: falso negativo, la mascara automatica se quedo corta.
    """
    base_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    if auto_mask is None or manual_mask is None:
        return base_rgb

    auto = auto_mask.astype(bool)
    manual = manual_mask.astype(bool)

    true_positive = auto & manual
    false_positive = auto & ~manual
    false_negative = ~auto & manual

    overlay = base_rgb.copy()
    overlay[true_positive] = np.array([0, 255, 0], dtype=np.uint8)
    overlay[false_positive] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[false_negative] = np.array([0, 120, 255], dtype=np.uint8)

    output = cv2.addWeighted(overlay, alpha, base_rgb, 1 - alpha, 0)
    return output


def compute_segmentation_metrics(auto_mask: np.ndarray, manual_mask: np.ndarray) -> dict[str, float | int]:
    """Calcula indices de comparacion pixel a pixel entre mascara automatica y manual."""
    auto = auto_mask.astype(bool)
    manual = manual_mask.astype(bool)

    tp = int(np.logical_and(auto, manual).sum())
    fp = int(np.logical_and(auto, ~manual).sum())
    fn = int(np.logical_and(~auto, manual).sum())
    tn = int(np.logical_and(~auto, ~manual).sum())

    def safe_div(num: float, den: float) -> float:
        if den == 0:
            return float("nan")
        return num / den

    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    jaccard = safe_div(tp, tp + fp + fn)
    sensitivity = safe_div(tp, tp + fn)  # Recall / TPR
    precision = safe_div(tp, tp + fp)    # PPV

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Dice": dice,
        "Jaccard": jaccard,
        "Sensibilidad": sensitivity,
        "Precision": precision,
    }


def format_metric(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if np.isnan(value):
        return "N/A"
    return f"{value:.4f}"


class MainWindow(QMainWindow):
    """Ventana principal de la aplicacion."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BUSI - Segmentacion con ROI + comparacion contra mascara manual")
        self.resize(1350, 850)

        self.image_path: Optional[Path] = None
        self.manual_mask_path: Optional[Path] = None
        self.original_gray: Optional[np.ndarray] = None
        self.preprocessed_gray: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.manual_mask: Optional[np.ndarray] = None
        self.current_seed: Optional[Point] = None
        self.roi_rect: Optional[RoiRect] = None

        self.main_image_label = ClickableImageLabel("Imagen + ROI + mascara automatica", (540, 360))
        self.main_image_label.image_clicked.connect(self.on_image_clicked)
        self.main_image_label.roi_selected.connect(self.on_roi_selected)

        self.auto_mask_label = ClickableImageLabel("Mascara automatica", (360, 260))
        self.auto_mask_label.set_interaction_mode("none")

        self.manual_mask_label = ClickableImageLabel("Mascara manual", (360, 260))
        self.manual_mask_label.set_interaction_mode("none")

        self.comparison_label = ClickableImageLabel("Comparacion auto vs manual", (360, 260))
        self.comparison_label.set_interaction_mode("none")

        self.load_button = QPushButton("Cargar imagen BUSI")
        self.load_button.clicked.connect(self.load_image)

        self.roi_button = QPushButton("Dibujar ROI y segmentar")
        self.roi_button.clicked.connect(self.start_roi_selection)
        self.roi_button.setEnabled(False)

        self.update_auto_button = QPushButton("Actualizar mascara")
        self.update_auto_button.clicked.connect(self.segment_from_current_roi)
        self.update_auto_button.setEnabled(False)

        self.load_manual_mask_button = QPushButton("Cargar mascara manual")
        self.load_manual_mask_button.clicked.connect(self.load_manual_mask)
        self.load_manual_mask_button.setEnabled(False)

        self.clear_roi_button = QPushButton("Borrar ROI")
        self.clear_roi_button.clicked.connect(self.clear_roi)
        self.clear_roi_button.setEnabled(False)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_all_except_image)
        self.reset_button.setEnabled(False)

        self.save_mask_button = QPushButton("Guardar mascara auto")
        self.save_mask_button.clicked.connect(self.save_mask)
        self.save_mask_button.setEnabled(False)

        self.tolerance_spinbox = QSpinBox()
        self.tolerance_spinbox.setRange(1, 100)
        self.tolerance_spinbox.setValue(25)
        self.tolerance_spinbox.setSuffix(" niveles")
        self.tolerance_spinbox.setToolTip("Diferencia maxima de intensidad respecto de la semilla automatica.")

        self.median_spinbox = QSpinBox()
        self.median_spinbox.setRange(3, 15)
        self.median_spinbox.setSingleStep(2)
        self.median_spinbox.setValue(3)
        self.median_spinbox.setToolTip("Kernel impar del filtro mediano.")

        self.dark_lesion_checkbox = QCheckBox("Lesion hipoecoica")
        self.dark_lesion_checkbox.setChecked(True)
        self.dark_lesion_checkbox.setToolTip(
            "Activado: incluye pixeles mas oscuros que la semilla y hasta seed+tolerancia. "
            "Desactivado: usa criterio simetrico seed±tolerancia."
        )

        self.metrics_label = QLabel(
            "Indices contra mascara manual:\n"
            "Dice: - | Jaccard: - | Sensibilidad: - | Precision: -\n"
            "TP: - | FP: - | FN: - | TN: -"
        )
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet(
            "font-family: Consolas, monospace; background-color: #222; color: #eee; "
            "border: 1px solid #777; padding: 8px;"
        )

        self.status_label = QLabel(
            "Flujo recomendado: 1) cargar ecografia, 2) dibujar ROI alrededor de la lesion, "
            "3) revisar mascara automatica, 4) cargar mascara manual y comparar indices."
        )
        self.status_label.setWordWrap(True)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.roi_button)
        controls_layout.addWidget(self.update_auto_button)
        controls_layout.addWidget(self.load_manual_mask_button)
        controls_layout.addWidget(self.clear_roi_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.save_mask_button)
        controls_layout.addWidget(QLabel("Tol:"))
        controls_layout.addWidget(self.tolerance_spinbox)
        controls_layout.addWidget(QLabel("Mediana:"))
        controls_layout.addWidget(self.median_spinbox)
        controls_layout.addWidget(self.dark_lesion_checkbox)
        controls_layout.addStretch()

        views_layout = QGridLayout()
        views_layout.addWidget(QLabel("Imagen con ROI + mascara automatica"), 0, 0)
        views_layout.addWidget(QLabel("Mascara automatica"), 0, 1)
        views_layout.addWidget(self.main_image_label, 1, 0, 3, 1)
        views_layout.addWidget(self.auto_mask_label, 1, 1)
        views_layout.addWidget(QLabel("Mascara manual / ground truth"), 2, 1)
        views_layout.addWidget(self.manual_mask_label, 3, 1)
        views_layout.addWidget(QLabel("Comparacion: verde=TP, rojo=FP, azul=FN"), 4, 0)
        views_layout.addWidget(QLabel("Indices de clase"), 4, 1)
        views_layout.addWidget(self.comparison_label, 5, 0)
        views_layout.addWidget(self.metrics_label, 5, 1)
        views_layout.setColumnStretch(0, 2)
        views_layout.setColumnStretch(1, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(views_layout, stretch=1)
        main_layout.addWidget(self.status_label)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Eventos de GUI
    # ------------------------------------------------------------------

    def load_image(self) -> None:
        """Abre dialogo y carga una imagen PNG/JPG."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen BUSI",
            "",
            "Imagenes (*.png *.jpg *.jpeg *.bmp);;Todos los archivos (*)",
        )

        if not file_path:
            return

        path = Path(file_path)

        if "_mask" in path.stem.lower():
            QMessageBox.warning(
                self,
                "Archivo no recomendado",
                "Seleccionaste una mascara manual (_mask). Carga la imagen ecografica original.",
            )
            return

        try:
            self.original_gray = read_image_grayscale(path)
        except Exception as exc:  # noqa: BLE001 - mostrar error en GUI
            QMessageBox.critical(self, "Error al cargar imagen", str(exc))
            return

        self.image_path = path
        self.manual_mask_path = None
        self.current_mask = None
        self.manual_mask = None
        self.current_seed = None
        self.preprocessed_gray = None
        self.roi_rect = None
        self.main_image_label.set_interaction_mode("seed")

        self.roi_button.setEnabled(True)
        self.update_auto_button.setEnabled(False)
        self.load_manual_mask_button.setEnabled(False)
        self.clear_roi_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.save_mask_button.setEnabled(False)

        self.display_all_views()
        self.reset_metrics_label()

        label_hint = self._infer_busi_label(path)
        self.status_label.setText(
            f"Imagen cargada: {path.name} {label_hint}. "
            "Ahora presione 'Dibujar ROI y segmentar', encierre la lesion, y la mascara automatica se calcula sola. "
            "Tambien puede hacer click manual dentro de la lesion si quiere forzar una semilla."
        )

    def start_roi_selection(self) -> None:
        """Activa el modo de arrastre para dibujar ROI."""
        if self.original_gray is None:
            return

        self.current_mask = None
        self.current_seed = None
        self.main_image_label.set_interaction_mode("roi")
        self.save_mask_button.setEnabled(False)
        self.load_manual_mask_button.setEnabled(False)
        self.display_all_views()
        self.reset_metrics_label()

        self.status_label.setText(
            "Modo ROI activo: haga click izquierdo y arrastre un rectangulo alrededor de la lesion. "
            "Al soltar el mouse, se calcula automaticamente la mascara dentro del ROI."
        )

    def on_roi_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Guarda ROI, elige semilla automatica y segmenta."""
        self.roi_rect = (x0, y0, x1, y1)
        self.main_image_label.set_interaction_mode("seed")
        self.clear_roi_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.update_auto_button.setEnabled(True)

        self.segment_from_current_roi()

    def segment_from_current_roi(self) -> None:
        """Segmenta usando el ROI actual y una semilla automatica."""
        if self.original_gray is None:
            return
        if self.roi_rect is None:
            QMessageBox.warning(
                self,
                "Falta ROI",
                "Primero dibuje un ROI alrededor de la lesion.",
            )
            return

        tolerance = int(self.tolerance_spinbox.value())
        median_kernel = int(self.median_spinbox.value())
        grow_dark = bool(self.dark_lesion_checkbox.isChecked())

        self.preprocessed_gray = preprocess_for_region_growing(
            self.original_gray,
            median_kernel_size=median_kernel,
        )
        self.current_seed = choose_seed_from_roi(self.preprocessed_gray, self.roi_rect)

        try:
            self.current_mask = region_growing(
                self.preprocessed_gray,
                seed_xy=self.current_seed,
                tolerance=tolerance,
                connectivity=8,
                max_area_fraction=0.90,
                roi_rect=self.roi_rect,
                grow_dark_lesion=grow_dark,
            )
        except Exception as exc:  # noqa: BLE001 - mostrar error en GUI
            QMessageBox.critical(self, "Error en segmentacion", str(exc))
            return

        self.save_mask_button.setEnabled(True)
        self.load_manual_mask_button.setEnabled(True)
        self.update_auto_button.setEnabled(True)
        self.display_all_views()
        self.update_metrics_if_possible()

        area_px = int(self.current_mask.sum())
        x0, y0, x1, y1 = self.roi_rect
        roi_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        roi_fraction = 100 * area_px / roi_area
        criterion = "hipoecoico" if grow_dark else "simetrico"

        self.status_label.setText(
            f"ROI fijado: x=[{x0},{x1}], y=[{y0},{y1}] | "
            f"Semilla automatica: {self.current_seed} | Tolerancia: {tolerance} | "
            f"Filtro mediano: {median_kernel} | Criterio: {criterion} | "
            f"Area automatica: {area_px} px ({roi_fraction:.1f}% del ROI). "
            "Ahora puede cargar la mascara manual para calcular Dice, Jaccard, sensibilidad y precision."
        )

    def on_image_clicked(self, x: int, y: int) -> None:
        """
        Refinamiento opcional: si el usuario hace click, se usa esa semilla manual.
        No es obligatorio para cumplir el flujo ROI -> mascara automatica.
        """
        if self.original_gray is None:
            return
        if self.roi_rect is None:
            QMessageBox.information(
                self,
                "Primero ROI",
                "Para este flujo se recomienda dibujar primero el ROI. Luego la mascara se calcula sola.",
            )
            return
        if not self._point_inside_roi((x, y), self.roi_rect):
            QMessageBox.warning(
                self,
                "Semilla fuera del ROI",
                "La semilla debe estar dentro del rectangulo ROI.",
            )
            return

        tolerance = int(self.tolerance_spinbox.value())
        median_kernel = int(self.median_spinbox.value())
        grow_dark = bool(self.dark_lesion_checkbox.isChecked())

        self.preprocessed_gray = preprocess_for_region_growing(
            self.original_gray,
            median_kernel_size=median_kernel,
        )
        self.current_seed = (x, y)

        try:
            self.current_mask = region_growing(
                self.preprocessed_gray,
                seed_xy=self.current_seed,
                tolerance=tolerance,
                connectivity=8,
                max_area_fraction=0.90,
                roi_rect=self.roi_rect,
                grow_dark_lesion=grow_dark,
            )
        except Exception as exc:  # noqa: BLE001 - mostrar error en GUI
            QMessageBox.critical(self, "Error en segmentacion", str(exc))
            return

        self.save_mask_button.setEnabled(True)
        self.load_manual_mask_button.setEnabled(True)
        self.display_all_views()
        self.update_metrics_if_possible()
        self.status_label.setText(
            f"Mascara recalculada con semilla manual: (x={x}, y={y}). "
            "Este paso es opcional; el flujo principal ya funciona solo con ROI."
        )

    def load_manual_mask(self) -> None:
        """Carga mascara manual/ground truth y actualiza comparacion."""
        if self.original_gray is None:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar mascara manual / ground truth",
            "",
            "Imagenes (*.png *.jpg *.jpeg *.bmp);;Todos los archivos (*)",
        )

        if not file_path:
            return

        try:
            self.manual_mask, resized = read_manual_mask(file_path, self.original_gray.shape)
        except Exception as exc:  # noqa: BLE001 - mostrar error en GUI
            QMessageBox.critical(self, "Error al cargar mascara manual", str(exc))
            return

        self.manual_mask_path = Path(file_path)
        self.display_all_views()
        self.update_metrics_if_possible()

        resize_text = " Fue redimensionada con vecino mas cercano para coincidir con la ecografia." if resized else ""
        self.status_label.setText(
            f"Mascara manual cargada: {self.manual_mask_path.name}.{resize_text} "
            "Se actualizaron la comparacion visual y los indices de clase."
        )

    def clear_roi(self) -> None:
        """Borra ROI y mascara automatica; conserva la imagen y la mascara manual si existiera."""
        self.roi_rect = None
        self.current_mask = None
        self.current_seed = None
        self.clear_roi_button.setEnabled(False)
        self.update_auto_button.setEnabled(False)
        self.save_mask_button.setEnabled(False)
        self.load_manual_mask_button.setEnabled(False)
        self.main_image_label.set_interaction_mode("seed")
        self.display_all_views()
        self.reset_metrics_label()
        self.status_label.setText(
            "ROI borrado. Dibuje un nuevo ROI para volver a generar la mascara automatica."
        )

    def reset_all_except_image(self) -> None:
        """Borra ROI, mascaras y metricas, pero conserva la imagen cargada."""
        if self.original_gray is None:
            return

        self.roi_rect = None
        self.current_mask = None
        self.manual_mask = None
        self.manual_mask_path = None
        self.current_seed = None
        self.preprocessed_gray = None
        self.main_image_label.set_interaction_mode("seed")

        self.clear_roi_button.setEnabled(False)
        self.update_auto_button.setEnabled(False)
        self.load_manual_mask_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.save_mask_button.setEnabled(False)

        self.display_all_views()
        self.reset_metrics_label()
        self.status_label.setText(
            "Reset realizado. La imagen sigue cargada. Dibuje un ROI para generar una nueva mascara."
        )

    def save_mask(self) -> None:
        """Guarda la mascara automatica actual como PNG binaria."""
        if self.current_mask is None:
            return

        default_name = "region_growing_mask.png"
        if self.image_path is not None:
            default_name = f"{self.image_path.stem}_auto_mask.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar mascara automatica",
            default_name,
            "PNG (*.png);;Todos los archivos (*)",
        )

        if not file_path:
            return

        mask_u8 = (self.current_mask.astype(np.uint8)) * 255
        ok = cv2.imwrite(str(file_path), mask_u8)
        if not ok:
            QMessageBox.critical(self, "Error", "No se pudo guardar la mascara.")
            return

        self.status_label.setText(f"Mascara automatica guardada en: {file_path}")

    # ------------------------------------------------------------------
    # Actualizacion visual y metricas
    # ------------------------------------------------------------------

    def display_all_views(self) -> None:
        """Actualiza los cuatro paneles principales."""
        if self.original_gray is None:
            self.main_image_label.clear_image("Imagen + ROI + mascara automatica")
            self.auto_mask_label.clear_image("Mascara automatica")
            self.manual_mask_label.clear_image("Mascara manual")
            self.comparison_label.clear_image("Comparacion auto vs manual")
            return

        main_rgb = make_overlay(
            self.original_gray,
            mask=self.current_mask,
            seed_xy=self.current_seed,
            roi_rect=self.roi_rect,
            alpha=0.35,
        )
        self.main_image_label.set_image(main_rgb)

        if self.current_mask is not None:
            self.auto_mask_label.set_image(mask_to_rgb(self.current_mask, color=(255, 0, 0)))
        else:
            empty = np.zeros((*self.original_gray.shape, 3), dtype=np.uint8)
            self.auto_mask_label.set_image(empty)

        if self.manual_mask is not None:
            self.manual_mask_label.set_image(mask_to_rgb(self.manual_mask, color=(0, 255, 255)))
        else:
            empty = np.zeros((*self.original_gray.shape, 3), dtype=np.uint8)
            self.manual_mask_label.set_image(empty)

        comparison_rgb = make_comparison_overlay(self.original_gray, self.current_mask, self.manual_mask)
        self.comparison_label.set_image(comparison_rgb)

    def update_metrics_if_possible(self) -> None:
        """Calcula metricas si hay mascara automatica y manual."""
        if self.current_mask is None or self.manual_mask is None:
            self.reset_metrics_label()
            return

        metrics = compute_segmentation_metrics(self.current_mask, self.manual_mask)
        self.metrics_label.setText(
            "Indices contra mascara manual\n"
            f"Dice         : {format_metric(metrics['Dice'])}\n"
            f"Jaccard      : {format_metric(metrics['Jaccard'])}\n"
            f"Sensibilidad : {format_metric(metrics['Sensibilidad'])}\n"
            f"Precision    : {format_metric(metrics['Precision'])}\n"
            "\nMatriz pixel a pixel\n"
            f"TP: {metrics['TP']} | FP: {metrics['FP']}\n"
            f"FN: {metrics['FN']} | TN: {metrics['TN']}"
        )

    def reset_metrics_label(self) -> None:
        self.metrics_label.setText(
            "Indices contra mascara manual\n"
            "Dice         : -\n"
            "Jaccard      : -\n"
            "Sensibilidad : -\n"
            "Precision    : -\n"
            "\nMatriz pixel a pixel\n"
            "TP: - | FP: -\n"
            "FN: - | TN: -"
        )

    @staticmethod
    def _point_inside_roi(point: Point, roi_rect: RoiRect) -> bool:
        x, y = point
        x0, y0, x1, y1 = roi_rect
        return x0 <= x <= x1 and y0 <= y <= y1

    @staticmethod
    def _infer_busi_label(path: Path) -> str:
        """Intenta identificar si la imagen viene de benign o malignant segun la carpeta."""
        parts = [p.lower() for p in path.parts]
        if "benign" in parts:
            return "[benign]"
        if "malignant" in parts:
            return "[malignant]"
        if "normal" in parts:
            return "[normal: no recomendada para esta interfaz]"
        return ""


def main() -> None:
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
