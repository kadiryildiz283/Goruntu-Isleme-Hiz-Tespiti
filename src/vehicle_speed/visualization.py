"""OpenCV visualization and graphical user interface layer.

ISOLATION CONTRACT:
This is the ONLY module in the entire codebase permitted to invoke
OpenCV GUI functions (cv2.imshow, cv2.waitKey, cv2.destroyAllWindows).
All business logic, geometric math, and tracking MUST remain headless.
"""

import contextlib
import logging
from collections.abc import Sequence

import cv2
import numpy as np

from vehicle_speed.detector import Detection

logger = logging.getLogger(__name__)


def draw_detection_overlay(
    frame: np.ndarray,
    detections: Sequence[Detection],
    speeds_ms: dict[int, float],
    fps: float | None = None,
) -> np.ndarray:
    """Tespit edilen nesneleri, kimlikleri ve hesaplanan hızları kareye çizer.

    Args:
        frame: BGR formatındaki OpenCV görüntü matrisi.
        detections: Mevcut karedeki tespitler listesi.
        speeds_ms: İz kimliğine karşılık gelen hız sözlüğü {track_id: m/s}.
        fps: Ekrana yazdırılacak anlık FPS değeri (opsiyonel).

    Returns:
        np.ndarray: Görselleştirme katmanları eklenmiş kopya kare matrisi.
    """
    canvas = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        track_id = det.track_id

        # Sınır kutusu
        color = (0, 255, 0) if track_id is not None else (0, 165, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Hız etiketi (m/s -> km/h dönüşümü arayüzde yapılır)
        speed_kmh_str = "Hiz: -- km/h"
        if track_id is not None and track_id in speeds_ms:
            v_ms = speeds_ms[track_id]
            v_kmh = v_ms * 3.6
            speed_kmh_str = f"Hiz: {v_kmh:.1f} km/h ({v_ms:.1f} m/s)"

        label = f"ID: {track_id} | {speed_kmh_str}" if track_id is not None else speed_kmh_str

        # Etiket arka planı ve yazısı
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # FPS göstergesi
    if fps is not None and fps > 0:
        cv2.putText(
            canvas,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def show_frame(window_name: str, frame: np.ndarray, wait_ms: int = 1) -> int:
    """Kareyi ekranda gösterir ve tuş basımını bekler.

    Headless ortamlarda cv2.error fırlatmadan güvenle -1 döner.

    Args:
        window_name: Pencere başlığı.
        frame: Gösterilecek görüntü.
        wait_ms: Bekleme süresi (ms).

    Returns:
        int: Basılan tuşun ASCII kodu veya headless ortamda -1.
    """
    try:
        cv2.imshow(window_name, frame)
        return int(cv2.waitKey(wait_ms) & 0xFF)
    except cv2.error as exc:
        logger.debug("Headless ortam: cv2.imshow açılamadı (%s)", exc)
        return -1


def close_all_windows() -> None:
    """Tüm açık OpenCV pencerelerini kapatır."""
    with contextlib.suppress(cv2.error):
        cv2.destroyAllWindows()
