"""Deep learning object detection and keypoint tracking wrapper.

Abstracts YOLO model invocations, isolates PyTorch imports,
and converts raw tensor outputs into strictly typed, immutable Detection contracts.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

DetectionMode = Literal["ball", "plate", "plate_with_pose"]

# 14 anahtar noktalı araç poz modelinde far indeksleri (modül seviyesi sabitler)
LEFT_HEADLIGHT_IDX: int = 4
RIGHT_HEADLIGHT_IDX: int = 5


@dataclass(frozen=True)
class Detection:
    """Tek bir nesne tespiti veya takibi için dondurulmuş veri sözleşmesi.

    Attributes:
        track_id: YOLO izleme kimliği (izleme yapılmıyorsa veya henüz atanmadıysa None).
        bbox_xyxy: Sınırlayıcı kutu koordinatları (x1, y1, x2, y2).
        confidence: Model tespit güven skoru (0.0 - 1.0).
        pixel_width: Nesnenin piksel genişliği (x2 - x1).
        center_px: Nesne merkez piksel koordinatları (c_x, c_y).
        aspect_ratio: Genişlik / Yükseklik oranı (NRC girdi özniteliği).
        y_normalized: Görüntü tabanına göre normalize y konumu (y2 / H_img).
        class_id: Tespit edilen sınıf kimliği.
    """

    track_id: int | None
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    pixel_width: float
    center_px: tuple[float, float]
    aspect_ratio: float
    y_normalized: float
    class_id: int = 0


def to_bbox(raw: object) -> tuple[int, int, int, int]:
    """YOLO sınır kutusu tensörünü tipli 4'lü tamsayı tuple'ına dönüştürür.

    Any ve numpy tip yayılımını durdurur.

    Args:
        raw: Array-like kutu tensörü veya listesi.

    Returns:
        tuple[int, int, int, int]: (x1, y1, x2, y2)

    Raises:
        ValueError: Girdi 4 elemanlı değilse.
    """
    if not hasattr(raw, "__iter__"):
        msg = f"Kutu koordinatları yinelenebilir olmalıdır, alınan: {type(raw)}"
        raise ValueError(msg)

    items: list[int] = []
    for val in raw:
        items.append(int(float(val)))

    if len(items) != 4:
        msg = f"Kutu koordinatları 4 elemanlı olmalıdır, alınan: {len(items)}"
        raise ValueError(msg)

    return items[0], items[1], items[2], items[3]


class ObjectDetector:
    """YOLO tabanlı nesne tespiti ve takip motoru."""

    def __init__(
        self,
        model_path: Path,
        mode: DetectionMode = "ball",
        target_class_ids: Sequence[int] | None = None,
        conf_threshold: float = 0.40,
    ) -> None:
        """ObjectDetector yapılandırıcısı.

        Args:
            model_path: Model ağırlık dosyası yolu (.pt veya .onnx).
            mode: Tespit modu ('ball', 'plate', veya 'plate_with_pose').
            target_class_ids: İlgilenilen COCO veya özel sınıf kimlikleri.
            conf_threshold: Asgari tespit güven eşiği.
        """
        self.model_path = model_path
        self.mode = mode
        self.target_class_ids = list(target_class_ids) if target_class_ids is not None else None
        self.conf_threshold = conf_threshold
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Modeli ilk ihtiyaç anında güvenle yükler (Lazy Loading)."""
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as err:
                msg = "Ultralytics paketi gerekli: uv pip install ultralytics"
                raise ImportError(msg) from err

            logger.info("YOLO modeli yükleniyor: %s", self.model_path)
            self._model = YOLO(str(self.model_path))
        return self._model

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Kare üzerinde nesne takibi yapar ve tipli tespit listesi döner.

        A1 hatasını Fail-Closed ve güvenli şekilde çözer (UnboundLocalError engellenir).

        Args:
            frame: BGR formatında OpenCV görüntü matrisi.

        Returns:
            list[Detection]: Ayrıştırılmış tespitler listesi.
        """
        model = self._ensure_model()
        img_h, _img_w = frame.shape[:2]

        kwargs: dict[str, Any] = {
            "source": frame,
            "persist": True,
            "conf": self.conf_threshold,
            "verbose": False,
        }
        if self.target_class_ids:
            kwargs["classes"] = self.target_class_ids

        results = model.track(**kwargs)
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        detections: list[Detection] = []

        for i in range(len(boxes)):
            track_id: int | None = None
            try:
                raw_box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = to_bbox(raw_box)

                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy().item())

                conf = float(boxes.conf[i].cpu().numpy().item())
                cls_id = int(boxes.cls[i].cpu().numpy().item())

                pixel_w = float(max(1, x2 - x1))
                pixel_h = float(max(1, y2 - y1))
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                aspect_ratio = pixel_w / pixel_h
                y_norm = y2 / float(img_h) if img_h > 0 else 0.0

                det = Detection(
                    track_id=track_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=conf,
                    pixel_width=pixel_w,
                    center_px=(center_x, center_y),
                    aspect_ratio=aspect_ratio,
                    y_normalized=y_norm,
                    class_id=cls_id,
                )
                detections.append(det)
            except (IndexError, ValueError, AttributeError) as exc:
                logger.warning("Kare nesnesi ayrıştırılamadı (ID=%s): %s", track_id, exc)
                continue

        return detections
