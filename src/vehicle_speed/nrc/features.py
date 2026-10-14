r"""Feature extraction contract for Neural Residual Calibrator (NRC).

Constructs the 5-dimensional physical-optical input vector:
$\mathbf{x} = [v_{\text{raw}}, w_{\text{px}}, y_{\text{norm}}, AR, conf]^T \in \mathbb{R}^5$
"""

from dataclasses import dataclass

import numpy as np

from vehicle_speed.detector import Detection


@dataclass(frozen=True)
class NRCFeatures:
    """5 boyutlu Nöral Artık Kalibratör girdi öznitelik sözleşmesi.

    Attributes:
        v_raw_ms: Kalman filtresi / izleyiciden gelen ham hız (m/s).
        pixel_width: Nesnenin piksel genişliği ($w_{px} = x_2 - x_1$).
        y_normalized: Nesne alt sınırının görüntü yüksekliğine normalize konumu ($y_2 / H$).
        aspect_ratio: Nesne en/boy oranı ($w_{px} / h_{px}$).
        confidence: YOLO tespit güven skoru ($0.0 - 1.0$).
    """

    v_raw_ms: float
    pixel_width: float
    y_normalized: float
    aspect_ratio: float
    confidence: float

    def to_numpy(self) -> np.ndarray:
        """Öznitelik vektörünü 1D NumPy dizisine dönüştürür."""
        return np.array(
            [
                self.v_raw_ms,
                self.pixel_width,
                self.y_normalized,
                self.aspect_ratio,
                self.confidence,
            ],
            dtype=np.float32,
        )


def extract_nrc_features(
    detection: Detection,
    raw_speed_ms: float,
) -> NRCFeatures:
    """Tek bir tespitten ve anlık ham hızdan NRC öznitelik nesnesi üretir.

    Args:
        detection: Modüler tespit nesnesi.
        raw_speed_ms: İzleyiciden elde edilen ham hız (m/s).

    Returns:
        NRCFeatures: Dondurulmuş 5 boyutlu öznitelik nesnesi.
    """
    return NRCFeatures(
        v_raw_ms=max(0.0, raw_speed_ms),
        pixel_width=max(1.0, detection.pixel_width),
        y_normalized=min(1.0, max(0.0, detection.y_normalized)),
        aspect_ratio=max(0.01, detection.aspect_ratio),
        confidence=min(1.0, max(0.0, detection.confidence)),
    )
