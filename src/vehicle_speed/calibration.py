"""Camera calibration representations and physical scale references.

Defines immutable data contracts for camera intrinsics and tracking targets.
Maintains clear provenance (datasheet vs dynamic) for research reproducibility.
"""

import math
from dataclasses import dataclass
from typing import Literal

from vehicle_speed.geometry import focal_length_from_hfov, focal_length_from_reference

IntrinsicsSource = Literal["datasheet", "dynamic"]
ReferenceKind = Literal["license_plate", "headlight_pair", "sports_ball"]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Kamera iç yöneltme parametreleri (Intrinsics).

    Attributes:
        f_px: Piksel cinsinden odak uzaklığı.
        optical_center_px: Asal nokta / optik merkez (c_x, c_y).
        image_width_px: Görüntü genişliği (piksel).
        image_height_px: Görüntü yüksekliği (piksel).
        hfov_rad: Yatay görüş açısı (radyan).
        source: Değerin kaynağı ('datasheet' veya 'dynamic').
    """

    f_px: float
    optical_center_px: tuple[float, float]
    image_width_px: int
    image_height_px: int
    hfov_rad: float
    source: IntrinsicsSource


@dataclass(frozen=True)
class ScaleReference:
    """Metrik ölçek referans sözleşmesi.

    Attributes:
        real_width_m: Referans nesnenin fiziksel genişliği (metre).
        kind: Referans nesne türü.
        min_pixel_width: Güvenilir ölçüm yapılabilecek asgari piksel genişliği.
        citation: Ölçünün dayandığı bilimsel veya endüstri standardı atfı.
    """

    real_width_m: float
    kind: ReferenceKind
    min_pixel_width: float
    citation: str


def create_intrinsics_from_hfov(
    hfov_rad: float,
    width_px: int,
    height_px: int,
    source: IntrinsicsSource = "datasheet",
) -> CameraIntrinsics:
    """Yatay görüş açısı (HFOV) ve çözünürlükten iç parametreleri oluşturur.

    Args:
        hfov_rad: Radyan cinsinden yatay görüş açısı.
        width_px: Görüntü genişliği (piksel).
        height_px: Görüntü yüksekliği (piksel).
        source: Kalibrasyon kaynağı.

    Returns:
        CameraIntrinsics: Dondurulmuş kamera iç parametre nesnesi.
    """
    f_px = focal_length_from_hfov(hfov_rad, width_px)
    c_x = width_px / 2.0
    c_y = height_px / 2.0

    return CameraIntrinsics(
        f_px=f_px,
        optical_center_px=(c_x, c_y),
        image_width_px=width_px,
        image_height_px=height_px,
        hfov_rad=hfov_rad,
        source=source,
    )


def calibrate_dynamic(
    pixel_width_px: float,
    known_distance_m: float,
    real_width_m: float,
    width_px: int,
    height_px: int,
) -> CameraIntrinsics:
    """Görüntülenen bilinen referans nesneden çalışma zamanı kalibrasyonu üretir.

    Args:
        pixel_width_px: Referans nesnenin piksel genişliği.
        known_distance_m: Referans nesnenin kameraya mesafesi (metre).
        real_width_m: Referans nesnenin fiziksel genişliği (metre).
        width_px: Görüntü genişliği (piksel).
        height_px: Görüntü yüksekliği (piksel).

    Returns:
        CameraIntrinsics: 'dynamic' kaynaklı kamera iç parametreleri.
    """
    f_px = focal_length_from_reference(pixel_width_px, known_distance_m, real_width_m)
    # Ters HFOV hesabı: HFOV = 2 * atan((W / 2) / f)
    hfov_rad = 2.0 * math.atan((width_px / 2.0) / f_px)
    c_x = width_px / 2.0
    c_y = height_px / 2.0

    return CameraIntrinsics(
        f_px=f_px,
        optical_center_px=(c_x, c_y),
        image_width_px=width_px,
        image_height_px=height_px,
        hfov_rad=hfov_rad,
        source="dynamic",
    )
