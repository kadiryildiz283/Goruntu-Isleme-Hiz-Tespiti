"""Unit tests for vehicle_speed.calibration module."""

import math
from dataclasses import FrozenInstanceError

import pytest

from vehicle_speed.calibration import (
    CameraIntrinsics,
    ScaleReference,
    calibrate_dynamic,
    create_intrinsics_from_hfov,
)


def test_camera_intrinsics_is_immutable() -> None:
    """CameraIntrinsics nesnesi dondurulmuştur (frozen), alanları sonradan değiştirilemez."""
    intr = CameraIntrinsics(
        f_px=1000.0,
        optical_center_px=(640.0, 360.0),
        image_width_px=1280,
        image_height_px=720,
        hfov_rad=math.radians(85.0),
        source="datasheet",
    )
    with pytest.raises(FrozenInstanceError):
        intr.f_px = 1200.0  # type: ignore[misc]


def test_scale_reference_is_immutable() -> None:
    """ScaleReference nesnesi dondurulmuştur (frozen)."""
    ref = ScaleReference(
        real_width_m=0.520,
        kind="license_plate",
        min_pixel_width=15.0,
        citation="TR Standart Plaka (520x110 mm)",
    )
    with pytest.raises(FrozenInstanceError):
        ref.real_width_m = 0.60  # type: ignore[misc]


def test_create_intrinsics_from_hfov() -> None:
    """HFOV'dan üretilen iç parametrelerin asal noktası merkezde olmalıdır."""
    hfov_rad = math.radians(90.0)
    intr = create_intrinsics_from_hfov(hfov_rad, width_px=1000, height_px=500)
    assert intr.image_width_px == 1000
    assert intr.image_height_px == 500
    assert intr.optical_center_px == (500.0, 250.0)
    assert intr.source == "datasheet"
    # 90 derece HFOV için f = (1000/2) / tan(45°) = 500 px
    assert intr.f_px == pytest.approx(500.0)


def test_calibrate_dynamic_roundtrip() -> None:
    """Dinamik kalibrasyon nesnesi oluşturulması ve source='dynamic' kontrolü."""
    intr = calibrate_dynamic(
        pixel_width_px=100.0,
        known_distance_m=10.0,
        real_width_m=0.52,
        width_px=1920,
        height_px=1080,
    )
    assert intr.source == "dynamic"
    assert intr.f_px == pytest.approx((100.0 * 10.0) / 0.52)
    assert intr.optical_center_px == (960.0, 540.0)
