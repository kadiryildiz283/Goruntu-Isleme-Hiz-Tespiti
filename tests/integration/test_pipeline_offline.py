"""Integration test running an offline pipeline on synthetic detection data."""

import math

import pytest

from vehicle_speed.calibration import create_intrinsics_from_hfov
from vehicle_speed.config import DEFAULT_PLATE_WIDTH_M
from vehicle_speed.geometry import estimate_depth, project_3d
from vehicle_speed.tracking import SpeedTracker


def test_offline_synthetic_pipeline_run() -> None:
    """Sentetik nesne hareketiyle tüm geometri ve hız boru hattını doğrular."""
    intrinsics = create_intrinsics_from_hfov(
        hfov_rad=math.radians(85.0),
        width_px=1280,
        height_px=720,
    )
    tracker = SpeedTracker()
    fps = 30.0
    real_width_m = DEFAULT_PLATE_WIDTH_M  # 0.520 m

    # Simülasyon: Araç kameraya doğru 15 m/s sabit hızla yaklaşıyor
    # Z(t) = 30.0 - 15.0 * t
    true_speed_ms = 15.0
    computed_speeds: list[float] = []

    for frame_idx in range(15):
        t = frame_idx / fps
        z_true = 30.0 - (true_speed_ms * t)
        if z_true <= 0:
            break

        # Pinhole tersine çevrilerek o mesafedeki piksel genişliği üretilir:
        # w_px = (f * W) / Z
        pixel_width = (intrinsics.f_px * real_width_m) / z_true
        center_x = 640.0
        center_y = 360.0

        # Pipeline adımları:
        # 1. Pinhole derinlik kestirimi
        z_est = estimate_depth(intrinsics.f_px, real_width_m, pixel_width)
        # 2. 3B izdüşüm
        pos_3d = project_3d(
            center_x,
            center_y,
            z_est,
            intrinsics.f_px,
            intrinsics.optical_center_px,
        )
        # 3. Hız güncellemesi
        v_est = tracker.update(
            track_id=42,
            position_m=pos_3d,
            frame_index=frame_idx,
            fps=fps,
        )
        computed_speeds.append(v_est)

    # İlk kareden sonraki tüm karelerde hız 15 m/s'ye eşit olmalıdır
    for v in computed_speeds[1:]:
        assert v == pytest.approx(true_speed_ms, rel=1e-5)
