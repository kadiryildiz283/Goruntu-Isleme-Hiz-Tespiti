"""Unit tests for vehicle_speed.pipeline module."""

import math

import pytest

from vehicle_speed.calibration import CameraIntrinsics, create_intrinsics_from_hfov
from vehicle_speed.config import DEFAULT_PLATE_WIDTH_M
from vehicle_speed.detector import Detection
from vehicle_speed.nrc.model import NeuralResidualCalibrator
from vehicle_speed.pipeline import SpeedEstimationPipeline


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    """Test için varsayılan kamera içsel parametreleri."""
    return create_intrinsics_from_hfov(
        hfov_rad=math.radians(85.0),
        width_px=1280,
        height_px=720,
    )


def make_detection(
    track_id: int | None,
    center_x: float = 640.0,
    center_y: float = 360.0,
    width_px: float = 100.0,
    height_px: float = 50.0,
    confidence: float = 0.9,
    class_id: int = 0,
) -> Detection:
    """Testler için tip-güvenli Detection fabrikası."""
    x1 = int(center_x - width_px / 2.0)
    y1 = int(center_y - height_px / 2.0)
    x2 = int(center_x + width_px / 2.0)
    y2 = int(center_y + height_px / 2.0)
    return Detection(
        track_id=track_id,
        bbox_xyxy=(x1, y1, x2, y2),
        confidence=confidence,
        pixel_width=width_px,
        center_px=(center_x, center_y),
        aspect_ratio=width_px / height_px if height_px > 0 else 1.0,
        y_normalized=y2 / 720.0,
        class_id=class_id,
    )


def test_pipeline_baseline_mode(intrinsics: CameraIntrinsics) -> None:
    """Baseline modunda pinhole ve SpeedTracker ile hız hesaplanır."""
    pipeline = SpeedEstimationPipeline(
        intrinsics=intrinsics,
        real_width_m=DEFAULT_PLATE_WIDTH_M,
        filter_mode="none",
        math_interval=1,
    )

    # 1. kare: nesne uzakta (30 metre)
    w_px1 = (pipeline.intrinsics.f_px * DEFAULT_PLATE_WIDTH_M) / 30.0
    det1 = make_detection(track_id=1, width_px=w_px1)
    v1 = pipeline.process_detection(det1, frame_index=0, fps=30.0)
    assert v1 == pytest.approx(0.0)

    # 2. kare: nesne yaklaşıyor (29.5 metre, 1 karede 0.5m = 15 m/s)
    w_px2 = (pipeline.intrinsics.f_px * DEFAULT_PLATE_WIDTH_M) / 29.5
    det2 = make_detection(track_id=1, width_px=w_px2)
    v2 = pipeline.process_detection(det2, frame_index=1, fps=30.0)
    assert v2 is not None
    assert v2 == pytest.approx(15.0, rel=1e-2)


def test_pipeline_kalman_mode(intrinsics: CameraIntrinsics) -> None:
    """Kalman modunda 3B Kalman filtresi ile hız filtrelenir."""
    pipeline = SpeedEstimationPipeline(
        intrinsics=intrinsics,
        real_width_m=DEFAULT_PLATE_WIDTH_M,
        filter_mode="kalman",
    )

    # İlk tespitte başlatma yapılır, hız 0.0 döner
    w_px = (pipeline.intrinsics.f_px * DEFAULT_PLATE_WIDTH_M) / 20.0
    det = make_detection(track_id=10, width_px=w_px)
    v_init = pipeline.process_detection(det, frame_index=0, fps=30.0)
    assert v_init == 0.0
    assert 10 in pipeline.kalman_filters

    # 10 kare boyunca sabit 10 m/s hızla yaklaşma
    v_last: float | None = None
    for i in range(1, 10):
        z_curr = 20.0 - (10.0 * (i / 30.0))
        w = (pipeline.intrinsics.f_px * DEFAULT_PLATE_WIDTH_M) / z_curr
        det_i = make_detection(track_id=10, width_px=w)
        v_last = pipeline.process_detection(det_i, frame_index=i, fps=30.0)

    assert v_last is not None
    assert v_last > 0.0


def test_pipeline_kalman_plus_nrc_mode(intrinsics: CameraIntrinsics) -> None:
    """NRC kalibratörü entegre edildiğinde artık kalibrasyon uygulanır."""
    nrc = NeuralResidualCalibrator()
    pipeline = SpeedEstimationPipeline(
        intrinsics=intrinsics,
        real_width_m=DEFAULT_PLATE_WIDTH_M,
        filter_mode="kalman",
        nrc_calibrator=nrc,
    )

    det = make_detection(track_id=5, width_px=80.0)
    v = pipeline.process_detection(det, frame_index=0, fps=30.0)
    assert v is not None
    # Negatif hız üretilemez
    assert v >= 0.0


def test_pipeline_invalid_detections_return_none(intrinsics: CameraIntrinsics) -> None:
    """track_id olmayan veya negatif genişlikli tespitler None dönmelidir."""
    pipeline = SpeedEstimationPipeline(
        intrinsics=intrinsics,
        real_width_m=DEFAULT_PLATE_WIDTH_M,
    )
    # track_id None
    det_no_track = make_detection(track_id=None)
    assert pipeline.process_detection(det_no_track, 0, 30.0) is None

    # non-positive width
    det_zero_w = make_detection(track_id=1, width_px=0.0)
    assert pipeline.process_detection(det_zero_w, 0, 30.0) is None


def test_pipeline_prune_and_clear(intrinsics: CameraIntrinsics) -> None:
    """Boru hattı budama ve sıfırlama mekanizması."""
    pipeline = SpeedEstimationPipeline(
        intrinsics=intrinsics,
        real_width_m=DEFAULT_PLATE_WIDTH_M,
        filter_mode="kalman",
    )
    det = make_detection(track_id=99)
    pipeline.process_detection(det, frame_index=0, fps=30.0)
    assert 99 in pipeline.kalman_filters

    # 50 kare sonra prune çağrıldığında 30 kare eşiğiyle silinir
    pipeline.prune(current_frame_index=50, max_age_frames=30)
    assert 99 not in pipeline.kalman_filters

    # clear her şeyi boşaltır
    pipeline.clear()
    assert len(pipeline.kalman_filters) == 0
    assert pipeline.tracker.active_track_count == 0
