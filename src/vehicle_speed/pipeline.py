"""Optical 3D speed estimation pipeline coordinator.

Integrates pinhole depth calculation, 3D spatial projection,
trajectory tracking/Kalman filtering, and Neural Residual Calibration (NRC).
Supports systematic ablation configurations: baseline, kalman, and kalman+nrc.
"""

from typing import Literal

from vehicle_speed.calibration import CameraIntrinsics
from vehicle_speed.detector import Detection
from vehicle_speed.filters.kalman import KalmanFilter3D
from vehicle_speed.geometry import estimate_depth, project_3d
from vehicle_speed.nrc.features import extract_nrc_features
from vehicle_speed.nrc.model import NeuralResidualCalibrator
from vehicle_speed.tracking import SpeedTracker

FilterMode = Literal["none", "kalman"]


class SpeedEstimationPipeline:
    """Tek bir kamera akışında nesnelerin 3B hızını hesaplayan uçtan uca boru hattı."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        real_width_m: float,
        filter_mode: FilterMode = "none",
        nrc_calibrator: NeuralResidualCalibrator | None = None,
        math_interval: int = 5,
    ) -> None:
        """Boru hattı ilklendirici.

        Args:
            intrinsics: Kamera içsel parametreleri.
            real_width_m: Hedef nesnenin fiziksel genişliği (metre).
            filter_mode: 'none' (sonlu farklar) veya 'kalman' (3B Kalman Filtresi).
            nrc_calibrator: Opsiyonel Nöral Artık Kalibratör (NRC) modeli.
            math_interval: Hız türetme kare aralığı.
        """
        self.intrinsics = intrinsics
        self.real_width_m = real_width_m
        self.filter_mode: FilterMode = filter_mode
        self.nrc_calibrator = nrc_calibrator

        self.tracker = SpeedTracker(math_interval=math_interval)
        self.kalman_filters: dict[int, KalmanFilter3D] = {}
        self._last_frames: dict[int, int] = {}

    def process_detection(
        self,
        det: Detection,
        frame_index: int,
        fps: float,
    ) -> float | None:
        """Tek bir tespiti işleyerek anlık hız kestirimi döndürür.

        Args:
            det: Algılanan nesne (Detection).
            frame_index: Anlık video kare indeksi.
            fps: Video kare hızı (FPS).

        Returns:
            float | None: Kestirilen hız (m/s) veya geçerli bir takip yoksa None.
        """
        if det.track_id is None or det.pixel_width <= 0:
            return None

        # 1. Pinhole modeli ile eksenel derinlik kestirimi (Z)
        z_m = estimate_depth(
            f_px=self.intrinsics.f_px,
            real_width_m=self.real_width_m,
            pixel_width_px=det.pixel_width,
        )

        # 2. 3B kamera koordinat düzlemine izdüşüm (X, Y, Z)
        pos_3d = project_3d(
            px_x=det.center_px[0],
            px_y=det.center_px[1],
            z_m=z_m,
            f_px=self.intrinsics.f_px,
            optical_center=self.intrinsics.optical_center_px,
        )

        # 3. Yörünge ve hız kestirimi (Baseline sonlu farklar vs. Kalman)
        speed_raw_ms: float
        if self.filter_mode == "kalman":
            speed_raw_ms = self._update_kalman(
                track_id=det.track_id,
                position_m=pos_3d,
                frame_index=frame_index,
                fps=fps,
            )
        else:
            speed_raw_ms = self.tracker.update(
                track_id=det.track_id,
                position_m=pos_3d,
                frame_index=frame_index,
                fps=fps,
            )

        # 4. İsteğe bağlı Nöral Artık Kalibratör (NRC) düzeltmesi
        if self.nrc_calibrator is not None:
            features = extract_nrc_features(
                detection=det,
                raw_speed_ms=speed_raw_ms,
            )
            return self.nrc_calibrator.calibrate(
                raw_speed_ms=speed_raw_ms,
                features=features,
            )

        return speed_raw_ms

    def _update_kalman(
        self,
        track_id: int,
        position_m: tuple[float, float, float],
        frame_index: int,
        fps: float,
    ) -> float:
        """3B Kalman filtresini günceller ve anlık hızı döndürür."""
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = KalmanFilter3D(initial_position=position_m)
            self._last_frames[track_id] = frame_index
            return 0.0

        kf = self.kalman_filters[track_id]
        last_frame = self._last_frames[track_id]
        frame_diff = frame_index - last_frame

        if frame_diff > 0 and fps > 0:
            dt = frame_diff / fps
            kf.predict(dt=dt)
            kf.update(measurement=position_m)
            self._last_frames[track_id] = frame_index

        return kf.speed_ms

    def prune(self, current_frame_index: int, max_age_frames: int = 30) -> int:
        """Eski izleri hem izleyiciden hem Kalman filtrelerinden temizler."""
        pruned_tracker = self.tracker.prune(
            current_frame_index=current_frame_index,
            max_age_frames=max_age_frames,
        )

        stale_ids = [
            tid
            for tid, last_frame in self._last_frames.items()
            if (current_frame_index - last_frame) > max_age_frames
        ]
        for tid in stale_ids:
            self.kalman_filters.pop(tid, None)
            self._last_frames.pop(tid, None)

        return pruned_tracker

    def clear(self) -> None:
        """Boru hattındaki tüm durumları sıfırlar."""
        self.tracker.clear()
        self.kalman_filters.clear()
        self._last_frames.clear()
