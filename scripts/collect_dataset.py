"""Synchronized dataset generation tool for Neural Residual Calibrator (NRC).

Matches video frames with ESP32 stepper telemetry to build training datasets:
X: [v_raw, w_px, y_norm, AR, conf] -> y: Delta v = (v_gt - v_raw)
"""

import argparse
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from vehicle_speed.calibration import create_intrinsics_from_hfov
from vehicle_speed.config import (
    DEFAULT_BALL_WIDTH_M,
    DEFAULT_FPS,
    DEFAULT_HFOV_DEG,
    DEFAULT_PLATE_WIDTH_M,
)
from vehicle_speed.detector import DetectionMode, ObjectDetector
from vehicle_speed.pipeline import FilterMode, SpeedEstimationPipeline
from vehicle_speed.telemetry.ground_truth import steps_to_metres
from vehicle_speed.telemetry.protocol import StatusEvent, parse_line

logger = logging.getLogger(__name__)


def load_telemetry_status_events(telemetry_path: Path) -> list[StatusEvent]:
    """Telemetri dosyasından zaman sıralı StatusEvent listesini yükler."""
    events: list[StatusEvent] = []
    with telemetry_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            ev = parse_line(clean)
            if isinstance(ev, StatusEvent):
                events.append(ev)

    events.sort(key=lambda x: x.elapsed_time_ms)
    return events


def interpolate_ground_truth_speed(
    target_time_ms: float,
    status_events: Sequence[StatusEvent],
) -> float | None:
    """Belirli bir milisaniyedeki yer gerçeği hızını türetir."""
    if len(status_events) < 2:
        return None

    if target_time_ms < status_events[0].elapsed_time_ms:
        return None
    if target_time_ms > status_events[-1].elapsed_time_ms:
        return None

    for i in range(1, len(status_events)):
        prev = status_events[i - 1]
        curr = status_events[i]
        if prev.elapsed_time_ms <= target_time_ms <= curr.elapsed_time_ms:
            dt_s = (curr.elapsed_time_ms - prev.elapsed_time_ms) / 1000.0
            if dt_s <= 0.0:
                return None
            p_curr = steps_to_metres(curr.current_position)
            p_prev = steps_to_metres(prev.current_position)
            dp_m = abs(p_curr - p_prev)
            return dp_m / dt_s

    return None


def build_parser() -> argparse.ArgumentParser:
    """Komut satırı argüman ayrıştırıcısını oluşturur."""
    parser = argparse.ArgumentParser(
        prog="collect-dataset",
        description="Video ve ESP32 telemetrisini eşleştirerek NRC eğitim veri seti üretir.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Girdi video dosyası yolu (.mp4)",
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        required=True,
        help="ESP32 telemetri günlüğü yolu (.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/nrc_dataset.npz"),
        help="Çıktı veri seti yolu (.npz formatında)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/detector.pt"),
        help="YOLO model ağırlıkları yolu",
    )
    parser.add_argument(
        "--mode",
        choices=["ball", "plate"],
        default="ball",
        help="Hedef nesne modu",
    )
    parser.add_argument(
        "--filter",
        choices=["none", "kalman"],
        default="none",
        help="Optik boru hattı filtre modu",
    )
    parser.add_argument(
        "--time-offset-ms",
        type=float,
        default=0.0,
        help="Telemetri ile video arasındaki başlangıç zaman farkı (ms)",
    )
    return parser


def process_dataset(
    video_path: Path,
    telemetry_path: Path,
    output_path: Path,
    weights_path: Path,
    mode: DetectionMode = "ball",
    filter_mode: FilterMode = "none",
    time_offset_ms: float = 0.0,
) -> int:
    """Senkronize veri setini derler ve kaydeder."""
    status_events = load_telemetry_status_events(telemetry_path)
    if len(status_events) < 2:
        logger.error("Telemetri dosyasında yeterli status olayı bulunamadı: %s", telemetry_path)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Video dosyası açılamadı: %s", video_path)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    real_width_m = DEFAULT_BALL_WIDTH_M if mode == "ball" else DEFAULT_PLATE_WIDTH_M

    intrinsics = create_intrinsics_from_hfov(
        hfov_rad=math.radians(DEFAULT_HFOV_DEG),
        width_px=width,
        height_px=height,
    )
    detector = ObjectDetector(model_path=weights_path, mode=mode)
    pipeline = SpeedEstimationPipeline(
        intrinsics=intrinsics,
        real_width_m=real_width_m,
        filter_mode=filter_mode,
    )

    features_list: list[list[float]] = []
    targets_list: list[float] = []
    v_gt_list: list[float] = []
    timestamps_list: list[float] = []

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            frame_time_ms = ((frame_idx / fps) * 1000.0) + time_offset_ms
            v_gt = interpolate_ground_truth_speed(frame_time_ms, status_events)
            if v_gt is None:
                continue

            detections = detector.track(frame)
            for det in detections:
                if det.track_id is None or det.pixel_width <= 0:
                    continue

                speed_raw = pipeline.process_detection(
                    det=det,
                    frame_index=frame_idx,
                    fps=fps,
                )
                if speed_raw is None:
                    continue

                delta_v = v_gt - speed_raw
                features_list.append(
                    [
                        speed_raw,
                        det.pixel_width,
                        det.y_normalized,
                        det.aspect_ratio,
                        det.confidence,
                    ]
                )
                targets_list.append(delta_v)
                v_gt_list.append(v_gt)
                timestamps_list.append(frame_time_ms)
    finally:
        cap.release()

    if not features_list:
        logger.warning("Hiçbir senkronize veri eşleşmedi. Zaman ofsetini kontrol edin.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_arr: np.ndarray = np.array(features_list, dtype=np.float32)
    y_arr: np.ndarray = np.reshape(np.array(targets_list, dtype=np.float32), (-1, 1))
    v_gt_arr: np.ndarray = np.reshape(np.array(v_gt_list, dtype=np.float32), (-1, 1))
    ts_arr: np.ndarray = np.array(timestamps_list, dtype=np.float64)

    np.savez(
        output_path,
        X=x_arr,
        y=y_arr,
        v_gt=v_gt_arr,
        timestamps=ts_arr,
    )
    logger.info(
        "Veri seti kaydedildi: %d örnek -> %s",
        len(features_list),
        output_path,
    )
    return 0


def main() -> None:
    """CLI giriş noktası."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    code = process_dataset(
        video_path=args.video,
        telemetry_path=args.telemetry,
        output_path=args.output,
        weights_path=args.weights,
        mode=args.mode,
        filter_mode=args.filter,
        time_offset_ms=args.time_offset_ms,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
