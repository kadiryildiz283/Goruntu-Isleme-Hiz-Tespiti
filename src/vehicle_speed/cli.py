"""Command-line interface (CLI) entry point for Vehicle Speed Estimation.

Orchestrates video processing pipeline, optical calculations, tracking,
and optional real-time visualization.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from vehicle_speed.calibration import create_intrinsics_from_hfov
from vehicle_speed.config import (
    DEFAULT_BALL_WIDTH_M,
    DEFAULT_FPS,
    DEFAULT_HFOV_DEG,
    DEFAULT_PLATE_WIDTH_M,
)
from vehicle_speed.detector import ObjectDetector
from vehicle_speed.nrc.model import NeuralResidualCalibrator
from vehicle_speed.pipeline import SpeedEstimationPipeline
from vehicle_speed.visualization import close_all_windows, draw_detection_overlay, show_frame

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Komut satırı argüman ayrıştırıcısını oluşturur."""
    parser = argparse.ArgumentParser(
        prog="vehicle-speed",
        description="Tek kamera ve ESP32 telemetrisi ile Taşıt Hız Tespiti Sistemi",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Girdi video dosyası yolu veya kamera indeksi (varsayılan: 0)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/detector.pt"),
        help="YOLO model ağırlık dosyası yolu",
    )
    parser.add_argument(
        "--mode",
        choices=["ball", "plate"],
        default="ball",
        help="Takip hedefi modu: 'ball' veya 'plate'",
    )
    parser.add_argument(
        "--real-width",
        type=float,
        default=None,
        help="Hedef nesnenin fiziksel genişliği (metre). Belirtilmezse mod varsayılanı alınır.",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=DEFAULT_HFOV_DEG,
        help=f"Kamera yatay görüş açısı (derece, varsayılan: {DEFAULT_HFOV_DEG})",
    )
    parser.add_argument(
        "--filter",
        choices=["none", "kalman"],
        default="none",
        help="Yörünge gürültü filtreleme modu: 'none' veya 'kalman' (varsayılan: none)",
    )
    parser.add_argument(
        "--nrc-weights",
        type=Path,
        default=None,
        help="NRC (Neural Residual Calibrator) model ağırlıkları (.npz dosya yolu)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Grafik ekranı devre dışı bırakır (CI veya sunucu modu)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="İşlenecek azami kare sayısı",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    """Görüntü işleme ve hız tahmini boru hattını çalıştırır.

    Args:
        args: Ayrıştırılmış CLI argümanları.

    Returns:
        int: Çıkış durum kodu (0 = başarı, 1 = hata).
    """
    # Kaynak tespiti (sayı ise kamera indeksi, metin ise video dosyası)
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Video kaynağı açılamadı: %s", args.source)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS

    # Fiziksel genişlik seçimi
    if args.real_width is not None and args.real_width > 0:
        real_width_m = args.real_width
    elif args.mode == "ball":
        real_width_m = DEFAULT_BALL_WIDTH_M
    else:
        real_width_m = DEFAULT_PLATE_WIDTH_M

    import math

    intrinsics = create_intrinsics_from_hfov(
        hfov_rad=math.radians(args.hfov),
        width_px=width,
        height_px=height,
        source="datasheet",
    )

    nrc_calibrator: NeuralResidualCalibrator | None = None
    if args.nrc_weights is not None:
        nrc_calibrator = NeuralResidualCalibrator(weights_path=args.nrc_weights)

    detector = ObjectDetector(model_path=args.weights, mode=args.mode)
    pipeline = SpeedEstimationPipeline(
        intrinsics=intrinsics,
        real_width_m=real_width_m,
        filter_mode=args.filter,
        nrc_calibrator=nrc_calibrator,
    )

    logger.info(
        "Boru hattı başlatıldı: %dx%d @ %.1f FPS | f=%.1f px | Hedef W=%.3f m | Filtre=%s | NRC=%s",
        width,
        height,
        fps,
        intrinsics.f_px,
        real_width_m,
        args.filter,
        bool(nrc_calibrator),
    )

    frame_idx = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            detections = detector.track(frame)
            speeds: dict[int, float] = {}

            for det in detections:
                if det.track_id is None or det.pixel_width <= 0:
                    continue

                try:
                    speed_ms = pipeline.process_detection(
                        det=det,
                        frame_index=frame_idx,
                        fps=fps,
                    )
                    if speed_ms is not None:
                        speeds[det.track_id] = speed_ms
                except ValueError as err:
                    logger.debug("Hesaplama atlandı (ID=%s): %s", det.track_id, err)
                    continue

            # Periyodik bellek temizliği
            if frame_idx % 60 == 0:
                pipeline.prune(current_frame_index=frame_idx)

            if not args.headless:
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else fps
                annotated = draw_detection_overlay(
                    frame=frame,
                    detections=detections,
                    speeds_ms=speeds,
                    fps=current_fps,
                )
                key = show_frame("Vehicle Speed Estimation", annotated)
                if key == ord("q"):
                    logger.info("Kullanıcı tarafından sonlandırıldı.")
                    break
    finally:
        cap.release()
        if not args.headless:
            close_all_windows()

    logger.info("Tamamlandı: Toplam %d kare işlendi.", frame_idx)
    return 0


def main() -> None:
    """CLI ana giriş noktası."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    code = run_pipeline(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
