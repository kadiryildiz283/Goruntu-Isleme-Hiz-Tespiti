"""Unit tests for vehicle_speed.cli module."""

from pathlib import Path

from vehicle_speed.cli import build_parser
from vehicle_speed.config import DEFAULT_HFOV_DEG


def test_build_parser_defaults() -> None:
    """CLI varsayılan argümanları doğru yapılandırılmalıdır."""
    parser = build_parser()
    args = parser.parse_args([])

    assert args.source == "0"
    assert args.weights == Path("models/detector.pt")
    assert args.mode == "ball"
    assert args.real_width is None
    assert args.hfov == DEFAULT_HFOV_DEG
    assert args.headless is False
    assert args.max_frames is None


def test_build_parser_custom_args() -> None:
    """CLI özel argümanları doğru ayrıştırılmalıdır."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source",
            "video.mp4",
            "--mode",
            "plate",
            "--real-width",
            "0.52",
            "--hfov",
            "78.5",
            "--headless",
            "--max-frames",
            "100",
        ]
    )
    assert args.source == "video.mp4"
    assert args.mode == "plate"
    assert args.real_width == 0.52
    assert args.hfov == 78.5
    assert args.headless is True
    assert args.max_frames == 100
