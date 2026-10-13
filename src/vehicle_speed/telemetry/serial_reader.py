"""Robust, configurable serial port reader for ESP32 telemetry streaming.

Replaces legacy listen.py with modern pyserial context management,
configurable CLI options, and safe file handling.
"""

import logging
import sys
from collections.abc import Generator
from pathlib import Path

from vehicle_speed.config import DEFAULT_BAUD_RATE, DEFAULT_SERIAL_PORT, DEFAULT_SERIAL_TIMEOUT_S
from vehicle_speed.telemetry.protocol import TelemetryEvent, parse_line

logger = logging.getLogger(__name__)


def stream_telemetry(
    port: str = DEFAULT_SERIAL_PORT,
    baud_rate: int = DEFAULT_BAUD_RATE,
    timeout_s: float = DEFAULT_SERIAL_TIMEOUT_S,
) -> Generator[TelemetryEvent, None, None]:
    """Seri porttan sürekli telemetri olayları akıtır (Generator).

    Args:
        port: Seri cihaz port yolu (örn: /dev/ttyUSB0).
        baud_rate: Baud hızı (varsayılan: 115200).
        timeout_s: Okuma zaman aşımı saniye.

    Yields:
        TelemetryEvent: Ayrıştırılmış ESP32 telemetri olayı.

    Raises:
        ImportError: pyserial kurulu değilse.
        serial.SerialException: Port açılamazsa veya bağlantı koparsa.
    """
    try:
        import serial
    except ImportError as err:
        msg = "pyserial paketi gerekli: uv pip install pyserial"
        raise ImportError(msg) from err

    logger.info("Seri port açılıyor: %s @ %d baud", port, baud_rate)
    with serial.Serial(port=port, baudrate=baud_rate, timeout=timeout_s) as ser:
        while ser.is_open:
            try:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue
                line_str = line_bytes.decode("utf-8", errors="replace").strip()
                event = parse_line(line_str)
                if event is not None:
                    yield event
            except (UnicodeDecodeError, serial.SerialException) as exc:
                logger.warning("Seri okuma uyarısı: %s", exc)
                continue


def record_telemetry_to_file(
    output_path: Path,
    port: str = DEFAULT_SERIAL_PORT,
    baud_rate: int = DEFAULT_BAUD_RATE,
    max_events: int | None = None,
) -> int:
    """Seri porttan gelen ham telemetri satırlarını dosyaya güvenle kaydeder.

    Args:
        output_path: Kaydedilecek hedef dosya yolu.
        port: Seri cihaz portu.
        baud_rate: Baud hızı.
        max_events: Kaydedilecek azami olay sayısı (None = sonsuz).

    Returns:
        int: Başarıyla kaydedilen olay sayısı.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    logger.info("Telemetri kaydı başladı -> %s", output_path)
    with output_path.open("a", encoding="utf-8") as f:
        for event in stream_telemetry(port=port, baud_rate=baud_rate):
            f.write(f"{event}\n")
            f.flush()
            count += 1
            if max_events is not None and count >= max_events:
                break

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESP32 Telemetri Dinleyici")
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT, help="Seri port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE, help="Baud hızı")
    parser.add_argument("--output", type=Path, default=Path("telemetry.log"), help="Log dosyası")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        record_telemetry_to_file(output_path=args.output, port=args.port, baud_rate=args.baud)
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu.")
        sys.exit(0)
