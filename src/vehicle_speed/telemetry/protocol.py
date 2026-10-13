"""ESP32 serial telemetry JSON protocol definition and parser.

Maps firmware string messages into strictly typed Python events.
Tolerates line fragmentation and framing errors gracefully without crashing.
"""

import json
import logging
from dataclasses import dataclass
from typing import Final, Literal

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final[int] = 1
EVENT_STARTUP: Final[str] = "startup"
EVENT_DIRECTION_CHANGE: Final[str] = "direction_change"
EVENT_STATUS: Final[str] = "status"
EVENT_COMPLETE: Final[str] = "complete"
STEPS_PER_MM: Final[int] = 400
TRAVEL_DISTANCE_MM: Final[int] = 500


@dataclass(frozen=True)
class StartupEvent:
    """ESP32 cihaz başlatma olayı (setup tamamlandığında)."""

    event: Literal["startup"]
    message: str
    max_speed: float  # step/s


@dataclass(frozen=True)
class DirectionChangeEvent:
    """Ray uç noktalarında hareket yönü ve hız değişimi olayı."""

    event: Literal["direction_change"]
    new_target_steps: int
    new_max_speed: float
    acceleration: float


@dataclass(frozen=True)
class StatusEvent:
    """Periyodik durum telemetrisi (5000 ms aralıkla)."""

    event: Literal["status"]
    elapsed_time_ms: int
    current_position: int  # step


@dataclass(frozen=True)
class CompleteEvent:
    """Test süresi (300.000 ms) tamamlanma olayı."""

    event: Literal["complete"]
    message: str


TelemetryEvent = StartupEvent | DirectionChangeEvent | StatusEvent | CompleteEvent


def parse_line(line: str) -> TelemetryEvent | None:
    """Tek satır ham JSON telemetri metnini tipli olaya ayrıştırır.

    Bozuk, eksik veya seri porttan yarım gelmiş satırlarda istisna fırlatmaz,
    None döndürür (Fail-Safe framing).

    Args:
        line: Seri porttan okunan ham metin satırı.

    Returns:
        TelemetryEvent | None: Ayrıştırılmış olay veya geçersiz satırda None.
    """
    clean_line = line.strip()
    if not clean_line or not (clean_line.startswith("{") and clean_line.endswith("}")):
        return None

    try:
        data = json.loads(clean_line)
        if not isinstance(data, dict):
            return None

        event_type = data.get("event")
        if event_type == "startup":
            return StartupEvent(
                event="startup",
                message=str(data.get("message", "")),
                max_speed=float(data.get("max_speed", 0.0)),
            )
        if event_type == "direction_change":
            return DirectionChangeEvent(
                event="direction_change",
                new_target_steps=int(data.get("new_target_steps", 0)),
                new_max_speed=float(data.get("new_max_speed", 0.0)),
                acceleration=float(data.get("acceleration", 0.0)),
            )
        if event_type == "status":
            return StatusEvent(
                event="status",
                elapsed_time_ms=int(data.get("elapsed_time_ms", 0)),
                current_position=int(data.get("current_position", 0)),
            )
        if event_type == "complete":
            return CompleteEvent(
                event="complete",
                message=str(data.get("message", "")),
            )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug("Telemetri satırı ayrıştırılamadı: %s (%s)", clean_line, exc)
        return None

    return None
