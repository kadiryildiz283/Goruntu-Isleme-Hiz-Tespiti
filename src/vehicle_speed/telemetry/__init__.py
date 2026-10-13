"""Telemetry and ground-truth subpackage for ESP32 hardware synchronization."""

from typing import Final

from vehicle_speed.telemetry.ground_truth import ground_truth_speed, steps_to_metres
from vehicle_speed.telemetry.protocol import (
    CompleteEvent,
    DirectionChangeEvent,
    StartupEvent,
    StatusEvent,
    TelemetryEvent,
    parse_line,
)

__all__: Final[list[str]] = [
    "CompleteEvent",
    "DirectionChangeEvent",
    "StartupEvent",
    "StatusEvent",
    "TelemetryEvent",
    "ground_truth_speed",
    "parse_line",
    "steps_to_metres",
]
