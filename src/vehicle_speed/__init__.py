"""Vehicle Speed Estimation Package.

Monocular camera vehicle speed estimation with ESP32 ground-truth telemetry.
"""

from typing import Final

from vehicle_speed.pipeline import SpeedEstimationPipeline

__version__: Final[str] = "0.1.0"
__all__: Final[list[str]] = ["SpeedEstimationPipeline", "__version__"]
