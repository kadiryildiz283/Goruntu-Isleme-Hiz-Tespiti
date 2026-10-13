"""Neural Residual Calibrator (NRC) module."""

from typing import Final

from vehicle_speed.nrc.features import NRCFeatures, extract_nrc_features
from vehicle_speed.nrc.model import NeuralResidualCalibrator

__all__: Final[list[str]] = [
    "NRCFeatures",
    "NeuralResidualCalibrator",
    "extract_nrc_features",
]
