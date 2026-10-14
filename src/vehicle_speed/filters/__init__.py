"""Spatial trajectory and motion filtering algorithms."""

from typing import Final

from vehicle_speed.filters.kalman import KalmanFilter3D

__all__: Final[list[str]] = ["KalmanFilter3D"]
