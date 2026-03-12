"""Calibration package for confidence calibration methods.

This module exposes all of the package's public classes so that users can
import directly from ``calibration`` rather than a submodule.

The real implementation lives in :mod:`calibration.methods`; the code here
is intentionally minimal to avoid duplication.
"""

from __future__ import annotations

from .methods import (
    CalibrationMethod,
    PlattScaling,
    IsotonicCalibration,
    TemperatureScaling,
    EnsembleCalibration,
    CalibrationEvaluator,
)

__all__ = [
    "CalibrationMethod",
    "PlattScaling",
    "IsotonicCalibration",
    "TemperatureScaling",
    "EnsembleCalibration",
    "CalibrationEvaluator",
]

