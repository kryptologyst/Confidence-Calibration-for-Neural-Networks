import os
import sys

# ensure the src directory is on the import path when running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pytest

from calibration import (
    PlattScaling,
    IsotonicCalibration,
    TemperatureScaling,
    EnsembleCalibration,
    CalibrationEvaluator,
)


def simple_binary_data(n: int = 100):
    """Create random logits and binary labels.

    The logits have two columns to mimic a two-class neural network output
    and the labels take values 0 or 1.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(n, 2)
    y = rng.randint(0, 2, size=n)
    return X, y


def test_platt_scaling():
    X, y = simple_binary_data()
    ps = PlattScaling(random_state=0).fit(X, y)
    probs = ps.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)


def test_isotonic_calibration():
    X, y = simple_binary_data()
    # isotonic only consumes a single column of scores
    X1 = X[:, -1]
    iso = IsotonicCalibration().fit(X1, y)
    probs = iso.predict_proba(X1)
    assert probs.shape == (X.shape[0], 2)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)


def test_temperature_scaling():
    X, y = simple_binary_data()
    ts = TemperatureScaling(device="cpu").fit(X, y)
    probs = ts.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)


def test_ensemble_calibration():
    X, y = simple_binary_data()
    # ensemble includes an isotonic calibrator, so pass only one column
    X1 = X[:, -1]
    ens = EnsembleCalibration().fit(X1, y)
    probs = ens.predict_proba(X1)
    assert probs.shape == (X.shape[0], 2)


def test_calibration_evaluator():
    X, y = simple_binary_data()
    ps = PlattScaling(random_state=0).fit(X, y)
    probs = ps.predict_proba(X)[:, 1]
    evaluator = CalibrationEvaluator()
    results = evaluator.evaluate_calibration(y, probs, "Platt")
    assert "brier_score" in results
    assert "ece" in results
    assert "mce" in results
    assert "reliability_data" in results
    # basic sanity checks
    assert isinstance(results["brier_score"], float)
    assert 0 <= results["ece"] <= 1
    assert 0 <= results["mce"] <= 1
