"""Core calibration methods for neural network confidence calibration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, TensorDataset

from utils.device import get_device
from utils.seeding import set_seed


class CalibrationMethod(ABC):
    """Abstract base class for calibration methods."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibrationMethod":
        """Fit the calibration method to the data."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the calibration method."""
        pass


class PlattScaling(CalibrationMethod):
    """Platt scaling (logistic calibration) for binary classification."""

    def __init__(self, random_state: Optional[int] = None) -> None:
        """Initialize Platt scaling calibrator.
        
        Args:
            random_state: Random state for reproducibility.
        """
        self.random_state = random_state
        self.calibrator: Optional[LogisticRegression] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlattScaling":
        """Fit Platt scaling calibrator.
        
        Args:
            X: Input features (logits or probabilities).
            y: True binary labels.
            
        Returns:
            Self for method chaining.
        """
        self.calibrator = LogisticRegression(
            random_state=self.random_state, max_iter=1000
        )
        self.calibrator.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities.
        
        Args:
            X: Input features (logits or probabilities).
            
        Returns:
            Calibrated probabilities.
        """
        if not self.is_fitted or self.calibrator is None:
            raise ValueError("Calibrator must be fitted before prediction.")
        
        return self.calibrator.predict_proba(X)

    def get_name(self) -> str:
        """Get the name of the calibration method."""
        return "Platt Scaling"


class IsotonicCalibration(CalibrationMethod):
    """Isotonic regression calibration for binary classification."""

    def __init__(self) -> None:
        """Initialize isotonic calibration."""
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IsotonicCalibration":
        """Fit isotonic regression calibrator.
        
        Args:
            X: Input features (logits or probabilities).
            y: True binary labels.
            
        Returns:
            Self for method chaining.
        """
        # For isotonic regression we work with a single-dimensional score.
        # If the input has multiple columns we assume the last column contains
        # the probability/logit for the positive class.  The previous version
        # flattened the entire array which accidentally doubled the sample count
        # when more than one feature was passed.
        if X.ndim > 1:
            if X.shape[1] == 1:
                X = X.ravel()
            else:
                # take the last column by convention
                X = X[:, -1]
        
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities.
        
        Args:
            X: Input features (logits or probabilities).
            
        Returns:
            Calibrated probabilities.
        """
        if not self.is_fitted or self.calibrator is None:
            raise ValueError("Calibrator must be fitted before prediction.")
        
        # For isotonic regression, we need 1D input
        if X.ndim > 1:
            X = X.flatten()
        
        calibrated_probs = self.calibrator.transform(X)
        # Convert to 2D array for consistency
        return np.column_stack([1 - calibrated_probs, calibrated_probs])

    def get_name(self) -> str:
        """Get the name of the calibration method."""
        return "Isotonic Regression"


class TemperatureScaling(CalibrationMethod):
    """Temperature scaling calibration for neural networks."""

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize temperature scaling calibrator.
        
        Args:
            device: Device to use for computation.
        """
        self.device = device or get_device()
        self.temperature: Optional[torch.Tensor] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemperatureScaling":
        """Fit temperature scaling calibrator.
        
        Args:
            X: Input logits from neural network.
            y: True binary labels.
            
        Returns:
            Self for method chaining.
        """
        # Convert to tensors
        logits = torch.tensor(X, dtype=torch.float32, device=self.device)
        labels = torch.tensor(y, dtype=torch.long, device=self.device)
        
        # Initialize temperature parameter
        self.temperature = nn.Parameter(torch.ones(1, device=self.device))
        
        # Optimize temperature using NLL loss
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities.
        
        Args:
            X: Input logits from neural network.
            
        Returns:
            Calibrated probabilities.
        """
        if not self.is_fitted or self.temperature is None:
            raise ValueError("Calibrator must be fitted before prediction.")
        
        logits = torch.tensor(X, dtype=torch.float32, device=self.device)
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        # detach to avoid the "requires_grad" error when converting to numpy
        return probs.detach().cpu().numpy()

    def get_name(self) -> str:
        """Get the name of the calibration method."""
        return "Temperature Scaling"


class EnsembleCalibration(CalibrationMethod):
    """Ensemble-based calibration using multiple methods."""

    def __init__(
        self,
        methods: Optional[List[CalibrationMethod]] = None,
        weights: Optional[List[float]] = None,
    ) -> None:
        """Initialize ensemble calibration.
        
        Args:
            methods: List of calibration methods to ensemble.
            weights: Weights for each method (if None, equal weights).
        """
        self.methods = methods or [
            PlattScaling(),
            IsotonicCalibration(),
            TemperatureScaling(),
        ]
        self.weights = weights or [1.0 / len(self.methods)] * len(self.methods)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleCalibration":
        """Fit all calibration methods.
        
        Args:
            X: Input features (logits or probabilities).
            y: True binary labels.
            
        Returns:
            Self for method chaining.
        """
        for method in self.methods:
            # many underlying calibrators expect 2D inputs; reshape if we were
            # given a flat vector (e.g. from the isotonic pre‑processing in
            # tests).
            X_in = X
            if X_in.ndim == 1:
                X_in = X_in.reshape(-1, 1)
            method.fit(X_in, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble calibrated probabilities.
        
        Args:
            X: Input features (logits or probabilities).
            
        Returns:
            Ensemble calibrated probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction.")
        
        predictions = []
        for method in self.methods:
            X_in = X
            if X_in.ndim == 1:
                X_in = X_in.reshape(-1, 1)
            pred = method.predict_proba(X_in)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred

    def get_name(self) -> str:
        """Get the name of the calibration method."""
        return "Ensemble Calibration"


class CalibrationEvaluator:
    """Evaluator for calibration methods."""

    def __init__(self) -> None:
        """Initialize calibration evaluator."""
        pass

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        method_name: str,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Evaluate calibration performance.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.
            method_name: Name of the calibration method.
            n_bins: Number of bins for calibration curve.
            
        Returns:
            Dictionary of calibration metrics.
        """
        # Brier score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y_true, y_prob, n_bins)
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(y_true, y_prob, n_bins)
        
        # Reliability diagram data
        reliability_data = self._compute_reliability_diagram(y_true, y_prob, n_bins)
        
        return {
            "method": method_name,
            "brier_score": brier_score,
            "ece": ece,
            "mce": mce,
            "reliability_data": reliability_data,
        }

    def _compute_ece(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

    def _compute_mce(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce

    def _compute_reliability_diagram(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """Compute reliability diagram data."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(y_true[in_bin].mean())
                bin_confidences.append(y_prob[in_bin].mean())
                bin_counts.append(in_bin.sum())
        
        return {
            "bin_centers": np.array(bin_centers),
            "bin_accuracies": np.array(bin_accuracies),
            "bin_confidences": np.array(bin_confidences),
            "bin_counts": np.array(bin_counts),
        }
