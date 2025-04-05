from abc import abstractmethod
from typing import Literal, Optional

import torch

from gragod.predictions.prediction import get_system_scores
from gragod.types import Datasets


class ThresholdCalculator:
    """
    Class for calculating anomaly detection thresholds using different methods.
    """

    def __init__(
        self,
        dataset: Datasets,
        labels: torch.Tensor,
        n_thresholds: int,
        range_based: bool,
        range_metrics_alpha: float,
    ):
        """
        Initialize the ThresholdCalculator with data and parameters.

        Args:
            dataset: The dataset being analyzed
            labels: Tensor containing ground truth labels
            n_thresholds: Number of threshold values to test
            range_based: Whether to use range-based metrics for evaluation
            range_metrics_alpha: Alpha parameter for range-based recall calculation
        """
        self.dataset = dataset
        self.labels = labels
        self.n_thresholds = n_thresholds
        self.range_based = range_based
        self.range_metrics_alpha = range_metrics_alpha

    @abstractmethod
    def calculate_f1_optimized_threshold(self) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_otsu_threshold(self) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_gmm_threshold(self) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_mse_dynamic_threshold(self) -> torch.Tensor:
        pass

    def calculate_threshold(
        self, method: Literal["f1_optimize", "otsu", "gmm", "mse_dynamic"]
    ) -> torch.Tensor:
        if method == "f1_optimize":
            return self.calculate_f1_optimized_threshold()
        elif method == "otsu":
            return self.calculate_otsu_threshold()
        elif method == "gmm":
            return self.calculate_gmm_threshold()
        elif method == "mse_dynamic":
            return self.calculate_mse_dynamic_threshold()
        else:
            raise ValueError(f"Invalid method: {method}")


def get_thresholds(
    dataset: Datasets,
    scores: torch.Tensor,
    labels: torch.Tensor,
    method: Literal["f1_optimize", "otsu", "gmm", "mse_dynamic"],
    n_thresholds: int = 100,
    range_based: bool = True,
    range_metrics_alpha: float = 0.5,
    system_output_mode: Optional[Literal["max", "mean", "sum"]] = None,
) -> torch.Tensor:
    """
    Calculate anomaly detection thresholds using the specified method.

    This is a convenience function that wraps the ThresholdCalculator class.

    Args:
        dataset: The dataset being analyzed
        scores: Tensor containing anomaly scores
        labels: Tensor containing ground truth labels
        method: The threshold calculation method to use:
            - "f1_optimize": Optimize thresholds using F1 score
        n_thresholds: Number of threshold values to test
        range_based: Whether to use range-based metrics for evaluation
        range_metrics_alpha: Alpha parameter for range-based recall calculation
        system_output_mode: Mode for aggregating scores in system-level metrics
        test_score: Test scores tensor, required when method is "spot"

    Returns:
        Calculated thresholds

    Raises:
        ValueError: If an invalid method is specified or required parameters are missing
    """
    from gragod.predictions.per_class_threshold_calculator import (
        PerClassThresholdCalculator,
    )
    from gragod.predictions.system_threshold_calculator import SystemThresholdCalculator

    if labels.ndim in [0, 1] or labels.shape[1] in [0, 1]:
        if system_output_mode is None:
            raise ValueError(
                "system_output_mode must be provided for system-level thresholding"
            )
        system_scores = get_system_scores(scores, system_output_mode)
        calculator = SystemThresholdCalculator(
            dataset=dataset,
            labels=labels,
            n_thresholds=n_thresholds,
            range_based=range_based,
            range_metrics_alpha=range_metrics_alpha,
            system_scores=system_scores,
        )
    else:
        calculator = PerClassThresholdCalculator(
            dataset=dataset,
            scores=scores,
            labels=labels,
            n_thresholds=n_thresholds,
            range_based=range_based,
            range_metrics_alpha=range_metrics_alpha,
        )

    return calculator.calculate_threshold(method=method)
