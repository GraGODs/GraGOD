from typing import Literal, Optional, cast

import torch

from gragod.metrics.per_class_calculator import PerClassCalculator
from gragod.metrics.system_calculator import SystemCalculator
from gragod.predictions.prediction import get_system_scores
from gragod.types import Datasets


class ThresholdCalculator:
    """
    Class for calculating anomaly detection thresholds using different methods.
    """

    def __init__(
        self,
        dataset: Datasets,
        scores: torch.Tensor,
        labels: torch.Tensor,
        n_thresholds: int,
        range_based: bool = True,
        range_metrics_alpha: float = 0.5,
        system_output_mode: Optional[Literal["max", "mean", "sum"]] = None,
    ):
        """
        Initialize the ThresholdCalculator with data and parameters.

        Args:
            dataset: The dataset being analyzed
            scores: Tensor containing anomaly scores
            labels: Tensor containing ground truth labels
            n_thresholds: Number of threshold values to test
            range_based: Whether to use range-based metrics for evaluation
            range_metrics_alpha: Alpha parameter for range-based recall calculation
            system_output_mode: Mode for aggregating scores in system-level metrics
        """
        self.dataset = dataset
        self.scores = scores
        self.labels = labels
        self.n_thresholds = n_thresholds
        self.range_based = range_based
        self.range_metrics_alpha = range_metrics_alpha
        self.system_output_mode = system_output_mode

    def calculate_f1_optimized(self) -> torch.Tensor:
        """
        Calculate thresholds using the F1 score optimization method.

        Returns:
            Calculated thresholds

        Raises:
            ValueError: If system_output_mode is missing for system-level thresholding
        """
        if self.labels.ndim == 0 or self.labels.shape[1] in [0, 1]:
            if self.system_output_mode is None:
                raise ValueError(
                    "system_output_mode must be provided for system-level metrics"
                )
            return self.calculate_f1_optimized_system()
        else:
            return self.calculate_f1_optimize_per_class()

    def calculate_f1_optimize_per_class(self) -> torch.Tensor:
        """
        Determine optimal thresholds for each feature/class independently.

        This function finds the threshold that maximizes the F1 score for each feature.

        Returns:
            Tensor of shape (n_features,) containing optimal thresholds for each feature
        """
        # Initial best thresholds with highest scores
        max_scores = best_thresholds = torch.max(self.scores, dim=0)[0]
        preds = self.scores > best_thresholds.unsqueeze(0)

        calculator = PerClassCalculator(
            dataset=self.dataset,
            labels=self.labels,
            predictions=preds,
            scores=self.scores,
        )
        if self.range_based:
            precision = calculator.calculate_precision()
            recall = calculator.calculate_range_based_recall(self.range_metrics_alpha)
            f1 = calculator.calculate_f1(precision, recall)
        else:
            precision = calculator.calculate_precision()
            recall = calculator.calculate_recall()
            f1 = calculator.calculate_f1(precision, recall)

        best_f1s = f1.metric_per_class

        thresholds = torch.stack(
            [
                torch.linspace(0, max_score, self.n_thresholds)
                for max_score in max_scores
            ],
            dim=1,
        )
        for threshold in thresholds:
            preds = (self.scores > threshold.unsqueeze(0)).float()

            calculator = PerClassCalculator(
                dataset=self.dataset,
                labels=self.labels,
                predictions=preds,
                scores=self.scores,
            )
            if self.range_based:
                precision = calculator.calculate_precision()
                recall = calculator.calculate_range_based_recall(
                    self.range_metrics_alpha
                )
                f1 = calculator.calculate_f1(precision, recall)
            else:
                precision = calculator.calculate_precision()
                recall = calculator.calculate_recall()
                f1 = calculator.calculate_f1(precision, recall)

            # Update best thresholds where F1 improved
            improved = f1.metric_per_class > best_f1s
            best_f1s[improved] = f1.metric_per_class[improved]
            best_thresholds[improved] = threshold[improved]
        return best_thresholds

    def calculate_f1_optimized_system(self) -> torch.Tensor:
        """
        Determine the optimal system-level threshold for anomaly detection.

        This function finds a single threshold that maximizes the system-level F1 score.

        Returns:
            A single threshold value for system-level anomaly detection

        Raises:
            ValueError: If system_output_mode is not provided
        """
        if self.system_output_mode is None:
            raise ValueError(
                "system_output_mode must be provided for system-level metrics"
            )

        # here we only have system class so there will be only one threshold
        # Initial best thresholds with highest scores
        mode = cast(Literal["max", "mean", "sum"], self.system_output_mode)
        system_scores = get_system_scores(self.scores, mode=mode)
        max_score = best_threshold = torch.max(system_scores)

        system_predictions = (system_scores > max_score).int()

        metrics = SystemCalculator(
            dataset=self.dataset,
            system_labels=self.labels,
            system_predictions=system_predictions,
            system_scores=system_scores,
        )
        if self.range_based:
            precision = metrics.calculate_precision()
            recall = metrics.calculate_range_based_recall(self.range_metrics_alpha)
            f1 = metrics.calculate_f1(precision, recall)
        else:
            precision = metrics.calculate_precision()
            recall = metrics.calculate_recall()
            f1 = metrics.calculate_f1(precision, recall)

        system_f1 = f1.metric_system

        thresholds = torch.linspace(0, max_score, self.n_thresholds)

        for threshold in thresholds:
            system_predictions = (system_scores > threshold).int()

            calculator = SystemCalculator(
                dataset=self.dataset,
                system_labels=self.labels,
                system_predictions=system_predictions,
                system_scores=system_scores,
            )
            if self.range_based:
                precision = calculator.calculate_precision()
                recall = calculator.calculate_range_based_recall(
                    self.range_metrics_alpha
                )
                f1 = calculator.calculate_f1(precision, recall)
            else:
                precision = calculator.calculate_precision()
                recall = calculator.calculate_recall()
                f1 = calculator.calculate_f1(precision, recall)

            # Update best thresholds where F1 improved
            if f1.metric_system > system_f1:
                system_f1 = f1.metric_system
                best_threshold = threshold

        return best_threshold

    def calculate_threshold(
        self,
        method: Literal["f1_optimize"],
    ) -> torch.Tensor:
        """
        Calculate thresholds using the specified method.

        Args:
            method: The threshold calculation method to use:
                - "f1_optimize": Optimize thresholds using F1 score

        Returns:
            Calculated thresholds

        Raises:
            ValueError: If an invalid method is specified
        """
        if method == "f1_optimize":
            return self.calculate_f1_optimized()
        else:
            raise ValueError(f"Unknown threshold calculation method: {method}")


def get_thresholds(
    dataset: Datasets,
    scores: torch.Tensor,
    labels: torch.Tensor,
    method: Literal["f1_optimize"],
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
    calculator = ThresholdCalculator(
        dataset=dataset,
        scores=scores,
        labels=labels,
        n_thresholds=n_thresholds,
        range_based=range_based,
        range_metrics_alpha=range_metrics_alpha,
        system_output_mode=system_output_mode,
    )

    return calculator.calculate_threshold(method=method)
