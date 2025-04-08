import numpy as np
import torch
from prts import ts_precision, ts_recall
from timeeval.metrics.vus_metrics import RangePrVUS, RangeRocVUS

from gragod.metrics.calculator import (
    MAX_BUFFER_SIZE_DEFAULT,
    N_TH_SAMPLES_DEFAULT,
    MetricsCalculator,
)
from gragod.metrics.models import SystemMetricsResult
from gragod.types import Datasets


class SystemCalculator(MetricsCalculator[SystemMetricsResult]):
    """Calculator for system-level precision, recall, and F1 metrics."""

    def __init__(
        self,
        dataset: Datasets,
        system_labels: torch.Tensor,
        system_predictions: torch.Tensor,
        system_scores: torch.Tensor,
    ):
        """
        Initialize calculator with system-level labels and predictions.

        Args:
            dataset: Dataset type
            system_labels: System-level ground truth labels tensor (n_samples)
            system_predictions: System-level predicted labels tensor (n_samples)
            system_scores: System-level prediction scores tensor (n_samples)
        """
        super().__init__(dataset)
        self.system_labels = system_labels.squeeze()
        self.system_predictions = system_predictions.squeeze()
        self.system_scores = system_scores.squeeze()

    def calculate_precision(self) -> SystemMetricsResult:
        """
        Calculate precision metrics.

        Precision = True Positives / Predicted Positives

        Returns:
            SystemMetricsResult: Precision metrics.
        """
        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_predicted_positives = torch.sum(self.system_predictions)

        system_precision = (
            system_true_positives / system_predicted_positives
            if system_predicted_positives > 0
            else 0
        )
        return SystemMetricsResult(metric_system=float(system_precision))

    def calculate_recall(self) -> SystemMetricsResult:
        """
        Calculate recall metrics.

        Recall = True Positives / Actual Positives

        Returns:
            SystemMetricsResult: Recall metrics.
        """
        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_actual_positives = torch.sum(self.system_labels)
        system_recall = (
            system_true_positives / system_actual_positives
            if system_actual_positives > 0
            else 0
        )

        return SystemMetricsResult(metric_system=float(system_recall))

    def calculate_f1(
        self, precision: SystemMetricsResult, recall: SystemMetricsResult
    ) -> SystemMetricsResult:
        """
        Calculate F1 score from precision and recall results.

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        When Precision + Recall = 0, F1 = 0

        Returns:
            SystemMetricsResult: F1 score metrics.
        """
        system_f1 = (
            0.0
            if precision.metric_system + recall.metric_system == 0
            else (2 * precision.metric_system * recall.metric_system)
            / (precision.metric_system + recall.metric_system)
        )
        return SystemMetricsResult(metric_system=float(system_f1))

    def calculate_range_based_recall(self, alpha: float) -> SystemMetricsResult:
        """
        Calculate range-based recall metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            SystemMetricsResult: Recall metrics.
        """
        system_labels_np = np.array(self.system_labels)
        system_predictions_np = np.array(self.system_predictions)

        system_recall = (
            ts_recall(
                system_labels_np,
                system_predictions_np,
                alpha=alpha,
                cardinality="reciprocal",
                bias="flat",
            )
            if not (
                np.allclose(np.unique(system_predictions_np), np.array([0]))
                or np.allclose(np.unique(system_labels_np), np.array([0]))
            )
            else 0
        )

        return SystemMetricsResult(metric_system=float(system_recall))

    def calculate_range_based_precision(self) -> SystemMetricsResult:
        """
        Calculate range-based precision metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Returns:
            SystemMetricsResult: Precision metrics.
        """
        system_labels_np = np.array(self.system_labels)
        system_predictions_np = np.array(self.system_predictions)
        system_precision = (
            ts_precision(
                system_labels_np,
                system_predictions_np,
                alpha=0,
                cardinality="reciprocal",
                bias="flat",
            )
            if not (
                np.allclose(np.unique(system_predictions_np), np.array([0]))
                or np.allclose(np.unique(system_labels_np), np.array([0]))
            )
            else 0
        )

        return SystemMetricsResult(metric_system=float(system_precision))

    def calculate_vus_roc(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> SystemMetricsResult:
        """
        Calculate VUS-ROC metrics.
        Based on https://www.paparrizos.org/papers/PaparrizosVLDB22b.pdf.

        Args:
            max_buffer_size: Maximum size of the buffer region around an anomaly.
                We iterate over all buffer sizes from 0 to ``max_buffer_size`` to
                create the surface.
            max_th_samples: Calculating precision and recall for many thresholds is
                quite slow. We, therefore, uniformly sample thresholds from the
                available score space. This parameter controls the maximum number of
                thresholds; too low numbers degrade the metrics' quality.

        Returns:
            SystemMetricsResult: VUS-ROC metrics.
        """
        if max_buffer_size is None:
            max_buffer_size = MAX_BUFFER_SIZE_DEFAULT[self.dataset]

        system_labels_float64 = np.array(self.system_labels, dtype=np.float64)
        system_scores_float64 = np.array(self.system_scores, dtype=np.float64)

        vus_roc = RangeRocVUS(
            max_buffer_size=max_buffer_size,
            compatibility_mode=True,
            max_samples=max_th_samples,
        )

        system_vus_roc = (
            vus_roc(
                y_true=system_labels_float64,
                y_score=system_scores_float64,
            )
            if torch.sum(self.system_labels) > 0
            else 0
        )

        return SystemMetricsResult(metric_system=float(system_vus_roc))

    def calculate_vus_pr(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> SystemMetricsResult:
        """
        Calculate VUS-PR metrics.
        Based on https://www.paparrizos.org/papers/PaparrizosVLDB22b.pdf.

        Args:
            max_buffer_size: Maximum size of the buffer region around an anomaly.
                We iterate over all buffer sizes from 0 to ``max_buffer_size`` to
                create the surface.
            max_th_samples: Calculating precision and recall for many thresholds is
                quite slow. We, therefore, uniformly sample thresholds from the
                available score space. This parameter controls the maximum number of
                thresholds; too low numbers degrade the metrics' quality.

        Returns:
            SystemMetricsResult: VUS-PR metrics.
        """
        if max_buffer_size is None:
            max_buffer_size = MAX_BUFFER_SIZE_DEFAULT[self.dataset]

        system_labels_float64 = np.array(self.system_labels, dtype=np.float64)
        system_scores_float64 = np.array(self.system_scores, dtype=np.float64)

        vus_pr = RangePrVUS(
            max_buffer_size=max_buffer_size,
            compatibility_mode=True,
            max_samples=max_th_samples,
        )

        system_vus_pr = (
            vus_pr(
                y_true=system_labels_float64,
                y_score=system_scores_float64,
            )
            if torch.sum(self.system_labels) > 0
            else 0
        )

        return SystemMetricsResult(metric_system=float(system_vus_pr))
