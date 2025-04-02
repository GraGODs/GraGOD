import numpy as np
import torch
from prts import ts_precision, ts_recall
from timeeval.metrics.vus_metrics import RangePrVUS, RangeRocVUS

from gragod.metrics.calculator import (
    MAX_BUFFER_SIZE_DEFAULT,
    N_TH_SAMPLES_DEFAULT,
    MetricsCalculator,
)
from gragod.metrics.models import PerClassMetricsResult
from gragod.types import Datasets


class PerClassCalculator(MetricsCalculator[PerClassMetricsResult]):
    """Calculator for per-class precision, recall, and F1 metrics."""

    def __init__(
        self,
        dataset: Datasets,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        scores: torch.Tensor,
        system_labels: torch.Tensor | None = None,
        system_predictions: torch.Tensor | None = None,
        system_scores: torch.Tensor | None = None,
    ):
        """
        Initialize calculator with per-class labels and predictions.

        Args:
            dataset: Dataset type
            labels: Ground truth labels tensor (n_samples, n_nodes)
            predictions: Predicted labels tensor (n_samples, n_nodes)
            scores: Prediction scores tensor (n_samples, n_nodes)
            system_labels: System-level ground truth labels tensor (n_samples)
            system_predictions: System-level predicted labels tensor (n_samples)
            system_scores: System-level prediction scores tensor (n_samples)
        """
        super().__init__(dataset)
        self.labels = labels
        self.predictions = predictions
        self.scores = scores

        # If system-level metrics are not provided, we don't calculate them
        self.calculate_only_system_metrics = False

        # System-level metrics - if not provided, we don't calculate them
        self.system_labels = system_labels
        self.system_predictions = system_predictions
        self.system_scores = system_scores

    def calculate_precision(self) -> PerClassMetricsResult:
        """
        Calculate precision metrics.

        Precision = True Positives / Predicted Positives

        Returns:
            PerClassMetricsResult: Precision metrics.
        """
        true_positives = torch.sum((self.labels == 1) & (self.predictions == 1), dim=0)
        predicted_positives = torch.sum(self.predictions == 1, dim=0)

        per_class_precision = torch.where(
            predicted_positives > 0,
            true_positives / predicted_positives,
            torch.zeros_like(predicted_positives, dtype=torch.float),
        )
        global_precision = (
            true_positives.sum() / predicted_positives.sum()
            if predicted_positives.sum() > 0
            else 0
        )
        mean_precision = torch.mean(per_class_precision)

        return PerClassMetricsResult(
            metric_global=float(global_precision),
            metric_mean=float(mean_precision),
            metric_per_class=per_class_precision,
        )

    def calculate_recall(self) -> PerClassMetricsResult:
        """
        Calculate recall metrics.

        Recall = True Positives / Actual Positives

        Returns:
            PerClassMetricsResult: Recall metrics.
        """
        true_positives = torch.sum((self.labels == 1) & (self.predictions == 1), dim=0)
        actual_positives = torch.sum(self.labels == 1, dim=0)

        per_class_recall = torch.where(
            actual_positives > 0,
            true_positives / actual_positives,
            torch.zeros_like(actual_positives, dtype=torch.float),
        )

        mean_recall = torch.mean(per_class_recall)
        global_recall = (
            true_positives.sum() / actual_positives.sum()
            if actual_positives.sum() > 0
            else 0
        )

        return PerClassMetricsResult(
            metric_global=float(global_recall),
            metric_mean=float(mean_recall),
            metric_per_class=per_class_recall,
        )

    def calculate_f1(
        self,
        precision: PerClassMetricsResult,
        recall: PerClassMetricsResult,
    ) -> PerClassMetricsResult:
        """
        Calculate F1 score from precision and recall results.

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        When Precision + Recall = 0, F1 = 0

        Returns:
            PerClassMetricsResult: F1 score metrics.
        """
        denominator = precision.metric_per_class + recall.metric_per_class
        per_class_f1 = torch.zeros_like(denominator)
        non_zero_mask = denominator > 0
        per_class_f1[non_zero_mask] = (
            2
            * (
                precision.metric_per_class[non_zero_mask]
                * recall.metric_per_class[non_zero_mask]
            )
            / denominator[non_zero_mask]
        )
        mean_f1 = torch.mean(per_class_f1)

        # Handle division by zero for global metrics
        global_f1 = 0.0
        if precision.metric_global is not None and recall.metric_global is not None:
            global_denominator = precision.metric_global + recall.metric_global
            global_f1 = float(
                0.0
                if global_denominator == 0
                else (2 * precision.metric_global * recall.metric_global)
                / global_denominator
            )

        # Return PerClassMetricsResult
        return PerClassMetricsResult(
            metric_global=global_f1,
            metric_mean=float(mean_f1),
            metric_per_class=per_class_f1,
        )

    def calculate_range_based_recall(self, alpha: float) -> PerClassMetricsResult:
        """
        Calculate range-based recall metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            PerClassMetricsResult: Recall metrics.
        """
        labels_np = np.array(self.labels)
        predictions_np = np.array(self.predictions)

        per_class_recall = [
            (
                ts_recall(
                    labels_np[:, i],
                    predictions_np[:, i],
                    alpha=alpha,
                    cardinality="reciprocal",
                    bias="flat",
                )
                # if there are no anomalies detected, recall is 0
                if not (
                    np.allclose(np.unique(predictions_np[:, i]), np.array([0]))
                    or np.allclose(np.unique(labels_np[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(self.labels.shape[1])
        ]
        per_class_recall = torch.tensor(per_class_recall, dtype=torch.float)
        mean_recall = torch.mean(per_class_recall)

        # doesn't make sense the global recall in range based metrics
        global_recall = 0.0

        return PerClassMetricsResult(
            metric_global=global_recall,
            metric_mean=float(mean_recall),
            metric_per_class=per_class_recall,
        )

    def calculate_range_based_precision(self) -> PerClassMetricsResult:
        """
        Calculate range-based precision metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Returns:
            PerClassMetricsResult: Precision metrics.
        """
        labels_np = np.array(self.labels)
        predictions_np = np.array(self.predictions)

        per_class_precision = [
            (
                ts_precision(
                    labels_np[:, i],
                    predictions_np[:, i],
                    alpha=0,
                    cardinality="reciprocal",
                    bias="flat",
                )
                # if there are no anomalies detected, precision is 0
                if not (
                    np.allclose(np.unique(predictions_np[:, i]), np.array([0]))
                    or np.allclose(np.unique(labels_np[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(self.labels.shape[1])
        ]
        per_class_precision = torch.tensor(per_class_precision, dtype=torch.float)

        mean_precision = torch.mean(per_class_precision)

        # doesn't make sense the global precision in range based metrics
        global_precision = 0.0

        return PerClassMetricsResult(
            metric_global=global_precision,
            metric_mean=float(mean_precision),
            metric_per_class=per_class_precision,
        )

    def calculate_vus_roc(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> PerClassMetricsResult:
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
            PerClassMetricsResult: VUS-ROC metrics.
        """
        if max_buffer_size is None:
            max_buffer_size = MAX_BUFFER_SIZE_DEFAULT[self.dataset]

        scores_float64 = np.array(self.scores, dtype=np.float64)
        labels_float64 = np.array(self.labels, dtype=np.float64)

        vus_roc = RangeRocVUS(
            max_buffer_size=max_buffer_size,
            compatibility_mode=True,
            max_samples=max_th_samples,
        )

        per_class_vus_roc = [
            (
                vus_roc(
                    y_true=labels_float64[:, i],
                    y_score=scores_float64[:, i],
                )
                if not (
                    np.allclose(np.unique(labels_float64[:, i]), np.array([0]))
                    or np.allclose(np.unique(scores_float64[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(labels_float64.shape[1])
        ]
        mean_vus_roc = torch.mean(torch.tensor(per_class_vus_roc))

        global_vus_roc = 0.0

        return PerClassMetricsResult(
            metric_global=global_vus_roc,
            metric_mean=float(mean_vus_roc),
            metric_per_class=torch.tensor(per_class_vus_roc),
        )

    def calculate_vus_pr(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> PerClassMetricsResult:
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
            PerClassMetricsResult: VUS-PR metrics.
        """
        if max_buffer_size is None:
            max_buffer_size = MAX_BUFFER_SIZE_DEFAULT[self.dataset]

        scores_float64 = np.array(self.scores, dtype=np.float64)
        labels_float64 = np.array(self.labels, dtype=np.float64)

        vus_pr = RangePrVUS(
            max_buffer_size=max_buffer_size,
            compatibility_mode=True,
            max_samples=max_th_samples,
        )

        per_class_vus_pr = [
            (
                vus_pr(
                    y_true=labels_float64[:, i],
                    y_score=scores_float64[:, i],
                )
                if not (
                    np.allclose(np.unique(labels_float64[:, i]), np.array([0]))
                    or np.allclose(np.unique(scores_float64[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(labels_float64.shape[1])
        ]
        mean_vus_pr = torch.mean(torch.tensor(per_class_vus_pr))

        global_vus_pr = 0.0

        return PerClassMetricsResult(
            metric_global=global_vus_pr,
            metric_mean=float(mean_vus_pr),
            metric_per_class=torch.tensor(per_class_vus_pr),
        )
