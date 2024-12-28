import numpy as np
import torch
from prts import ts_precision, ts_recall
from timeeval.metrics.vus_metrics import RangeRocVUS

from gragod.metrics.models import MetricsResult, SystemMetricsResult
from gragod.types import Datasets

N_TH_SAMPLES_DEFAULT = 100
MAX_BUFFER_SIZE_DEFAULT = {Datasets.TELCO: 2, Datasets.SWAT: 3}


class MetricsCalculator:
    """Calculator for precision, recall, and F1 metrics."""

    # TODO: Save scores, labels, predictions, system_scores, system_labels,
    #       system_predictions to calculate metrics later
    def __init__(
        self,
        dataset: Datasets,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        scores: torch.Tensor,
    ):
        """
        Initialize calculator with labels and predictions.

        Args:
            labels: Ground truth labels tensor (n_samples, n_nodes)
            predictions: Predicted labels tensor (n_samples, n_nodes)
        """
        self.dataset = dataset
        self.scores = scores
        self.labels = labels
        self.predictions = predictions
        self.system_scores = torch.sum(scores, dim=1)
        self.system_labels = (torch.sum(labels, dim=1) > 0).int()
        self.system_predictions = (torch.sum(predictions, dim=1) > 0).int()

        self.calculate_only_system_metrics = labels.ndim == 0 or labels.shape[1] in [
            0,
            1,
        ]

    def calculate_precision(self) -> MetricsResult | SystemMetricsResult:
        """
        Calculate precision metrics.

        Precision = True Positives / Predicted Positives

        Returns:
            MetricsResult | SystemMetricsResult: Precision metrics.
        """
        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_predicted_positives = torch.sum(self.system_predictions)

        system_precision = (
            system_true_positives / system_predicted_positives
            if system_predicted_positives > 0
            else 0
        )
        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_precision))

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

        return MetricsResult(
            metric_global=float(global_precision),
            metric_mean=float(mean_precision),
            metric_per_class=per_class_precision,
            metric_system=float(system_precision),
        )

    def calculate_recall(self) -> MetricsResult | SystemMetricsResult:
        """
        Calculate recall metrics.

        Recall = True Positives / Actual Positives

        Returns:
            MetricsResult | SystemMetricsResult: Recall metrics.
        """
        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_actual_positives = torch.sum(self.system_labels)
        system_recall = (
            system_true_positives / system_actual_positives
            if system_actual_positives > 0
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_recall))

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

        return MetricsResult(
            metric_global=float(global_recall),
            metric_mean=float(mean_recall),
            metric_per_class=per_class_recall,
            metric_system=float(system_recall),
        )

    def calculate_f1(
        self,
        precision: MetricsResult | SystemMetricsResult,
        recall: MetricsResult | SystemMetricsResult,
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate F1 score from precision and recall results.

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        When Precision + Recall = 0, F1 = 0

        Returns:
            MetricsResult | SystemMetricsResult: F1 score metrics.
        """

        # Handle division by zero for system metrics
        system_denominator = precision.metric_system + recall.metric_system
        system_f1 = (
            0.0
            if system_denominator == 0
            else (2 * precision.metric_system * recall.metric_system)
            / system_denominator
        )

        if isinstance(precision, SystemMetricsResult) or isinstance(
            recall, SystemMetricsResult
        ):
            return SystemMetricsResult(metric_system=float(system_f1))

        # Handle division by zero for per-class metrics
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
        if precision.metric_global is not None and recall.metric_global is not None:
            global_denominator = precision.metric_global + recall.metric_global
            global_f1 = float(
                0.0
                if global_denominator == 0
                else (2 * precision.metric_global * recall.metric_global)
                / global_denominator
            )
        else:
            global_f1 = None

        return MetricsResult(
            metric_global=global_f1,
            metric_mean=float(mean_f1),
            metric_per_class=per_class_f1,
            metric_system=float(system_f1),
        )

    def calculate_range_based_recall(
        self, alpha: float = 1.0
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate range-based recall metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            MetricsResult | SystemMetricsResult: Recall metrics.
        """
        system_labels_np = np.array(self.system_labels)
        system_predictions_np = np.array(self.system_predictions)

        system_recall = (
            ts_recall(system_labels_np, system_predictions_np, alpha=alpha)
            if not (
                np.allclose(np.unique(system_predictions_np), np.array([0]))
                or np.allclose(np.unique(system_labels_np), np.array([0]))
            )
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_recall))

        labels_np = np.array(self.labels)
        predictions_np = np.array(self.predictions)

        per_class_recall = [
            (
                ts_recall(labels_np[:, i], predictions_np[:, i], alpha=alpha)
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
        global_recall = None

        return MetricsResult(
            metric_global=global_recall,
            metric_mean=float(mean_recall),
            metric_per_class=per_class_recall,
            metric_system=float(system_recall),
        )

    def calculate_range_based_precision(
        self, alpha: float = 1.0
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate range-based precision metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            MetricsResult | SystemMetricsResult: Precision metrics.
        """
        system_labels_np = np.array(self.system_labels)
        system_predictions_np = np.array(self.system_predictions)

        system_precision = (
            ts_precision(system_labels_np, system_predictions_np, alpha=alpha)
            if not (
                np.allclose(np.unique(system_predictions_np), np.array([0]))
                or np.allclose(np.unique(system_labels_np), np.array([0]))
            )
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_precision))

        labels_np = np.array(self.labels)
        predictions_np = np.array(self.predictions)

        per_class_precision = [
            (
                ts_precision(labels_np[:, i], predictions_np[:, i], alpha=alpha)
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
        global_precision = None

        return MetricsResult(
            metric_global=global_precision,
            metric_mean=float(mean_precision),
            metric_per_class=per_class_precision,
            metric_system=float(system_precision),
        )

    def calculate_range_based_f1(
        self,
        range_based_precision: MetricsResult | SystemMetricsResult,
        range_based_recall: MetricsResult | SystemMetricsResult,
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate range-based F1 score metrics.
        """
        return self.calculate_f1(range_based_precision, range_based_recall)

    def calculate_vus_roc(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> MetricsResult | SystemMetricsResult:
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
            MetricsResult | SystemMetricsResult: VUS-ROC metrics.
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

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_vus_roc))

        scores_float64 = np.array(self.scores, dtype=np.float64)
        labels_float64 = np.array(self.labels, dtype=np.float64)

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

        global_vus_roc = None

        return MetricsResult(
            metric_global=global_vus_roc,
            metric_mean=float(mean_vus_roc),
            metric_per_class=torch.tensor(per_class_vus_roc),
            metric_system=float(system_vus_roc),
        )

    def get_all_metrics(self, alpha: float = 1.0) -> dict[str, torch.Tensor]:
        """
        Calculate all metrics and return as dictionary.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of metrics.
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1 = self.calculate_f1(precision, recall)
        range_based_precision = self.calculate_range_based_precision(alpha=alpha)
        range_based_recall = self.calculate_range_based_recall(alpha=alpha)
        range_based_f1 = self.calculate_range_based_f1(
            range_based_precision, range_based_recall
        )
        vus_roc = self.calculate_vus_roc()

        return {
            **precision.model_dump("precision"),
            **recall.model_dump("recall"),
            **f1.model_dump("f1"),
            **range_based_precision.model_dump("range_based_precision"),
            **range_based_recall.model_dump("range_based_recall"),
            **range_based_f1.model_dump("range_based_f1"),
            **vus_roc.model_dump("vus_roc"),
        }


def get_metrics(
    dataset: Datasets,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    range_metrics_alpha: float = 1.0,
) -> dict:
    """
    Calculate and visualize all metrics for given predictions and labels.

    Args:
        predictions: Predicted labels tensor
        labels: Ground truth labels tensor

    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = MetricsCalculator(
        dataset=dataset, labels=labels, predictions=predictions, scores=scores
    )
    metrics = calculator.get_all_metrics(alpha=range_metrics_alpha)

    return metrics
