import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate
import torch
from prts import ts_precision, ts_recall
from pydantic import BaseModel, ConfigDict, Field
from timeeval.metrics import RangeRocVUS

N_TH_SAMPLES_DEFAULT = 1000
N_MAX_BUFFER_SIZE_DEFAULT = 100


class MetricsResult(BaseModel):
    """
    Container for metric calculation results with validation.

    Per class metrics are metric calculated for each class independently.

    Mean metric is the mean of the per-class metrics.

    Global metric is the metric calculated across all classes. It's like flattening the
    tensor and calculating the metric.

    System metric is the metric calculated for the system, where the label/prediction
    are 1 if any of the labels/predictions is 1 for any variable, and 0 otherwise.

    Attributes:
        metric_global: Global metric across all classes.
        metric_mean: Mean metric across classes.
        metric_per_class: Per-class metrics,
        metric_system: System-level metric.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metric_global: float | None = Field(
        ...,
        description="Global metric across all classes",
        ge=0,
        le=1,
    )
    metric_mean: float = Field(
        ...,
        description="Mean metric across classes",
        ge=0,
        le=1,
    )
    metric_per_class: torch.Tensor = Field(..., description="Per-class metrics")
    metric_system: float = Field(
        ...,
        description="System-level metric",
        ge=0,
        le=1,
    )
    round_digits: int = Field(
        default=4, description="Number of decimal places to round to", exclude=True
    )

    def model_post_init(self, _context):
        if self.metric_global is not None:
            self.metric_global = round(self.metric_global, self.round_digits)
        self.metric_mean = round(self.metric_mean, self.round_digits)
        self.metric_system = round(self.metric_system, self.round_digits)

    def model_dump(self, metric_name: str, *args, **kwargs) -> dict:
        """Convert to dictionary with tensor conversion."""
        d = super().model_dump(*args, **kwargs)

        # Convert tensor to list for serialization
        d["metric_per_class"] = [round(x, 4) for x in self.metric_per_class.tolist()]

        d = {k.replace("metric", metric_name): v for k, v in d.items()}

        return d


class MetricsCalculator:
    """Calculator for precision, recall, and F1 metrics."""

    # TODO: Save scores, labels, predictions, system_scores, system_labels,
    #       system_predictions to calculate metrics later
    def __init__(
        self, labels: torch.Tensor, predictions: torch.Tensor, scores: torch.Tensor
    ):
        """
        Initialize calculator with labels and predictions.

        Args:
            labels: Ground truth labels tensor (n_samples, n_nodes)
            predictions: Predicted labels tensor (n_samples, n_nodes)
        """
        self.scores = scores
        self.labels = labels
        self.predictions = predictions
        self.system_scores = torch.sum(scores, dim=1)
        self.system_labels = (torch.sum(labels, dim=1) > 0).int()
        self.system_predictions = (torch.sum(predictions, dim=1) > 0).int()

    def calculate_precision(self) -> MetricsResult:
        """
        Calculate precision metrics.

        Precision = True Positives / Predicted Positives

        Returns:
            MetricsResult: Precision metrics.
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

        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_predicted_positives = torch.sum(self.system_predictions)

        system_precision = (
            system_true_positives / system_predicted_positives
            if system_predicted_positives > 0
            else 0
        )

        return MetricsResult(
            metric_global=float(global_precision),
            metric_mean=float(mean_precision),
            metric_per_class=per_class_precision,
            metric_system=float(system_precision),
        )

    def calculate_recall(self) -> MetricsResult:
        """
        Calculate recall metrics.

        Recall = True Positives / Actual Positives

        Returns:
            MetricsResult: Recall metrics.
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

        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_actual_positives = torch.sum(self.system_labels)
        system_recall = (
            system_true_positives / system_actual_positives
            if system_actual_positives > 0
            else 0
        )

        return MetricsResult(
            metric_global=float(global_recall),
            metric_mean=float(mean_recall),
            metric_per_class=per_class_recall,
            metric_system=float(system_recall),
        )

    def calculate_f1(
        self, precision: MetricsResult, recall: MetricsResult
    ) -> MetricsResult:
        """
        Calculate F1 score from precision and recall results.

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        When Precision + Recall = 0, F1 = 0

        Returns:
            MetricsResult: F1 score metrics.
        """
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

        # Handle division by zero for system metrics
        system_denominator = precision.metric_system + recall.metric_system
        system_f1 = (
            0.0
            if system_denominator == 0
            else (2 * precision.metric_system * recall.metric_system)
            / system_denominator
        )

        return MetricsResult(
            metric_global=global_f1,
            metric_mean=float(mean_f1),
            metric_per_class=per_class_f1,
            metric_system=float(system_f1),
        )

    def calculate_range_based_recall(self, alpha: float = 0.0) -> MetricsResult:
        """
        Calculate range-based recall metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            MetricsResult: Recall metrics.
        """
        labels_np = np.array(self.labels)
        predictions_np = np.array(self.predictions)
        system_labels_np = np.array(self.system_labels)
        system_predictions_np = np.array(self.system_predictions)

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

        mean_recall = torch.mean(torch.tensor(per_class_recall, dtype=torch.float))

        # doesn't make sense the global recall in range based metrics
        global_recall = None

        system_recall = (
            ts_recall(system_labels_np, system_predictions_np, alpha=alpha)
            if not (
                np.allclose(np.unique(system_predictions_np), np.array([0]))
                or np.allclose(np.unique(system_labels_np), np.array([0]))
            )
            else 0
        )

        return MetricsResult(
            metric_global=global_recall,
            metric_mean=float(mean_recall),
            metric_per_class=torch.tensor(per_class_recall),
            metric_system=float(system_recall),
        )

    def calculate_range_based_precision(self, alpha: float = 0.0) -> MetricsResult:
        """
        Calculate range-based precision metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            MetricsResult: Precision metrics.
        """
        labels_np = np.array(self.labels)
        predictions_np = np.array(self.predictions)
        system_labels_np = np.array(self.system_labels)
        system_predictions_np = np.array(self.system_predictions)

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

        mean_precision = torch.mean(
            torch.tensor(per_class_precision, dtype=torch.float)
        )

        # doesn't make sense the global precision in range based metrics
        global_precision = None

        system_precision = (
            ts_precision(system_labels_np, system_predictions_np, alpha=alpha)
            if not (
                np.allclose(np.unique(system_predictions_np), np.array([0]))
                or np.allclose(np.unique(system_labels_np), np.array([0]))
            )
            else 0
        )

        return MetricsResult(
            metric_global=global_precision,
            metric_mean=float(mean_precision),
            metric_per_class=torch.tensor(per_class_precision),
            metric_system=float(system_precision),
        )

    def calculate_vus_roc(
        self,
        max_buffer_size: int = N_MAX_BUFFER_SIZE_DEFAULT,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ):
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
            MetricsResult: VUS-ROC metrics.
        """

        vus_roc = RangeRocVUS(
            max_buffer_size=max_buffer_size,
            compatibility_mode=True,
            max_samples=max_th_samples,
        )
        per_class_vus_roc = [
            vus_roc(y_true=self.labels[:, i].numpy(), y_score=self.scores[:, i].numpy())
            for i in range(self.labels.shape[1])
        ]
        mean_vus_roc = torch.mean(torch.tensor(per_class_vus_roc))

        global_vus_roc = None

        system_vus_roc = vus_roc(
            y_true=self.system_labels.numpy(), y_score=self.system_scores.numpy()
        )

        return MetricsResult(
            metric_global=global_vus_roc,
            metric_mean=float(mean_vus_roc),
            metric_per_class=torch.tensor(per_class_vus_roc),
            metric_system=float(system_vus_roc),
        )

    def get_all_metrics(self, alpha: float = 0.0) -> dict[str, torch.Tensor]:
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
        vus_roc = self.calculate_vus_roc()

        return {
            **precision.model_dump("precision"),
            **recall.model_dump("recall"),
            **f1.model_dump("f1"),
            **range_based_precision.model_dump("range_based_precision"),
            **range_based_recall.model_dump("range_based_recall"),
            **vus_roc.model_dump("vus_roc"),
        }


def visualize_metrics(
    metrics_dict: dict[str, torch.Tensor], output_path: str = "metrics_heatmap.png"
):
    """
    Create and save a heatmap visualization of per-class metrics.

    Args:
        metrics_dict: Dictionary containing calculated metrics
        output_path: Path to save the heatmap image
    """
    metrics_df = pd.DataFrame(
        {
            "Precision": metrics_dict["precision_per_class"],
            "Recall": metrics_dict["recall_per_class"],
            "F1 Score": metrics_dict["f1_per_class"],
            "Range-based Precision": metrics_dict["range_based_precision_per_class"],
            "Range-based Recall": metrics_dict["range_based_recall_per_class"],
            "VUS-ROC": metrics_dict["vus_roc_per_class"],
        }
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df.T, annot=True, cmap="YlOrRd", fmt=".3f")
    plt.title("Performance Metrics per Class")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("\nMetrics per class:")
    print(metrics_df.round(3).to_string())


def get_metrics(
    predictions: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor
) -> dict:
    """
    Calculate and visualize all metrics for given predictions and labels.

    Args:
        predictions: Predicted labels tensor
        labels: Ground truth labels tensor

    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = MetricsCalculator(labels, predictions, scores)
    metrics = calculator.get_all_metrics()

    return metrics


def generate_metrics_table(metrics: dict) -> str:
    """Generate a table of metrics as a string."""
    metrics_table = [
        ["Global", "Mean", "System"],
        [
            "Precision",
            f"{metrics['precision_global']}",
            f"{metrics['precision_mean']}",
            f"{metrics['precision_system']}",
        ],
        [
            "Recall",
            f"{metrics['recall_global']}",
            f"{metrics['recall_mean']}",
            f"{metrics['recall_system']}",
        ],
        [
            "F1",
            f"{metrics['f1_global']}",
            f"{metrics['f1_mean']}",
            f"{metrics['f1_system']}",
        ],
        [
            "Range-based Precision",
            f"{metrics['range_based_precision_global']}",
            f"{metrics['range_based_precision_mean']}",
            f"{metrics['range_based_precision_system']}",
        ],
        [
            "Range-based Recall",
            f"{metrics['range_based_recall_global']}",
            f"{metrics['range_based_recall_mean']}",
            f"{metrics['range_based_recall_system']}",
        ],
        [
            "VUS-ROC",
            f"{metrics['vus_roc_global']}",
            f"{metrics['vus_roc_mean']}",
            f"{metrics['vus_roc_system']}",
        ],
    ]
    return tabulate.tabulate(metrics_table, headers="firstrow", tablefmt="grid")


def generate_metrics_per_class_table(metrics: dict) -> str:
    """Generate a table of per-class metrics as a string."""
    precision = metrics["precision_per_class"]
    recall = metrics["recall_per_class"]
    f1 = metrics["f1_per_class"]
    range_based_precision = metrics["range_based_precision_per_class"]
    range_based_recall = metrics["range_based_recall_per_class"]
    vus_roc = metrics["vus_roc_per_class"]
    metrics_per_class_table = [
        [
            i,
            precision[i],
            recall[i],
            f1[i],
            range_based_precision[i],
            range_based_recall[i],
            vus_roc[i],
        ]
        for i in range(len(precision))
    ]

    return tabulate.tabulate(
        metrics_per_class_table,
        headers=[
            "Class",
            "Precision",
            "Recall",
            "F1",
            "Range-based Precision",
            "Range-based Recall",
            "VUS-ROC",
        ],
        tablefmt="grid",
    )
