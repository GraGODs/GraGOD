import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tabulate
import torch
from pydantic import BaseModel, ConfigDict, Field


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
    metric_global: float = Field(
        ...,
        description="Global metric across all classes",
        gt=0,
        le=1,
    )
    metric_mean: float = Field(
        ...,
        description="Mean metric across classes",
        gt=0,
        le=1,
    )
    metric_per_class: torch.Tensor = Field(..., description="Per-class metrics")
    metric_system: float = Field(
        ...,
        description="System-level metric",
        gt=0,
        le=1,
    )
    round_digits: int = Field(
        default=4, description="Number of decimal places to round to", exclude=True
    )

    def model_post_init(self, _context):
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

    def __init__(self, labels: torch.Tensor, predictions: torch.Tensor):
        """
        Initialize calculator with labels and predictions.

        Args:
            labels: Ground truth labels tensor (n_samples, n_nodes)
            predictions: Predicted labels tensor (n_samples, n_nodes)
        """
        self.labels = labels
        self.predictions = predictions
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

        per_class_precision = true_positives / predicted_positives
        global_precision = true_positives.sum() / predicted_positives.sum()
        mean_precision = torch.mean(per_class_precision)

        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_predicted_positives = torch.sum(self.system_predictions)
        system_precision = system_true_positives / system_predicted_positives

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

        per_class_recall = true_positives / actual_positives
        mean_recall = torch.mean(per_class_recall)
        global_recall = true_positives.sum() / actual_positives.sum()

        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_actual_positives = torch.sum(self.system_labels)
        system_recall = system_true_positives / system_actual_positives

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
        global_denominator = precision.metric_global + recall.metric_global
        global_f1 = (
            0.0
            if global_denominator == 0
            else (2 * precision.metric_global * recall.metric_global)
            / global_denominator
        )

        # Handle division by zero for system metrics
        system_denominator = precision.metric_system + recall.metric_system
        system_f1 = (
            0.0
            if system_denominator == 0
            else (2 * precision.metric_system * recall.metric_system)
            / system_denominator
        )

        return MetricsResult(
            metric_global=float(global_f1),
            metric_mean=float(mean_f1),
            metric_per_class=per_class_f1,
            metric_system=float(system_f1),
        )

    def get_all_metrics(self) -> dict[str, torch.Tensor]:
        """
        Calculate all metrics and return as dictionary.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of metrics.
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1 = self.calculate_f1(precision, recall)

        return {
            **precision.model_dump("precision"),
            **recall.model_dump("recall"),
            **f1.model_dump("f1"),
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
            "Precision": metrics_dict["per_class_precision"].cpu().numpy(),
            "Recall": metrics_dict["per_class_recall"].cpu().numpy(),
            "F1 Score": metrics_dict["per_class_f1"].cpu().numpy(),
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


def get_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Calculate and visualize all metrics for given predictions and labels.

    Args:
        predictions: Predicted labels tensor
        labels: Ground truth labels tensor

    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = MetricsCalculator(labels, predictions)
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
    ]
    return tabulate.tabulate(metrics_table, headers="firstrow", tablefmt="grid")


def generate_metrics_per_class_table(metrics: dict) -> str:
    """Generate a table of per-class metrics as a string."""
    precision = metrics["precision_per_class"]
    recall = metrics["recall_per_class"]
    f1 = metrics["f1_per_class"]
    metrics_per_class_table = [
        [i, precision[i], recall[i], f1[i]] for i in range(len(precision))
    ]

    return tabulate.tabulate(
        metrics_per_class_table,
        headers=["Class", "Precision", "Recall", "F1"],
        tablefmt="grid",
    )
