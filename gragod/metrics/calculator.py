import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar, Union

import torch

from gragod.metrics.models import PerClassMetricsResult, SystemMetricsResult
from gragod.metrics.visualization import print_all_metrics
from gragod.types import Datasets

N_TH_SAMPLES_DEFAULT = 100
MAX_BUFFER_SIZE_DEFAULT = {
    Datasets.TELCO: 2,
    Datasets.SWAT: 3,
    Datasets.UTE: 4,
}

T = TypeVar("T", bound=Union[PerClassMetricsResult, SystemMetricsResult])


class MetricsCalculator(Generic[T], ABC):
    """Base calculator interface for metrics calculation."""

    def __init__(
        self,
        dataset: Datasets,
    ):
        """
        Initialize calculator.

        Args:
            dataset: Dataset type
        """
        self.dataset = dataset

    @abstractmethod
    def calculate_precision(self) -> T:
        """Calculate precision metrics."""
        pass

    @abstractmethod
    def calculate_recall(self) -> T:
        """Calculate recall metrics."""
        pass

    @abstractmethod
    def calculate_f1(
        self,
        precision: T,
        recall: T,
    ) -> T:
        """Calculate F1 score from precision and recall results."""
        pass

    @abstractmethod
    def calculate_range_based_recall(self, alpha: float) -> T:
        """Calculate range-based recall metrics."""
        pass

    @abstractmethod
    def calculate_range_based_precision(
        self,
    ) -> T:
        """Calculate range-based precision metrics."""
        pass

    @abstractmethod
    def calculate_vus_roc(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> T:
        """Calculate VUS-ROC metrics."""
        pass

    @abstractmethod
    def calculate_vus_pr(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> T:
        """Calculate VUS-PR metrics."""
        pass

    def get_all_metrics(
        self, alpha: float, calculate_vus_metrics: bool
    ) -> dict[str, torch.Tensor]:
        """
        Calculate all metrics and return as dictionary.

        Since the VUS-ROC and VUS-PR metrics are computationally expensive,
        we only calculate them if the flag is set to True.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.
            calculate_vus_metrics: Whether to calculate VUS-ROC and VUS-PR metrics.
        Returns:
            Dict[str, torch.Tensor]: Dictionary of metrics.
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1 = self.calculate_f1(precision, recall)
        range_based_precision = self.calculate_range_based_precision()
        range_based_recall = self.calculate_range_based_recall(alpha=alpha)
        range_based_f1 = self.calculate_f1(range_based_precision, range_based_recall)
        custom_f1 = self.calculate_f1(precision, range_based_recall)

        vus_roc = self.calculate_vus_roc()
        vus_pr = self.calculate_vus_pr()

        return {
            **precision.model_dump("precision"),
            **recall.model_dump("recall"),
            **f1.model_dump("f1"),
            **range_based_precision.model_dump("range_based_precision"),
            **range_based_recall.model_dump("range_based_recall"),
            **range_based_f1.model_dump("range_based_f1"),
            **custom_f1.model_dump("custom_f1"),
            **vus_roc.model_dump("vus_roc"),
            **vus_pr.model_dump("vus_pr"),
        }


def get_metrics(
    dataset: Datasets,
    range_metrics_alpha: float,
    predictions: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    system_predictions: Optional[torch.Tensor] = None,
    system_labels: Optional[torch.Tensor] = None,
    system_scores: Optional[torch.Tensor] = None,
    calculate_vus_metrics: bool = True,
) -> dict:
    """
    Calculate and visualize all metrics for given predictions and labels.

    Args:
        dataset: Dataset type
        range_metrics_alpha: Alpha parameter for range-based metrics
        predictions: Predicted labels tensor (optional)
        labels: Ground truth labels tensor (optional)
        scores: Prediction scores tensor (optional)
        system_predictions: System-level predicted labels tensor (optional)
        system_labels: System-level ground truth labels tensor (optional)
        system_scores: System-level prediction scores tensor (optional)
        only_system_metrics: Whether to calculate only system-level metrics

    Returns:
        Dictionary containing all calculated metrics
    """
    # Import here to avoid circular imports
    from gragod.metrics.per_class_calculator import PerClassCalculator
    from gragod.metrics.system_calculator import SystemCalculator

    if (
        system_predictions is None or system_labels is None or system_scores is None
    ) and (predictions is None or labels is None or scores is None):
        raise ValueError(
            "Either system-level tensors or per-class tensors must be provided"
        )

    # Determine which calculators to use based on available tensors
    calculators = []
    if not (
        system_predictions is None or system_labels is None or system_scores is None
    ):
        calculators.append(
            SystemCalculator(
                dataset=dataset,
                system_labels=system_labels,
                system_predictions=system_predictions,
                system_scores=system_scores,
            )
        )
        print("Going to calculate system metrics")

    if not (predictions is None or labels is None or scores is None):
        calculators.append(
            PerClassCalculator(
                dataset=dataset,
                labels=labels,
                predictions=predictions,
                scores=scores,
                system_labels=system_labels,
                system_predictions=system_predictions,
                system_scores=system_scores,
            )
        )
        print("Going to calculate per-class metrics")
    metrics = {}
    for calculator in calculators:
        metrics.update(
            calculator.get_all_metrics(
                alpha=range_metrics_alpha, calculate_vus_metrics=calculate_vus_metrics
            )
        )

    return metrics


def get_metrics_and_save(
    dataset: Datasets,
    range_metrics_alpha: float,
    save_dir: Path,
    dataset_split: str,
    predictions: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    system_predictions: Optional[torch.Tensor] = None,
    system_labels: Optional[torch.Tensor] = None,
    system_scores: Optional[torch.Tensor] = None,
):
    """
    Calculate metrics and save them to a file.

    Args:
        dataset: Dataset type
        range_metrics_alpha: Alpha parameter for range-based metrics
        save_dir: Directory to save metrics
        dataset_split: Dataset split name
        predictions: Predicted labels tensor (optional)
        labels: Ground truth labels tensor (optional)
        scores: Prediction scores tensor (optional)
        system_predictions: System-level predicted labels tensor (optional)
        system_labels: System-level ground truth labels tensor (optional)
        system_scores: System-level prediction scores tensor (optional)

    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = get_metrics(
        dataset=dataset,
        range_metrics_alpha=range_metrics_alpha,
        predictions=predictions,
        labels=labels,
        scores=scores,
        system_predictions=system_predictions,
        system_labels=system_labels,
        system_scores=system_scores,
    )
    print_all_metrics(metrics, f"------- {dataset_split.capitalize()} -------")
    json.dump(
        metrics,
        open(
            os.path.join(
                save_dir,
                f"{dataset_split}_metrics.json",
            ),
            "w",
        ),
    )
    return metrics
