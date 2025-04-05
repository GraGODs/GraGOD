import numpy as np
import torch
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from gragod.metrics.system_calculator import SystemCalculator
from gragod.predictions.threshold_calculator import ThresholdCalculator
from gragod.types import Datasets


class SystemThresholdCalculator(ThresholdCalculator):
    def __init__(
        self,
        dataset: Datasets,
        labels: torch.Tensor,
        n_thresholds: int,
        system_scores: torch.Tensor,
        range_based: bool,
        range_metrics_alpha,
    ):
        super().__init__(
            dataset=dataset,
            labels=labels,
            n_thresholds=n_thresholds,
            range_based=range_based,
            range_metrics_alpha=range_metrics_alpha,
        )
        self.system_scores = system_scores

    def calculate_f1_optimized_threshold(self) -> torch.Tensor:
        """
        Determine the optimal system-level threshold for anomaly detection.

        This function finds a single threshold that maximizes the system-level F1 score.

        Returns:
            A single threshold value for system-level anomaly detection

        Raises:
            ValueError: If system_output_mode is not provided
        """
        # here we only have system class so there will be only one threshold
        # Initial best thresholds with highest scores
        print("Calculating F1 optimized threshold")
        max_score = best_threshold = torch.max(self.system_scores)

        system_predictions = (self.system_scores > max_score).int()

        metrics = SystemCalculator(
            dataset=self.dataset,
            system_labels=self.labels,
            system_predictions=system_predictions,
            system_scores=self.system_scores,
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
            system_predictions = (self.system_scores > threshold).int()

            calculator = SystemCalculator(
                dataset=self.dataset,
                system_labels=self.labels,
                system_predictions=system_predictions,
                system_scores=self.system_scores,
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

    def calculate_otsu_threshold(self) -> torch.Tensor:
        print("Calculating OTSU threshold")
        threshold = threshold_otsu(self.system_scores.numpy())
        return threshold

    def calculate_gmm_threshold(self) -> torch.Tensor:
        print("Calculating GMM threshold")
        data = self.system_scores.numpy().reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data)

        sorted_data = np.sort(data.reshape(-1))
        predictions = gmm.predict(sorted_data.reshape(-1, 1))

        for i in range(len(predictions) - 1):
            if predictions[i] != predictions[i + 1]:
                threshold = (sorted_data[i] + sorted_data[i + 1]) / 2
                return torch.tensor(threshold)

        print("No clear separation found, using median")
        return torch.tensor(np.median(data))
