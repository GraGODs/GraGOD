import numpy as np
import torch
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from gragod.metrics.system_calculator import SystemCalculator
from gragod.predictions.threshold_calculator import ThresholdCalculator
from gragod.types import Datasets


class SystemThresholdCalculator(ThresholdCalculator):
    """
    Calculator for determining system-level thresholds for anomaly detection.

    This class provides methods to calculate various types of thresholds
    at the system level, including F1-optimized, Otsu, GMM, and dynamic
    MSE-based thresholds.
    """

    def __init__(
        self,
        dataset: Datasets,
        labels: torch.Tensor,
        n_thresholds: int,
        system_scores: torch.Tensor,
        range_based: bool,
        range_metrics_alpha,
    ):
        """
        Initialize the SystemThresholdCalculator.

        Args:
            dataset: The dataset type being analyzed
            labels: Ground truth labels for the data
            n_thresholds: Number of threshold values to test
            system_scores: System-level anomaly scores
            range_based: Whether to use range-based metrics
            range_metrics_alpha: Alpha parameter for range-based metrics
        """
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

        This function finds a single threshold that maximizes the system-level F1 score
        by testing multiple threshold values and selecting the one with the highest F1.

        Returns:
            torch.Tensor: A single threshold value for system-level anomaly detection
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
            precision = metrics.calculate_range_based_precision()
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
                precision = calculator.calculate_range_based_precision()
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
        """
        Calculate the Otsu threshold for system-level anomaly detection.

        Uses Otsu's method to find the optimal threshold that separates the
        system scores into two classes (normal and anomalous).

        Returns:
            torch.Tensor: The calculated Otsu threshold
        """
        print("Calculating OTSU threshold")
        threshold = threshold_otsu(self.system_scores.numpy())
        return threshold

    def calculate_gmm_threshold(self) -> torch.Tensor:
        """
        Calculate the GMM-based threshold for system-level anomaly detection.

        Fits a Gaussian Mixture Model with two components to the system scores
        and finds the boundary between the two components as the threshold.

        Returns:
            torch.Tensor: The calculated GMM threshold, or the median if no clear
                          separation is found
        """
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

    def calculate_mse_dynamic_threshold(
        self, window_size: int = 100, k: float = 2
    ) -> torch.Tensor:
        """
        Calculate dynamic thresholds based on rolling statistics of system scores.

        Computes thresholds as rolling_mean(scores) + k × rolling_std(scores),
        where k is a parameter that controls the sensitivity of the threshold.
        This creates a dynamic threshold that adapts to local patterns in the data.

        Args:
            window_size: Size of the rolling window for calculating statistics
            k: Multiplier for the standard deviation, controls threshold sensitivity

        Returns:
            torch.Tensor: Dynamic thresholds for each time point in the system scores
        """
        print("Calculating MSE dynamic threshold")
        rolling_mean = torch.zeros_like(self.system_scores)
        rolling_std = torch.zeros_like(self.system_scores)

        for i in range(len(self.system_scores)):
            start_idx = max(0, i - window_size + 1)
            window = self.system_scores[start_idx : i + 1]
            if len(window) > 1:
                rolling_mean[i] = torch.mean(window)
                rolling_std[i] = torch.std(window)
            else:
                rolling_mean[i] = window[0]
                rolling_std[i] = 0.0

        # Calculate dynamic thresholds
        dynamic_thresholds = rolling_mean + k * rolling_std

        nan_mask = torch.isnan(dynamic_thresholds)
        if nan_mask.any():
            non_nan_values = dynamic_thresholds[~nan_mask]
            if len(non_nan_values) > 0:
                mean_value = torch.mean(non_nan_values)
                dynamic_thresholds = torch.where(
                    nan_mask, mean_value, dynamic_thresholds
                )

        return dynamic_thresholds
