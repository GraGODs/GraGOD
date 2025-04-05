import numpy as np
import torch
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from gragod.metrics.per_class_calculator import PerClassCalculator
from gragod.predictions.threshold_calculator import ThresholdCalculator
from gragod.types import Datasets


class PerClassThresholdCalculator(ThresholdCalculator):
    def __init__(
        self,
        dataset: Datasets,
        labels: torch.Tensor,
        n_thresholds: int,
        range_based: bool,
        range_metrics_alpha: float,
        scores: torch.Tensor,
    ):
        super().__init__(
            dataset=dataset,
            labels=labels,
            n_thresholds=n_thresholds,
            range_based=range_based,
            range_metrics_alpha=range_metrics_alpha,
        )
        self.scores = scores

    def calculate_f1_optimized_threshold(self) -> torch.Tensor:
        """
        Determine optimal thresholds for each feature/class independently.

        This function finds the threshold that maximizes the F1 score for each feature.

        Returns:
            Tensor of shape (n_features,) containing optimal thresholds for each feature
        """
        print("Calculating F1 optimized thresholds")
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

    def calculate_otsu_threshold(self) -> torch.Tensor:
        if self.scores is None:
            raise ValueError(
                "System scores must be provided for histogram-based thresholding"
            )
        print("Calculating OTSU thresholds")
        thresholds = []
        for i in range(self.scores.shape[1]):
            threshold = threshold_otsu(self.scores[:, i].numpy())
            thresholds.append(threshold)
        thresholds = torch.tensor(thresholds, device=self.scores.device)
        return thresholds

    def calculate_gmm_threshold(self) -> torch.Tensor:
        print("Calculating GMM thresholds")
        thresholds = []
        for i in range(self.scores.shape[1]):
            data = self.scores[:, i].numpy().reshape(-1, 1)

            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(data)

            sorted_data = np.sort(data.reshape(-1))
            predictions = gmm.predict(sorted_data.reshape(-1, 1))

            for i in range(len(predictions) - 1):
                if predictions[i] != predictions[i + 1]:
                    threshold = (sorted_data[i] + sorted_data[i + 1]) / 2
                    thresholds.append(threshold)

            print("No clear separation found, using median for feature", i)
            thresholds.append(torch.tensor(np.median(data)))

        return torch.tensor(thresholds, device=self.scores.device)

    def calculate_mse_dynamic_threshold(
        self, window_size: int = 100, k: float = 2
    ) -> torch.Tensor:
        """
            Compute thresholds as rolling_mean(MSE) + k × rolling_std(MSE),
            where k is a parameter that controls the sensitivity of the threshold.
            Args:
            dataset: Dataset to use for threshold calculation
            range_metrics_alpha: Alpha parameter for range-based metrics
        Returns:
            Tensor containing the best thresholds for each class
        """
        print("Calculating MSE dynamic thresholds")
        rolling_mean = torch.zeros_like(self.scores)
        rolling_std = torch.zeros_like(self.scores)
        thresholds = torch.zeros_like(self.scores)
        for i in range(self.scores.shape[1]):
            for j in range(self.scores.shape[0]):
                start_idx = max(0, j - window_size + 1)
                window = self.scores[start_idx : j + 1, i]
                if len(window) > 1:
                    rolling_mean[j, i] = torch.mean(window)
                    rolling_std[j, i] = torch.std(window)
                else:
                    rolling_mean[j, i] = window[0]
                    rolling_std[j, i] = 0.0

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

            thresholds[:, i] = dynamic_thresholds

        return thresholds
