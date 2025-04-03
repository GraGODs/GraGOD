from typing import Literal

import torch

from gragod.metrics.models import SystemMetricsResult
from gragod.metrics.per_class_calculator import PerClassCalculator
from gragod.metrics.system_calculator import SystemCalculator
from gragod.predictions.spot import SPOT
from gragod.types import Datasets


def get_threshold(
    dataset: Datasets,
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int,
    range_based: bool = True,
    range_metrics_alpha: float = 0.5,
    system_output_mode: Literal["max", "mean", "sum"] | None = None,
) -> torch.Tensor:
    """
    Get the optimal threshold for anomaly detection based on the input data.

    This function determines whether to use per-class or system-level thresholding
    based on the shape of the labels tensor.

    Args:
        dataset: The dataset being analyzed
        scores: Tensor of shape (n_samples, n_features) containing anomaly scores
        labels: Tensor containing ground truth labels
        n_thresholds: Number of threshold values to test
        range_based: Whether to use range-based metrics for evaluation
        range_metrics_alpha: Alpha parameter for range-based recall calculation
        system_output_mode: Mode for aggregating scores in system-level metrics
            (required when using system-level thresholding)

    Returns:
        The optimal threshold(s) for anomaly detection

    Raises:
        ValueError: If system_output_mode is not provided for system-level metrics
    """
    if labels.ndim == 0 or labels.shape[1] in [0, 1]:
        if system_output_mode is None:
            raise ValueError(
                "system_output_mode must be provided for system-level metrics"
            )
        return get_threshold_system(
            dataset=dataset,
            scores=scores,
            labels=labels,
            n_thresholds=n_thresholds,
            range_based=range_based,
            range_metrics_alpha=range_metrics_alpha,
            system_output_mode=system_output_mode,
        )
    else:
        return get_threshold_per_class(
            dataset=dataset,
            scores=scores,
            labels=labels,
            n_thresholds=n_thresholds,
            range_based=range_based,
            range_metrics_alpha=range_metrics_alpha,
        )


def get_system_scores(
    scores: torch.Tensor,
    mode: Literal["max", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    Aggregate feature-level scores into system-level scores.

    Args:
        scores: Tensor of shape (n_samples, n_features) containing anomaly scores
        mode: Aggregation method to use:
            - "max": Maximum score across features
            - "mean": Average score across features
            - "sum": Sum of scores across features

    Returns:
        Tensor of shape (n_samples, 1) containing system-level scores
    """
    if mode == "max":
        system_scores = torch.max(scores, dim=1).values
    elif mode == "mean":
        system_scores = torch.mean(scores, dim=1)
    elif mode == "sum":
        system_scores = torch.sum(scores, dim=1)

    return system_scores.unsqueeze(1)


def get_threshold_per_class(
    dataset: Datasets,
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int,
    range_based: bool = True,
    range_metrics_alpha: float = 0.5,
) -> torch.Tensor:
    """
    Determine optimal thresholds for each feature/class independently.

    This function finds the threshold that maximizes the F1 score for each feature.

    Args:
        dataset: The dataset being analyzed
        scores: Tensor of shape (n_samples, n_features) containing anomaly scores
        labels: Tensor of shape (n_samples, n_features) containing ground truth labels
        n_thresholds: Number of threshold values to test
        range_based: Whether to use range-based metrics for evaluation
        range_metrics_alpha: Alpha parameter for range-based recall calculation

    Returns:
        Tensor of shape (n_features,) containing optimal thresholds for each feature
    """
    # Initial best thresholds with highest scores
    max_scores = best_thresholds = torch.max(scores, dim=0)[0]
    preds = scores > best_thresholds.unsqueeze(0)

    calculator = PerClassCalculator(
        dataset=dataset,
        labels=labels,
        predictions=preds,
        scores=scores,
    )
    if range_based:
        precision = calculator.calculate_precision()
        recall = calculator.calculate_range_based_recall(range_metrics_alpha)
        f1 = calculator.calculate_f1(precision, recall)
    else:
        precision = calculator.calculate_precision()
        recall = calculator.calculate_recall()
        f1 = calculator.calculate_f1(precision, recall)

    # Check if we got a SystemMetricsResult
    if isinstance(f1, SystemMetricsResult):
        raise ValueError(
            "Expected per-class metrics but got system metrics."
            "Check input dimensions."
        )

    best_f1s = f1.metric_per_class

    thresholds = torch.stack(
        [torch.linspace(0, max_score, n_thresholds) for max_score in max_scores],
        dim=1,
    )
    for threshold in thresholds:
        preds = (scores > threshold.unsqueeze(0)).float()

        calculator = PerClassCalculator(
            dataset=dataset, labels=labels, predictions=preds, scores=scores
        )
        if range_based:
            precision = calculator.calculate_precision()
            recall = calculator.calculate_range_based_recall(range_metrics_alpha)
            f1 = calculator.calculate_f1(precision, recall)
        else:
            precision = calculator.calculate_precision()
            recall = calculator.calculate_recall()
            f1 = calculator.calculate_f1(precision, recall)

        if isinstance(f1, SystemMetricsResult):
            raise ValueError(
                "Expected per-class metrics but got system metrics."
                "Check input dimensions."
            )

        # Update best thresholds where F1 improved
        improved = f1.metric_per_class > best_f1s
        best_f1s[improved] = f1.metric_per_class[improved]
        best_thresholds[improved] = threshold[improved]
    return best_thresholds


def get_threshold_system(
    dataset: Datasets,
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int,
    system_output_mode: Literal["max", "mean", "sum"],
    range_based: bool = True,
    range_metrics_alpha: float = 0.5,
) -> torch.Tensor:
    """
    Determine the optimal system-level threshold for anomaly detection.

    This function finds a single threshold that maximizes the system-level F1 score.

    Args:
        dataset: The dataset being analyzed
        scores: Tensor of shape (n_samples, n_features) with anomaly scores
        labels: Tensor of shape (n_samples, 1) with system-level ground truth labels
        n_thresholds: Number of threshold values to test
        range_based: Whether to use range-based metrics for evaluation
        range_metrics_alpha: Alpha parameter for range-based recall calculation
        system_output_mode: Mode for aggregating scores in system-level metrics

    Returns:
        A single threshold value for system-level anomaly detection
    """
    # here we only have system class so there will be only one threshold
    # Initial best thresholds with highest scores
    system_scores = get_system_scores(scores, system_output_mode)
    max_score = best_threshold = torch.max(system_scores)

    system_predictions = (system_scores > max_score).int()

    metrics = SystemCalculator(
        dataset=dataset,
        system_labels=labels,
        system_predictions=system_predictions,
        system_scores=system_scores,
    )
    if range_based:
        precision = metrics.calculate_precision()
        recall = metrics.calculate_range_based_recall(range_metrics_alpha)
        f1 = metrics.calculate_f1(precision, recall)
    else:
        precision = metrics.calculate_precision()
        recall = metrics.calculate_recall()
        f1 = metrics.calculate_f1(precision, recall)

    system_f1 = f1.metric_system

    thresholds = torch.linspace(0, max_score, n_thresholds)

    for threshold in thresholds:
        system_predictions = (system_scores > threshold).int()

        calculator = SystemCalculator(
            dataset=dataset,
            system_labels=labels,
            system_predictions=system_predictions,
            system_scores=system_scores,
        )
        if range_based:
            precision = calculator.calculate_precision()
            recall = calculator.calculate_range_based_recall(range_metrics_alpha)
            f1 = calculator.calculate_f1(precision, recall)
        else:
            precision = calculator.calculate_precision()
            recall = calculator.calculate_recall()
            f1 = calculator.calculate_f1(precision, recall)

        # Update best thresholds where F1
        if f1.metric_system > system_f1:
            system_f1 = f1.metric_system
            best_threshold = threshold

    return best_threshold


def generate_scores(
    predictions: torch.Tensor,
    true_values: torch.Tensor,
    score_type: Literal["abs", "mse"] = "mse",
    post_process: bool = False,
    window_size_smooth: int = 5,
) -> torch.Tensor:
    """
    Generate anomaly scores by comparing predictions with true values.

    Args:
        predictions: Tensor of shape (n_samples, n_features) containing predictions
        true_values: Tensor of shape (n_samples, n_features) containing true values
        score_type: Type of score to use, either "mse" or "abs". Default is "mse".
        post_process: Whether to apply post-processing to the scores
        window_size_smooth: Window size for smoothing if post-processing is applied

    Returns:
        Tensor of shape (n_samples, n_features) containing anomaly scores
    """
    if score_type == "abs":
        scores = torch.abs(predictions - true_values)
    elif score_type == "mse":
        scores = torch.sqrt((predictions - true_values) ** 2)

    if post_process:
        scores = post_process_scores(scores, window_size_smooth)

    return scores


def post_process_scores(scores: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """
    Post process the scores by applying smoothing and standardization.

    This function performs two steps:
    1. Standardizes the scores using robust statistics (median and IQR) to normalize
       the scale across features
    2. Smooths the scores using a moving average

    Args:
        scores: Tensor of shape (n_samples, n_features) containing error values
        window_size: Size of the moving average window for smoothing

    Returns:
        Post processed scores using standardization and a moving average
    """
    scores = standarize_error_scores(scores)
    scores = smooth_scores(scores, window_size=window_size)
    return scores


def standarize_error_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Normalize error scores using robust statistics (median and IQR)
    to prevent any single sensor from dominating.

    Args:
        scores: Tensor of shape (n_samples, n_features) containing error values

    Returns:
        Normalized scores using median and IQR normalization
    """
    # Calculate median and IQR along time dimension (dim=0)
    medians = torch.median(scores, dim=0).values
    q75 = torch.quantile(scores, 0.75, dim=0)
    q25 = torch.quantile(scores, 0.25, dim=0)
    iqr = q75 - q25

    # Normalize using median and IQR
    normalized_scores = (scores - medians) / iqr

    return normalized_scores


def smooth_scores(scores: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Smooth scores using a moving average.

    Args:
        scores: Tensor of shape (n_samples, n_features) containing error values
        window_size: Size of the moving average window

    Returns:
        Smoothed scores using a moving average (n_samples, n_features)
    """
    # Pad the input to handle boundary effects
    pad_size = window_size - 1
    padded_scores = torch.nn.functional.pad(scores, (pad_size, 0), mode="replicate")
    return torch.nn.functional.avg_pool1d(
        padded_scores, kernel_size=window_size, stride=1
    )


def get_spot_predictions(
    train_score: torch.Tensor, test_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get anomaly predictions using the SPOT algorithm.

    SPOT (Statistical Process cOnTrol) is used to automatically determine
    thresholds for anomaly detection based on extreme value theory.

    Args:
        train_score: Tensor of shape (n_train_samples, n_features) with training scores
        test_score: Tensor of shape (n_test_samples, n_features) with test scores

    Returns:
        A tuple containing:
        - predictions: Binary tensor indicating anomalies (1) or normal points (0)
        - thresholds: The thresholds determined by SPOT for each feature
    """
    thresholds = []
    for i in range(train_score.shape[1]):
        s = SPOT(q=1e-3)
        s.fit(train_score[:, i].numpy(), test_score[:, i].numpy())
        s.initialize(level=0.95)
        ret = s.run(dynamic=False, with_alarm=False)
        threshold = torch.Tensor(ret["thresholds"]).mean()
        thresholds.append(threshold)
    thresholds = torch.stack(thresholds)
    predictions = test_score > thresholds
    predictions = predictions.int()
    return predictions, thresholds
