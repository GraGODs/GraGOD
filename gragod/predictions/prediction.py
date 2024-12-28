import torch

from gragod.metrics.calculator import MetricsCalculator
from gragod.metrics.models import SystemMetricsResult
from gragod.predictions.spot import SPOT
from gragod.types import Datasets


def get_threshold(
    dataset: Datasets, scores: torch.Tensor, labels: torch.Tensor, n_thresholds: int
) -> torch.Tensor:
    if labels.ndim == 0 or labels.shape[1] in [0, 1]:
        return get_threshold_system(dataset, scores, labels, n_thresholds)
    else:
        return get_threshold_per_class(dataset, scores, labels, n_thresholds)


def get_threshold_per_class(
    dataset: Datasets, scores: torch.Tensor, labels: torch.Tensor, n_thresholds: int
) -> torch.Tensor:
    """
    Gets the threshold for the scores for each time series.
    The best threshold is the one that maximizes the F1 score or
    as a default the maximum score in the training set.
    Args:
        scores: Tensor of shape (n_samples - window_size, n_features).
        labels: Tensor of shape (n_samples - window_size, n_features).
        n_thresholds: Number of thresholds to test.
    Returns:
        The best thresholds for each dimension (n_features,).
    """
    # Initial best thresholds with highest scores
    max_scores = best_thresholds = torch.max(scores, dim=0)[0]
    preds = scores > best_thresholds.unsqueeze(0)
    metrics = MetricsCalculator(
        dataset=dataset, labels=labels, predictions=preds, scores=scores
    )
    precision = metrics.calculate_precision()
    recall = metrics.calculate_recall()
    f1 = metrics.calculate_f1(precision, recall)

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

        metrics = MetricsCalculator(
            dataset=dataset, labels=labels, predictions=preds, scores=scores
        )
        precision = metrics.calculate_precision()
        recall = metrics.calculate_recall()
        f1 = metrics.calculate_f1(precision, recall)

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
    dataset: Datasets, scores: torch.Tensor, labels: torch.Tensor, n_thresholds: int
) -> torch.Tensor:
    """
    Get the threshold for the scores.
    The best threshold is the one that maximizes the F1 score or
    as a default the maximum score in the training set.
    Args:
        scores: Tensor of shape (n_samples - window_size, n_features).
        labels: Tensor of shape (n_samples - window_size, 1).
        n_thresholds: Number of thresholds to test.
    Returns:
        The best threshold for the system.
    """
    # here we only have system class so there will be only one threshold
    # Initial best thresholds with highest scores
    max_score = best_threshold = torch.max(scores)
    preds = scores > best_threshold
    metrics = MetricsCalculator(
        dataset=dataset, labels=labels, predictions=preds, scores=scores
    )
    precision = metrics.calculate_precision()
    recall = metrics.calculate_recall()
    f1 = metrics.calculate_f1(precision, recall)

    system_f1 = f1.metric_system

    thresholds = torch.linspace(0, max_score, n_thresholds)

    for threshold in thresholds:
        preds = (scores > threshold).float()

        metrics = MetricsCalculator(
            dataset=dataset, labels=labels, predictions=preds, scores=scores
        )
        precision = metrics.calculate_precision()
        recall = metrics.calculate_recall()
        f1 = metrics.calculate_f1(precision, recall)
        # Update best thresholds where F1
        if f1.metric_system > system_f1:
            system_f1 = f1.metric_system
            best_threshold = threshold

    return best_threshold


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
    Get threshold for anomaly detection.
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
