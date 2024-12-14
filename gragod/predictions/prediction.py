import torch

from gragod.metrics import MetricsCalculator, SystemMetricsResult
from gragod.predictions.spot import SPOT


def get_threshold(
    scores: torch.Tensor, labels: torch.Tensor, n_thresholds: int
) -> torch.Tensor:
    if labels.shape[1] > 1:
        return get_threshold_per_class(scores, labels, n_thresholds)
    else:
        return get_threshold_system(scores, labels, n_thresholds)


def get_threshold_per_class(
    scores: torch.Tensor, labels: torch.Tensor, n_thresholds: int
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
    metrics = MetricsCalculator(labels, preds, scores)
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

        metrics = MetricsCalculator(labels, preds, scores)
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
    scores: torch.Tensor, labels: torch.Tensor, n_thresholds: int
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
    metrics = MetricsCalculator(labels, preds, scores)
    precision = metrics.calculate_precision()
    recall = metrics.calculate_recall()
    f1 = metrics.calculate_f1(precision, recall)

    system_f1 = f1.metric_system

    thresholds = torch.linspace(0, max_score, n_thresholds)

    for threshold in thresholds:
        preds = (scores > threshold).float()

        metrics = MetricsCalculator(labels, preds, scores)
        precision = metrics.calculate_precision()
        recall = metrics.calculate_recall()
        f1 = metrics.calculate_f1(precision, recall)
        # Update best thresholds where F1
        if f1.metric_system > system_f1:
            system_f1 = f1.metric_system
            best_threshold = threshold

    return best_threshold


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
