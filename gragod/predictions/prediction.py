import torch

from gragod.metrics import MetricsCalculator
from gragod.predictions.spot import SPOT


def get_threshold(scores: torch.Tensor, labels: torch.Tensor, n_thresholds: int):
    """
    Get the threshold for the scores.
    The best threshold is the one that maximizes the F1 score or
    as a default the maximum score in the training set.

    Args:
        scores: Tensor of shape (n_samples - window_size,).
        labels: Tensor of shape (n_samples - window_size,).
        n_thresholds: Number of thresholds to test.

    Returns:
        The best thresholds for each dimension.
    """
    # Initial best thresholds with highest scores
    max_scores = best_thresholds = torch.max(scores, dim=0)[0]
    preds = scores > best_thresholds.unsqueeze(0)
    metrics = MetricsCalculator(preds, labels, scores)
    precision = metrics.calculate_precision()
    recall = metrics.calculate_recall()
    best_f1s = metrics.calculate_f1(precision, recall).metric_per_class

    thresholds = torch.stack(
        [torch.linspace(0, max_score, n_thresholds) for max_score in max_scores],
        dim=1,
    )

    for threshold in thresholds:
        preds = (scores > threshold.unsqueeze(0)).float()

        metrics = MetricsCalculator(preds, labels, scores)
        precision = metrics.calculate_precision()
        recall = metrics.calculate_recall()
        f1 = metrics.calculate_f1(precision, recall)

        # Update best thresholds where F1 improved
        improved = f1.metric_per_class > best_f1s
        best_f1s[improved] = f1.metric_per_class[improved]
        best_thresholds[improved] = threshold[improved]

    return best_thresholds


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
