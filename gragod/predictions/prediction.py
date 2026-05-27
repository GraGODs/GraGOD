from typing import Literal

import torch


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
