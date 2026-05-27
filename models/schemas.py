from typing import Any, TypedDict

import torch


class DatasetPredictOutput(TypedDict):
    output: Any
    predictions: torch.Tensor | None
    labels: torch.Tensor
    scores: torch.Tensor
    data: torch.Tensor
    thresholds: torch.Tensor
    metrics: dict | None


class PredictOutput(TypedDict):
    train: DatasetPredictOutput
    val: DatasetPredictOutput
    test: DatasetPredictOutput
