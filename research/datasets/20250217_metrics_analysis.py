# %%
from gragod.metrics.calculator import MetricsCalculator
from gragod.types import Datasets
import torch

import matplotlib.pyplot as plt

# %%
synthetic_data = torch.zeros(10000, 1)
synthetic_data[1000:2000] = 1
for i in range(3000, 10000, 1000):
    synthetic_data[i : i + 10] = 1

# %%
prediction_1 = torch.ones_like(synthetic_data)
prediction_1[1000] = 0
prediction_1[2000] = 0

prediction_2 = torch.ones_like(synthetic_data)
prediction_2[1000] = 0
for i in range(1000, 2000, 10):
    prediction_2[i] = 0
prediction_2[2000] = 0

prediction_3 = synthetic_data.clone()
prediction_3[:2000] = 0

prediction_4 = synthetic_data.clone()
prediction_4[2000:] = 0

prediction_5 = synthetic_data.clone()
prediction_5[:2000] = 0
prediction_5[1500] = 1

prediction_6 = synthetic_data.clone()
prediction_6[:2000] = 0
prediction_6[1250:1750] = 1


# %%
def plot_synthetic_data(data):
    plt.figure(figsize=(15, 5))
    plt.plot(data, label="Synthetic Data")
    plt.title("Synthetic Data Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_synthetic_data(synthetic_data)


# %%
def calculate_metrics(
    labels, predictions, scores, alpha_precision=0.0, alpha_recall=1.0
):
    metrics = MetricsCalculator(
        dataset=Datasets.TELCO,
        labels=labels,
        predictions=predictions,
        scores=scores,
    )

    plot_synthetic_data(labels)
    plot_synthetic_data(predictions)

    precision = metrics.calculate_precision()
    recall = metrics.calculate_recall()
    f1 = metrics.calculate_f1(precision, recall)
    range_based_precision = metrics.calculate_range_based_precision(
        alpha=alpha_precision
    )
    range_based_recall = metrics.calculate_range_based_recall(alpha=alpha_recall)
    range_based_f1 = metrics.calculate_range_based_f1(
        range_based_precision, range_based_recall
    )

    custom_f1 = metrics.calculate_f1(precision, range_based_recall)

    print(f"Precision: {precision.metric_system}")
    print(f"Recall: {recall.metric_system}")
    print(f"F1: {f1.metric_system}")
    print(f"Range-based Precision: {range_based_precision.metric_system}")
    print(f"Range-based Recall: {range_based_recall.metric_system}")
    print(f"Range-based F1: {range_based_f1.metric_system}")
    print(f"Custom F1: {custom_f1.metric_system}")


# %%
print(f"ONE PREDICTION:")
calculate_metrics(
    synthetic_data,
    torch.ones_like(synthetic_data),
    torch.ones_like(synthetic_data),
)
# %%
# print(f"ZERO PREDICTION:")
calculate_metrics(
    synthetic_data,
    prediction_1,
    torch.zeros_like(synthetic_data),
    alpha_recall=0.5,
)
calculate_metrics(
    synthetic_data,
    prediction_2,
    torch.zeros_like(synthetic_data),
    alpha_recall=0.5,
)

# %%
calculate_metrics(
    synthetic_data,
    prediction_3,
    torch.zeros_like(synthetic_data),
    alpha_recall=0.5,
)

calculate_metrics(
    synthetic_data,
    prediction_4,
    torch.zeros_like(synthetic_data),
    alpha_recall=0.5,
)


 %%
calculate_metrics(
    synthetic_data,
    prediction_5,
    torch.zeros_like(synthetic_data),
    alpha_recall=0.5,
)
calculate_metrics(
    synthetic_data,
    prediction_6,
    torch.zeros_like(synthetic_data),
    alpha_recall=0.5,
)

# %%
