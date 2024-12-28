import torch
from pydantic import BaseModel, ConfigDict, Field


class MetricsResult(BaseModel):
    """
    Container for metric calculation results with validation.

    Per class metrics are metric calculated for each class independently.

    Mean metric is the mean of the per-class metrics.

    Global metric is the metric calculated across all classes. It's like flattening the
    tensor and calculating the metric.

    System metric is the metric calculated for the system, where the label/prediction
    are 1 if any of the labels/predictions is 1 for any variable, and 0 otherwise.

    Attributes:
        metric_global: Global metric across all classes.
        metric_mean: Mean metric across classes.
        metric_per_class: Per-class metrics,
        metric_system: System-level metric.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metric_global: float | None = Field(
        ...,
        description="Global metric across all classes",
        ge=0,
        le=1,
    )
    metric_mean: float = Field(
        ...,
        description="Mean metric across classes",
        ge=0,
        le=1,
    )
    metric_per_class: torch.Tensor = Field(..., description="Per-class metrics")
    metric_system: float = Field(
        ...,
        description="System-level metric",
        ge=0,
        le=1,
    )
    round_digits: int = Field(
        default=4, description="Number of decimal places to round to", exclude=True
    )

    def model_post_init(self, _context):
        if self.metric_global is not None:
            self.metric_global = round(self.metric_global, self.round_digits)
        self.metric_mean = round(self.metric_mean, self.round_digits)
        self.metric_system = round(self.metric_system, self.round_digits)

    def model_dump(self, metric_name: str, *args, **kwargs) -> dict:
        """Convert to dictionary with tensor conversion."""
        d = super().model_dump(*args, **kwargs)

        # Convert tensor to list for serialization
        d["metric_per_class"] = [round(x, 4) for x in self.metric_per_class.tolist()]

        d = {k.replace("metric", metric_name): v for k, v in d.items()}

        return d


class SystemMetricsResult(BaseModel):
    """
    System metric is the metric calculated for the system, where the label/prediction
    are 1 if any of the labels/predictions is 1 for any variable, and 0 otherwise.

    This class is used when the given labels do not have per-class information.

    Attributes:
        metric_system: System-level metric
    """

    metric_system: float = Field(
        ...,
        description="System-level metric",
        ge=0,
        le=1,
    )

    round_digits: int = Field(
        default=4, description="Number of decimal places to round to", exclude=True
    )

    def model_post_init(self, _context):
        self.metric_system = round(self.metric_system, self.round_digits)

    def model_dump(self, metric_name: str, *args, **kwargs) -> dict:
        """Convert to dictionary with tensor conversion."""
        d = super().model_dump(*args, **kwargs)

        d = {k.replace("metric", metric_name): v for k, v in d.items()}

        return d
