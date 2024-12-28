import tabulate


def generate_metrics_table(metrics: dict, only_system: bool = False) -> str:
    """Generate a table of metrics as a string."""

    # Define metric groups and their display names
    metric_groups = {}
    for metric in metrics.keys():
        if metric.endswith("_system"):
            metric_name = metric.replace("_system", "")
            metric_groups[metric_name] = metric_name.title()

    # Create headers
    if only_system:
        metrics_table = [["System"]]
    else:
        metrics_table = [["Metric", "Global", "Mean", "System"]]

    # Build table rows dynamically
    for metric_key, metric_name in metric_groups.items():
        if only_system:
            row = [
                f"{metrics.get(f'{metric_key}_system', '')}",
            ]
        else:
            row = [
                metric_name,
                f"{metrics.get(f'{metric_key}_global', '')}",
                f"{metrics.get(f'{metric_key}_mean', '')}",
                f"{metrics.get(f'{metric_key}_system', '')}",
            ]
        metrics_table.append(row)

    return tabulate.tabulate(metrics_table, headers="firstrow", tablefmt="grid")


def generate_metrics_per_class_table(metrics: dict) -> str:
    """Generate a table of per-class metrics as a string."""

    n_classes = 0
    metrics_per_class = {}
    for metric in metrics.keys():
        if metric.endswith("_per_class"):
            metrics_per_class[metric] = metrics[metric]
            n_classes = len(metrics_per_class[metric])

    if n_classes == 0:
        raise ValueError("No per-class metrics found")

    metrics_per_class_table = []
    for i in range(n_classes):
        table_i = []
        table_i.append(i)
        for metric in metrics_per_class.keys():
            table_i.append(metrics_per_class[metric][i])
        metrics_per_class_table.append(table_i)

    headers = ["Class"] + [
        key.replace("_per_class", "").title() for key in metrics_per_class.keys()
    ]

    return tabulate.tabulate(
        metrics_per_class_table,
        headers=headers,
        tablefmt="grid",
    )


def print_all_metrics(metrics: dict, message: str):
    print(message)
    if "precision_per_class" in metrics:
        metrics_table = generate_metrics_table(metrics)
        print(metrics_table)
        metrics_per_class_table = generate_metrics_per_class_table(metrics)
        print(metrics_per_class_table)
    else:
        metrics_table = generate_metrics_table(metrics, only_system=True)
        print(metrics_table)
