import tabulate


def generate_metrics_table(metrics: dict) -> str:
    """Generate a table of metrics as a string."""
    metrics_table = [
        ["Global", "Mean", "System"],
        [
            "Precision",
            f"{metrics['precision_global']}",
            f"{metrics['precision_mean']}",
            f"{metrics['precision_system']}",
        ],
        [
            "Recall",
            f"{metrics['recall_global']}",
            f"{metrics['recall_mean']}",
            f"{metrics['recall_system']}",
        ],
        [
            "F1",
            f"{metrics['f1_global']}",
            f"{metrics['f1_mean']}",
            f"{metrics['f1_system']}",
        ],
        [
            "Range-based Precision",
            f"{metrics['range_based_precision_global']}",
            f"{metrics['range_based_precision_mean']}",
            f"{metrics['range_based_precision_system']}",
        ],
        [
            "Range-based Recall",
            f"{metrics['range_based_recall_global']}",
            f"{metrics['range_based_recall_mean']}",
            f"{metrics['range_based_recall_system']}",
        ],
        [
            "VUS-ROC",
            f"{metrics['vus_roc_global']}",
            f"{metrics['vus_roc_mean']}",
            f"{metrics['vus_roc_system']}",
        ],
    ]
    return tabulate.tabulate(metrics_table, headers="firstrow", tablefmt="grid")


def generate_metrics_per_class_table(metrics: dict) -> str:
    """Generate a table of per-class metrics as a string."""
    precision = metrics["precision_per_class"]
    recall = metrics["recall_per_class"]
    f1 = metrics["f1_per_class"]
    range_based_precision = metrics["range_based_precision_per_class"]
    range_based_recall = metrics["range_based_recall_per_class"]
    vus_roc = metrics["vus_roc_per_class"]
    metrics_per_class_table = [
        [
            i,
            precision[i],
            recall[i],
            f1[i],
            range_based_precision[i],
            range_based_recall[i],
            vus_roc[i],
        ]
        for i in range(len(precision))
    ]

    return tabulate.tabulate(
        metrics_per_class_table,
        headers=[
            "Class",
            "Precision",
            "Recall",
            "F1",
            "Range-based Precision",
            "Range-based Recall",
            "VUS-ROC",
        ],
        tablefmt="grid",
    )


def generate_system_metrics_table(metrics: dict) -> str:
    """Generate a table of system metrics as a string."""
    metrics_table = [
        ["System"],
        [
            "Precision",
            f"{metrics['precision_system']}",
        ],
        [
            "Recall",
            f"{metrics['recall_system']}",
        ],
        [
            "F1",
            f"{metrics['f1_system']}",
        ],
        [
            "Range-based Precision",
            f"{metrics['range_based_precision_system']}",
        ],
        [
            "Range-based Recall",
            f"{metrics['range_based_recall_system']}",
        ],
        [
            "VUS-ROC",
            f"{metrics['vus_roc_system']}",
        ],
    ]
    return tabulate.tabulate(metrics_table, headers="firstrow", tablefmt="grid")


def print_all_metrics(metrics: dict, message: str):
    print(message)
    if "precision_per_class" in metrics:
        metrics_table = generate_metrics_table(metrics)
        print(metrics_table)
        metrics_per_class_table = generate_metrics_per_class_table(metrics)
        print(metrics_per_class_table)
    else:
        metrics_table = generate_system_metrics_table(metrics)
        print(metrics_table)
