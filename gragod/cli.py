import json

import typer

app = typer.Typer()


@app.command()
def test_cli():
    typer.secho("GraGOD CLI is working!", fg=typer.colors.GREEN, bold=True)


@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")


@app.command()
def show_metrics(metric_path: str, per_class: bool = False):
    from pathlib import Path

    from gragod.metrics.visualization import (
        generate_metrics_per_class_table,
        generate_metrics_table,
    )

    path = Path(metric_path)
    if path.is_file():
        # Handle single file case
        with open(path, "r") as f:
            metric = json.load(f)
        typer.secho(f"Metrics for {path}", fg=typer.colors.GREEN, bold=True)
        typer.echo(generate_metrics_table(metric))
        if per_class:
            typer.echo(generate_metrics_per_class_table(metric))
    else:
        # Handle directory case
        metric_files = []
        for pattern in ["**/metric.json", "**/metrics_train.json"]:
            metric_files.extend(path.glob(pattern))

        if not metric_files:
            typer.secho(f"No metric files found in {path}", fg=typer.colors.RED)
            return

        for metric_file in sorted(metric_files):
            with open(metric_file, "r") as f:
                metric = json.load(f)
            typer.secho(
                f"\nMetrics for {metric_file}", fg=typer.colors.GREEN, bold=True
            )
            typer.echo(generate_metrics_table(metric))
            if per_class:
                typer.echo(generate_metrics_per_class_table(metric))


if __name__ == "__main__":
    app()
