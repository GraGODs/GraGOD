import json

import typer

from gragod.metrics import generate_metrics_per_class_table, generate_metrics_table

app = typer.Typer()


@app.command()
def test_cli():
    typer.secho("GraGOD CLI is working!", fg=typer.colors.GREEN, bold=True)


@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")


@app.command()
def show_metrics(metric_path: str, per_class: bool = False):
    with open(metric_path, "r") as f:
        metric = json.load(f)
    typer.echo(generate_metrics_table(metric))
    if per_class:
        typer.echo(generate_metrics_per_class_table(metric))


if __name__ == "__main__":
    app()
