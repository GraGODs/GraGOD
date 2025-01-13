import argparse
import importlib
import os
from pathlib import Path
from time import time
from typing import Callable, Dict

import optuna
import torch
import yaml

from gragod import ParamFileTypes
from gragod.training import load_params, set_seeds
from gragod.types import Datasets, Models
from models.predict import predict
from models.train import train

RANDOM_SEED = 42

OPTIMIZATION_SPLIT = {Datasets.TELCO: "train", Datasets.SWAT: "val"}


def load_model_functions(
    model_name: Models,
) -> Callable:
    """Load training, prediction and parameter tuning functions for a model.

    Args:
        model_name: Name of the model to load functions for

    Returns:
        Tuple containing training, prediction and parameter tuning functions
    """

    module = importlib.import_module(f"models.{model_name.value}.tune_params")
    get_tune_model_params = getattr(module, "get_tune_model_params")

    return get_tune_model_params


def objective(
    model_name: Models,
    dataset: Datasets,
    get_tune_params: Callable,
    trial: optuna.Trial,
    params: Dict,
) -> float:
    """Optuna objective function for hyperparameter optimization.

    Args:
        predict_func: Prediction function for the model
        get_tune_params: Function to get the model parameters
        trial: Current Optuna trial
        params: Dictionary containing model parameters

    Returns:
        Value of the optimization metric for current trial
    """
    start_time = time()
    print(f"Trial number: {trial.number}")

    # Get trial hyperparameters
    model_params = get_tune_params(trial)

    trainer = train(
        model=model_name,
        dataset=dataset,
        **params["train_params"],
        model_params=model_params,
        params=params,
    )

    ckpt_path = os.path.join(
        trainer.logger.log_dir, params["train_params"]["model_name"] + ".ckpt"
    )

    params["predictor_params"]["ckpt_folder"] = trainer.logger.log_dir

    predictions_dict = predict(
        model=model_name,
        dataset=dataset,
        **params["train_params"],
        model_params=model_params,
        params=params,
        ckpt_path=ckpt_path,
    )

    # Log metrics to tensorboard
    for split in ["train", "val", "test"]:
        if split in predictions_dict.keys():
            for metric_name, metric_value in predictions_dict[split]["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    trainer.logger.experiment.add_scalar(
                        f"{split}_metrics/{metric_name}", metric_value, trial.number
                    )
    # Deallocate memory
    del trainer
    torch.cuda.empty_cache()

    end_time = time()
    print(f"Trial {trial.number} completed in {end_time - start_time:.2f} seconds")

    metric = params["optimization_params"]["metric"]

    return predictions_dict[OPTIMIZATION_SPLIT[dataset]]["metrics"][metric]


def main(
    model: Models,
    dataset: Datasets,
    params_file: str,
) -> None:
    """Main function to run hyperparameter optimization.

    Args:
        model: Name of the model to optimize
        params_file: Path to parameter file
    """
    torch.set_float32_matmul_precision("medium")
    set_seeds(RANDOM_SEED)

    # Load parameters
    params = load_params(params_file, file_type=ParamFileTypes.YAML)

    study_name = params["optimization_params"]["study_name"]

    # Setup logging
    log_dir = Path(params["train_params"]["log_dir"]) / study_name
    params["train_params"]["log_dir"] = str(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create new study if none exists
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    get_tune_params = load_model_functions(model)

    # Run optimization
    study.optimize(
        lambda trial: objective(
            model_name=model,
            dataset=dataset,
            get_tune_params=get_tune_params,
            trial=trial,
            params=params,
        ),
        n_trials=params["optimization_params"]["n_trials"],
    )
    # Save results
    best_params = study.best_params
    best_value = study.best_value

    # Save the best parameters
    output_file = log_dir / "best_params.yaml"
    with open(output_file, "w") as f:
        yaml.dump({"best_params": best_params, "best_value": best_value}, f)

    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Models,
        help=f"Model to tune [{', '.join(model.value for model in Models)}]",
    )
    parser.add_argument(
        "--dataset",
        type=Datasets,
        help=f"Dataset to use [{', '.join(dataset.value for dataset in Datasets)}]",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default=None,
        help="Path to parameter file",
    )
    args = parser.parse_args()

    if args.params_file is None:
        args.params_file = f"models/{args.model.value}/params.yaml"
    if args.dataset is None:
        raise ValueError("Dataset is required")
    if args.model is None:
        raise ValueError("Model is required")

    # Cast the model string to enum
    main(args.model, args.dataset, args.params_file)
