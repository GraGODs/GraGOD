import argparse
import os
import pickle
from pathlib import Path
from time import time
from typing import Callable, Tuple

import optuna
import torch
import yaml

from gragod import ParamFileTypes
from gragod.training import load_params, set_seeds
from gragod.types import MODEL_NAMES

RANDOM_SEED = 42


def save_study(study, log_dir):
    """Save study to disk."""
    study_path = os.path.join(log_dir, "study.pkl")
    with open(study_path, "wb") as fout:
        pickle.dump(study, fout)


def load_model_functions(
    model_name: MODEL_NAMES,
) -> Tuple[Callable, Callable, Callable]:
    if model_name == "gru":
        from models.gru.predict import main as predict_gru
        from models.gru.train import main as train_gru
        from models.gru.tune_params import get_tune_model_params

        return train_gru, predict_gru, get_tune_model_params
    elif model_name == "gcn":
        from models.gcn.predict import main as predict_gcn
        from models.gcn.train import main as train_gcn
        from models.gcn.tune_params import get_tune_model_params

        return train_gcn, predict_gcn, get_tune_model_params
    elif model_name == "gdn":
        from models.gdn.predict import main as predict_gdn
        from models.gdn.train import main as train_gdn
        from models.gdn.tune_params import get_tune_model_params

        return train_gdn, predict_gdn, get_tune_model_params
    elif model_name == "mtad_gat":
        from models.mtad_gat.predict import main as predict_mtad_gat
        from models.mtad_gat.train import main as train_mtad_gat
        from models.mtad_gat.tune_params import get_tune_model_params

        return train_mtad_gat, predict_mtad_gat, get_tune_model_params


def objective(
    model: MODEL_NAMES,
    trial: optuna.Trial,
    params: dict,
    optimization_metric: str,
):
    """Optuna objective function."""
    start_time = time()
    print(f"Trial number: {trial.number}")

    train_func, predict_func, get_tune_params = load_model_functions(model)

    # Get trial hyperparameters
    model_params = get_tune_params(trial)

    trainer = train_func(
        dataset_name=params["dataset"],
        **params["train_params"],
        model_params=model_params,
        params=params,
    )

    ckpt_path = os.path.join(
        trainer.logger.log_dir, params["train_params"]["model_name"] + ".ckpt"
    )

    params["predictor_params"]["ckpt_folder"] = os.path.join(trainer.logger.log_dir)

    predictions_dict = predict_func(
        dataset_name=params["dataset"],
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
    del model
    del trainer
    torch.cuda.empty_cache()

    end_time = time()
    print(f"Trial {trial.number} completed in {end_time - start_time:.2f} seconds")

    # Save study after each trial
    save_study(trial.study, params["train_params"]["log_dir"])

    return predictions_dict["val"]["metrics"][optimization_metric]


def main(
    model: MODEL_NAMES,
    params_file: str,
    optimization_metric: str,
):
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

    # Run optimization
    study.optimize(
        lambda trial: objective(
            model=model,
            trial=trial,
            params=params,
            optimization_metric=optimization_metric,
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
        type=str,
        default="gru",
        help="Model to tune",
    )  # one of gru, gcn, gdn, mtad_gat
    parser.add_argument(
        "--params_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--optimization_metric",
        type=str,
        default="vus_roc_system",
    )
    args = parser.parse_args()

    if args.params_file is None:
        args.params_file = f"models/{args.model}/params.yaml"

    main(args.model, args.params_file, args.optimization_metric)
