import argparse
import os
from typing import Any, Dict

import optuna
import yaml

from gragod import ParamFileTypes
from gragod.training import load_params, set_seeds
from models.gdn.predict import main as predict_main
from models.gdn.train import main as train_main


def objective(
    trial: optuna.Trial, base_params: Dict[str, Any], study_name: str
) -> float:
    # Suggest hyperparameters
    model_params = {
        "window_size": trial.suggest_int("window_size", 10, 30),
        "embed_dim": trial.suggest_int("embed_dim", 32, 128),
        "out_layer_num": trial.suggest_int("out_layer_num", 1, 7),
        "out_layer_inter_dim": trial.suggest_int("out_layer_inter_dim", 128, 512),
        "topk": trial.suggest_int("topk", 3, 10),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }

    train_params = base_params["train_params"].copy()
    train_params.update(
        {
            "init_lr": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "betas": (
                trial.suggest_float("beta1", 0.8, 0.99),
                trial.suggest_float("beta2", 0.8, 0.999),
            ),
            "eps": trial.suggest_float("eps", 1e-8, 1e-6, log=True),
        }
    )

    # Update the base params with the trial suggestions
    trial_params = base_params.copy()
    trial_params["model_params"] = model_params
    trial_params["train_params"] = train_params

    # Set unique checkpoint folder for this trial
    train_params["log_dir"] = os.path.join(
        train_params["log_dir"], study_name, f"trial_{trial.number}"
    )
    trial_params["predictor_params"]["ckpt_folder"] = os.path.join(
        train_params["log_dir"], "gdn", "version_0"
    )

    # Train the model
    train_main(
        dataset_name=trial_params["dataset"],
        model_params=model_params,
        params=trial_params,
        **train_params,
    )

    # Run prediction and get metrics
    metrics = predict_main(
        dataset_name=trial_params["dataset"],
        model_params=model_params,
        params=trial_params,
        **train_params,
    )["metrics"]

    # Return negative VUS-ROC mean score (since Optuna minimizes)
    return -metrics["vus_roc_mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="models/gdn/params.yaml")
    args = parser.parse_args()

    # Load base parameters
    base_params = load_params(args.params_file, file_type=ParamFileTypes.YAML)
    set_seeds(base_params["env_params"]["random_seed"])

    # Create study
    study_name = base_params["optimization_params"]["study_name"]
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, base_params, study_name),
        n_trials=base_params["optimization_params"]["n_trials"],
    )

    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters
    best_params = base_params.copy()
    best_params["model_params"].update(
        {
            k: v
            for k, v in trial.params.items()
            if k
            in [
                "window_size",
                "embed_dim",
                "out_layer_num",
                "out_layer_inter_dim",
                "topk",
                "dropout",
            ]
        }
    )
    best_params["train_params"].update(
        {
            k: v
            for k, v in trial.params.items()
            if k in ["learning_rate", "weight_decay", "beta1", "beta2", "eps"]
        }
    )

    # Save best parameters to file
    output_file = os.path.join(
        base_params["train_params"]["log_dir"], f"{args.study_name}_best_params.yaml"
    )
    with open(output_file, "w") as f:
        yaml.dump(best_params, f)


if __name__ == "__main__":
    main()
