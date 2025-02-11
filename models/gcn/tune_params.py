import optuna


def get_tune_model_params(trial: optuna.Trial) -> dict:
    return {
        "window_size": trial.suggest_int("window_size", 5, 255, step=25),
        "n_layers": trial.suggest_int("n_layers", 1, 21, step=2),
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, step=16),
        "K": trial.suggest_int("K", 1, 10, step=1),
    }
