def get_tune_model_params(trial) -> dict:
    return {
        "window_size": trial.suggest_categorical("window_size", [64, 128, 256, 512]),
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256]),
        "n_layers": trial.suggest_categorical("n_layers", [1, 3, 5, 7]),
        "bidirectional": True,
        "rnn_dropout": trial.suggest_categorical("rnn_dropout", [0.1, 0.2, 0.3]),
        "fc_dropout": trial.suggest_categorical("fc_dropout", [0.1, 0.2, 0.3]),
    }
