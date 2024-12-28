def get_tune_model_params(trial) -> dict:
    shared_params = {
        "n_layers": 3,
        "hid_dim": trial.suggest_categorical("hid_dim", [32, 64]),
    }
    return {
        "window_size": trial.suggest_categorical(
            "window_size",
            [5, 25],
        ),
        "kernel_size": trial.suggest_categorical("kernel_size", [7, 9]),
        "use_gatv2": trial.suggest_categorical("use_gatv2", [True, False]),
        "feat_gat_embed_dim": None,
        "time_gat_embed_dim": None,
        "recon_n_layers": shared_params["n_layers"],
        "recon_hid_dim": shared_params["hid_dim"],
        "forecast_n_layers": shared_params["n_layers"],
        "forecast_hid_dim": shared_params["hid_dim"],
        "gru_n_layers": trial.suggest_categorical("gru_n_layers", [3, 5, 7]),
        "gru_hid_dim": trial.suggest_categorical("gru_hid_dim", [32, 64]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3, step=0.1),
        "alpha": 0.02,
    }
