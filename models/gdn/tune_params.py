def get_tune_model_params(trial):
    return {
        "window_size": trial.suggest_int("window_size", 5, 250, step=25),
        "embed_dim": trial.suggest_int("embed_dim", 32, 128, step=16),
        "out_layer_num": trial.suggest_int("out_layer_num", 1, 7, step=1),
        "out_layer_inter_dim": trial.suggest_int(
            "out_layer_inter_dim", 128, 512, step=64
        ),
        "topk": trial.suggest_int("topk", 3, 12, step=3),
        "heads": trial.suggest_int("heads", 1, 5, step=1),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
    }
