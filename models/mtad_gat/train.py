import argparse

from torch import nn
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import CleanMethods, InterPolationMethods, ParamFileTypes
from gragod.training import load_params, load_training_data, set_seeds
from gragod.types import cast_dataset
from models.mtad_gat.model import MTAD_GAT
from models.mtad_gat.trainer_pl import TrainerPL

RANDOM_SEED = 42


set_seeds(RANDOM_SEED)


def main(
    dataset_name: str,
    model_params: dict,
    n_epochs: int = 30,
    batch_size: int = 264,
    init_lr: float = 0.001,
    device: str = "mps",
    n_workers: int = 0,
    target_dims: int | None = None,
    log_dir: str = "output",
    log_every_n_steps: int = 1,
    ckpt_path: str | None = None,
    test_size: float = 0.1,
    val_size: float = 0.1,
    clean: CleanMethods = CleanMethods.NONE,
    interpolate_method: InterPolationMethods | None = None,
    shuffle: bool = True,
    horizon: int = 1,
    params: dict = {},
):
    dataset = cast_dataset(dataset_name)
    dataset_config = get_dataset_config(dataset=dataset)

    # Load data
    X_train, X_val, _, y_train, y_val, _ = load_training_data(
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        normalize=dataset_config.normalize,
        clean=clean == CleanMethods.INTERPOLATE.value,
        interpolate_method=interpolate_method,
    )

    # Create dataloaders
    window_size = model_params["window_size"]

    train_dataset = SlidingWindowDataset(
        X_train,
        window_size=window_size,
        labels=y_train,
        drop=clean == CleanMethods.DROP.value,
    )
    val_dataset = SlidingWindowDataset(
        X_val,
        window_size=window_size,
        labels=y_val,
        drop=clean == CleanMethods.DROP.value,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=shuffle,
        persistent_workers=n_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=False,
        persistent_workers=n_workers > 0,
    )

    # Create model
    n_features = X_train.shape[1]
    out_dim = X_train.shape[1]

    model = MTAD_GAT(
        n_features=n_features,
        out_dim=out_dim,
        **model_params,
    )
    args_summary = {
        "dataset": dataset,
        "model_params": model_params,
        "train_params": params["train_params"],
        "predictor_params": params["predictor_params"],
    }

    trainer = TrainerPL(
        model=model,
        model_params=params["model_params"],
        target_dims=target_dims,
        init_lr=init_lr,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        batch_size=batch_size,
        n_epochs=n_epochs,
        device=device,
        log_dir=log_dir,
        log_every_n_steps=log_every_n_steps,
    )
    if ckpt_path:
        trainer.load(ckpt_path)

    # Train model
    trainer.fit(train_loader, val_loader, args_summary=args_summary)

    input_example = next(iter(train_loader))[0]
    trainer.save_compiled_model(input_example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params_file", type=str, default="models/mtad_gat/params.yaml"
    )
    args = parser.parse_args()
    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    main(
        dataset_name=params["dataset"],
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
    )
