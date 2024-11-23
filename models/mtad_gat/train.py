import argparse

from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import CleanMethods, InterPolationMethods, ParamFileTypes
from gragod.training import load_params, load_training_data, set_seeds
from gragod.training.callbacks import get_training_callbacks
from gragod.training.trainer import TrainerPL
from gragod.types import cast_dataset
from models.mtad_gat.model import MTAD_GAT, MTAD_GAT_PLModule

RANDOM_SEED = 42


set_seeds(RANDOM_SEED)


def main(
    # Required arguments
    dataset_name: str,
    model_name: str,
    model_params: dict,
    params: dict,
    batch_size: int,
    n_epochs: int,
    init_lr: float,
    test_size: float,
    val_size: float,
    clean: str,
    interpolate_method: InterPolationMethods | None,
    shuffle: bool,
    target_dims: int | None,
    horizon: int,
    device: str,
    n_workers: int,
    log_dir: str,
    log_every_n_steps: int,
    ckpt_path: str | None = None,
    weight_decay: float = 1e-5,
    eps: float = 1e-8,
    betas: tuple[float, float] = (0.9, 0.999),
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
        horizon=horizon,
        labels=y_train,
        drop=clean == CleanMethods.DROP.value,
    )
    val_dataset = SlidingWindowDataset(
        X_val,
        window_size=window_size,
        horizon=horizon,
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

    logger = TensorBoardLogger(
        save_dir=log_dir, name=model_name, default_hp_metric=False
    )

    # Define callbacks
    callback_dict = get_training_callbacks(
        log_dir=logger.log_dir,
        model_name=model_name,
        monitor="Loss/val",
    )
    callbacks = list(callback_dict.values())

    trainer = TrainerPL(
        model=model,
        model_pl=MTAD_GAT_PLModule,
        model_params=params["model_params"],
        criterion={"forecast_criterion": nn.MSELoss(), "recon_criterion": nn.MSELoss()},
        batch_size=batch_size,
        n_epochs=n_epochs,
        init_lr=init_lr,
        device=device,
        log_dir=log_dir,
        logger=logger,
        callbacks=callbacks,
        checkpoint_cb=callback_dict["checkpoint"],
        target_dims=target_dims,
        log_every_n_steps=log_every_n_steps,
        weight_decay=weight_decay,
        eps=eps,
        betas=betas,
    )
    if ckpt_path:
        trainer.load(ckpt_path)

    trainer.fit(train_loader, val_loader, args_summary=args_summary)


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
