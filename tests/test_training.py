import pytest
from colorama import Fore, Style

from gragod.training import load_params, set_seeds
from gragod.types import Datasets, Models, ParamFileTypes
from models.train import train


@pytest.mark.parallel
def test_gcn_training():
    """
    Run 1 epoch of training for the GCN model for the given dataset.
    """

    dataset = Datasets.TELCO

    # Load parameters for GCN model
    gcn_params = load_params("models/gcn/params.yaml", file_type=ParamFileTypes.YAML)
    gcn_params["train_params"]["n_epochs"] = 2
    gcn_params["train_params"]["device"] = "cpu"

    # Set random seed for reproducibility
    set_seeds(42)

    # Run 1 epoch of training for GCN model
    train(
        model=Models.GCN,
        dataset=dataset,
        **gcn_params["train_params"],
        model_params=gcn_params["model_params"],
        params=gcn_params,
    )
    print(f"{Fore.GREEN}GCN training completed!{Style.RESET_ALL}")


@pytest.mark.parallel
def test_gdn_training():
    """
    Run 1 epoch of training for the GDN model for the given dataset.
    """

    dataset = Datasets.TELCO

    # Load parameters for GDN model
    gdn_params = load_params("models/gdn/params.yaml", file_type=ParamFileTypes.YAML)
    gdn_params["train_params"]["n_epochs"] = 2
    gdn_params["train_params"]["device"] = "cpu"

    # Set random seed for reproducibility
    set_seeds(42)

    # Run 1 epoch of training for GDN model
    train(
        model=Models.GDN,
        dataset=dataset,
        **gdn_params["train_params"],
        model_params=gdn_params["model_params"],
        params=gdn_params,
    )
    print(f"{Fore.GREEN}GDN training completed!{Style.RESET_ALL}")


@pytest.mark.parallel
def test_mtad_gat_training():
    """
    Run 1 epoch of training for the MTAD-GAT model for the given dataset.
    """

    dataset = Datasets.TELCO

    # Load parameters for MTAD-GAT model
    mtad_gat_params = load_params(
        "models/mtad_gat/params.yaml", file_type=ParamFileTypes.YAML
    )
    mtad_gat_params["train_params"]["n_epochs"] = 2
    mtad_gat_params["train_params"]["device"] = "cpu"
    # Set random seed for reproducibility
    set_seeds(42)

    # Run 1 epoch of training for MTAD-GAT model
    train(
        model=Models.MTAD_GAT,
        dataset=dataset,
        **mtad_gat_params["train_params"],
        model_params=mtad_gat_params["model_params"],
        params=mtad_gat_params,
    )
    print(f"{Fore.GREEN}MTAD-GAT training completed!{Style.RESET_ALL}")


@pytest.mark.parallel
def test_gru_training():
    """
    Run 1 epoch of training for the GRU model for the given dataset.
    """

    dataset = Datasets.TELCO

    # Load parameters for GRU model
    gru_params = load_params("models/gru/params.yaml", file_type=ParamFileTypes.YAML)
    gru_params["train_params"]["n_epochs"] = 2
    gru_params["train_params"]["device"] = "cpu"

    # Set random seed for reproducibility
    set_seeds(42)

    # Run 1 epoch of training for GRU model
    train(
        model=Models.GRU,
        dataset=dataset,
        **gru_params["train_params"],
        model_params=gru_params["model_params"],
        params=gru_params,
    )
    print(f"{Fore.GREEN}GRU training completed!{Style.RESET_ALL}")
