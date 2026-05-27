import pytest

from gragod.training import load_params, set_seeds
from gragod.types import Datasets, Models, ParamFileTypes
from models.predict import predict


@pytest.mark.parallel
def test_gcn_predictions():
    """
    Test predictions for the GCN model.
    """
    dataset = Datasets.TELCO

    # Load parameters for GCN model
    gcn_params = load_params("models/gcn/params.yaml", file_type=ParamFileTypes.YAML)
    gcn_params["train_params"]["device"] = "cpu"
    # Set random seed for reproducibility
    set_seeds(42)

    # Run predictions for GCN model
    result = predict(  # noqa: F841
        model=Models.GCN,
        dataset=dataset,
        model_params=gcn_params["model_params"],
        params=gcn_params,
        **gcn_params["train_params"],
    )

    # Add assertions to verify the prediction result
    assert result is not None, "GCN prediction result should not be None"


@pytest.mark.parallel
def test_gdn_predictions():
    """
    Test predictions for the GDN model.
    """
    dataset = Datasets.TELCO

    # Load parameters for GDN model
    gdn_params = load_params("models/gdn/params.yaml", file_type=ParamFileTypes.YAML)
    gdn_params["train_params"]["device"] = "cpu"

    # Set random seed for reproducibility
    set_seeds(42)

    # Run predictions for GDN model
    result = predict(  # noqa: F841
        model=Models.GDN,
        dataset=dataset,
        model_params=gdn_params["model_params"],
        params=gdn_params,
        **gdn_params["train_params"],
    )

    # Add assertions to verify the prediction result
    assert result is not None, "GDN prediction result should not be None"


@pytest.mark.parallel
def test_mtad_gat_predictions():
    """
    Test predictions for the MTAD-GAT model.
    """
    dataset = Datasets.TELCO

    # Load parameters for MTAD-GAT model
    mtad_gat_params = load_params(
        "models/mtad_gat/params.yaml", file_type=ParamFileTypes.YAML
    )
    mtad_gat_params["train_params"]["device"] = "cpu"

    # Set random seed for reproducibility
    set_seeds(42)

    # Run predictions for MTAD-GAT model
    result = predict(
        model=Models.MTAD_GAT,
        dataset=dataset,
        model_params=mtad_gat_params["model_params"],
        params=mtad_gat_params,
        **mtad_gat_params["train_params"],
    )

    # Add assertions to verify the prediction result
    assert result is not None, "MTAD-GAT prediction result should not be None"


@pytest.mark.parallel
def test_gru_predictions():
    """
    Test predictions for the GRU model.
    """
    dataset = Datasets.TELCO

    # Load parameters for GRU model
    gru_params = load_params("models/gru/params.yaml", file_type=ParamFileTypes.YAML)
    gru_params["train_params"]["device"] = "cpu"

    # Set random seed for reproducibility
    set_seeds(42)

    # Run predictions for GRU model
    result = predict(  # noqa: F841
        model=Models.GRU,
        dataset=dataset,
        **gru_params["train_params"],
        model_params=gru_params["model_params"],
        params=gru_params,
    )
    assert result is not None, "GRU prediction result should not be None"
