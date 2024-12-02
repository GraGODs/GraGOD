import pytest

from gragod.training import load_params, set_seeds
from gragod.types import Datasets, ParamFileTypes
from models.gcn.predict import main as predict_gcn
from models.gdn.predict import main as predict_gdn
from models.mtad_gat.predict import main as predict_mtad_gat


@pytest.mark.parallel
def test_gcn_predictions():
    """
    Test predictions for the GCN model.
    """
    dataset = Datasets.TELCO

    # Load parameters for GCN model
    gcn_params = load_params("models/gcn/params.yaml", file_type=ParamFileTypes.YAML)

    # Set random seed for reproducibility
    set_seeds(42)

    # Run predictions for GCN model
    result = predict_gcn(  # noqa: F841
        dataset_name=dataset.value,
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

    # Set random seed for reproducibility
    set_seeds(42)

    # Run predictions for GDN model
    result = predict_gdn(  # noqa: F841
        dataset_name=dataset.value,
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

    # Set random seed for reproducibility
    set_seeds(42)

    # Run predictions for MTAD-GAT model
    result = predict_mtad_gat(
        dataset_name=dataset.value,
        model_params=mtad_gat_params["model_params"],
        params=mtad_gat_params,
        **mtad_gat_params["train_params"],
    )

    # Add assertions to verify the prediction result
    assert result is not None, "MTAD-GAT prediction result should not be None"
