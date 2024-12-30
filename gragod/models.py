from torch.nn import Module

from gragod.training.trainer import PLBaseModule
from gragod.types import Models


def get_model_and_module(model: Models) -> tuple[type[Module], type[PLBaseModule]]:
    """Get the model and corresponding PyTorch Lightning module classes.

    Args:
        model: The model type to get classes for

    Returns:
        Tuple containing the model class and its corresponding Lightning module class
    """
    if model == Models.GRU:
        from models.gru.model import GRU_PLModule, GRUModel

        return GRUModel, GRU_PLModule
    elif model == Models.GCN:
        from models.gcn.model import GCN, GCN_PLModule

        return GCN, GCN_PLModule
    elif model == Models.GDN:
        from models.gdn.model import GDN, GDN_PLModule

        return GDN, GDN_PLModule
    elif model == Models.MTAD_GAT:
        from models.mtad_gat.model import MTAD_GAT, MTAD_GAT_PLModule

        return MTAD_GAT, MTAD_GAT_PLModule
