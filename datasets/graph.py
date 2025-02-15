import torch


def get_edge_index(
    X: torch.Tensor, device: str, path: str | None = None
) -> torch.Tensor:
    """
    Get the edge index of the graph.

    Args:
        X: The input data.
        device: The device to place the edge_index tensor on.
        path: The path to the edge_index file.

    Returns:
        The edge index of the graph.
    """
    if path:
        try:
            edge_index = torch.load(path)
            return edge_index
        except FileNotFoundError:
            print(f"Edge index file not found at {path}")

    print("Building fully connected edge index")
    return build_fully_connected_edge_index(X, device)


def build_fully_connected_edge_index(X: torch.Tensor, device: str) -> torch.Tensor:
    """
    Build a fully connected edge index for the graph.

    Args:
        X: The input data.
        device: The device to place the edge_index tensor on.

    Returns:
        The fully connected edge index of the graph.
    """
    edge_index = (
        torch.tensor(
            [[i, j] for i in range(X.shape[1]) for j in range(X.shape[1])],
            dtype=torch.long,  # edge_index must be long type
        )
        .t()
        .to(device)
    )

    return edge_index
