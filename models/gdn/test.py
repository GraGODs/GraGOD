import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[float, tuple[list[float], list[float], list[float]]]:
    """
    Test the model on the given dataloader.

    Args:
        model: The neural network model to test.
        dataloader: The DataLoader containing the test data.
        device: The device to run the model on.

    Returns:
        A tuple containing:
            - The average loss over the test set.
            - A tuple of three lists containing the predicted values,
              ground truth values, and labels for all test samples.
    """
    loss_func = nn.MSELoss(reduction="mean")

    test_loss_list = []

    test_predicted_list = torch.tensor([])
    test_ground_list = torch.tensor([])
    test_labels_list = torch.tensor([])

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [
            item.to(device).float()
            for item in [
                x.reshape(-1, x.shape[2], x.shape[1]),
                y.squeeze(1),
                labels,
                edge_index,
            ]
        ]

        with torch.no_grad():
            predicted = model(x).float().to(device)
            loss = loss_func(predicted, y)
            labels = labels.squeeze(1)
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(test_predicted_list) <= 0:
                test_predicted_list = predicted
                test_ground_list = y
                test_labels_list = labels
            else:
                test_predicted_list = torch.cat((test_predicted_list, predicted), dim=0)
                test_ground_list = torch.cat((test_ground_list, y), dim=0)
                test_labels_list = torch.cat((test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, (
        test_predicted_list.tolist(),
        test_ground_list.tolist(),
        test_labels_list.tolist(),
    )
