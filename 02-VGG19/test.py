import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from os import PathLike


def test(net: nn.Module,  # With optimizer and loss_function.
         dataset: torch.utils.data.TensorDataset,
         index: int,
         batch_size: int = None,
         num_workers: int = 0,
         tb: SummaryWriter = None):

    if batch_size is None:
        batch_size = len(dataset)

    is_gpu = True
    try:
        next(net.parameters()).get_device()
    except RuntimeError:
        is_gpu = False

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net.eval()
    sum_loss = 0
    sum_correct = 0
    for (data, tag) in data_loader:
        if is_gpu:
            data = data.cuda()
            tag = tag.cuda()
        output = net(data)
        sum_loss += net.loss_function(output, tag).detach().cpu()
        pred = output.detach().max(1)[1]
        sum_correct += pred.eq(tag.view_as(pred)).sum().cpu()
    average_loss = sum_loss / len(dataset)
    accuracy = sum_correct / len(dataset)

    if tb:
        tb.add_scalar('Test Loss Trace per Epoch', average_loss, index)
        tb.add_scalar('Test Accuracy Trace per Epoch', accuracy, index)
