import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
import os
from os import PathLike


def train(net: nn.Module,  # With optimizer and loss_function.
          dataset: torch.utils.data.TensorDataset,
          is_gpu: bool = torch.cuda.is_available(),
          batch_size: int = None,
          num_workers: int = 0,
          n_epoch: int = None,
          model_file: PathLike = 'latest_model.pt',
          tb: SummaryWriter = None,
          on_test=None):

    if batch_size is None:
        batch_size = len(dataset)

    if n_epoch is None:
        n_epoch = 1000000000

    if model_file and os.path.exists(model_file):
        try:
            net.load_state_dict(torch.load(model_file))
            print("[Info] Model loaded.")
        except:
            print("[Warning] Fail to load model. Skip loading.")
    else:
        print("[Info] No model to load.")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if is_gpu:
        net = net.cuda()
    else:
        net = net.cpu()

    try:
        batch_index = 0
        net.optimizer.state = defaultdict(dict)
        for e in range(n_epoch):
            sum_loss = 0
            sum_correct = 0
            for i, (data, tag) in enumerate(data_loader):
                net.train()
                net.optimizer.zero_grad()

                if is_gpu:
                    data = data.cuda()
                    tag = tag.cuda()
                output = net(data)
                loss = net.loss_function(output, tag)
                loss.backward()
                net.optimizer.step()

                crt_loss = loss.detach().cpu()
                sum_loss += crt_loss

                pred = output.detach().max(1)[1]
                sum_correct += pred.eq(tag.view_as(pred)).sum().cpu()

                batch_index += 1

            average_loss = sum_loss / len(dataset)
            accuracy = sum_correct / len(dataset)
            if tb:
                tb.add_scalar('Train Loss Trace per Epoch', average_loss, e)
                tb.add_scalar('Train Accuracy Trace per Epoch', accuracy, e)

            if on_test:
                on_test()
    except KeyboardInterrupt:
        pass

    net = net.cpu()
    torch.save(net.state_dict(), model_file)
    print('Model saved.')
