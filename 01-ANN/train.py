import math
import torch


def train(net, is_gpu, dataset: torch.utils.data.TensorDataset, batch_size, num_workers: int, n_epoch: int, vis, vis_win: list, vis_options: dict, on_test=None):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_batch = n_epoch * math.ceil(len(dataset) / batch_size)
    losses = torch.zeros(size=(n_batch,))
    indexes = torch.arange(n_batch)
    batch_index = 0

    for e in range(n_epoch):
        sum_loss = 0
        for i, (data, tag) in enumerate(data_loader):
            if is_gpu:
                net = net.cuda()
            else:
                net = net.cpu()
            net.train()
            net.optimizer.zero_grad()

            if is_gpu:
                data = data.cuda()
                tag = tag.cuda()
            output = net(data)
            loss = net.loss_function(output, tag)
            loss.backward()
            net.optimizer.step()

            crt_loss = loss.detach().cpu().item()
            sum_loss += crt_loss
            average_loss = crt_loss / len(data)
            losses[batch_index] = average_loss
            batch_index += 1
            if vis:
                vis_win[0] = vis.line(losses[:batch_index], indexes[:batch_index],
                                      win=vis_win[0], name='loss_per_batch',
                                      opts=vis_options)

            if on_test:
                on_test()

    net = net.cpu()
