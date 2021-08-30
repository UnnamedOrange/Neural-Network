import torch
from matplotlib import pyplot as plt


def test(times, losses, net, is_gpu, dataset: torch.utils.data.TensorDataset, batch_size, num_workers: int, vis, vis_win: list, vis_options: dict):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if is_gpu:
        net = net.cuda()
    else:
        net = net.cpu()

    net.eval()
    loss = 0
    for (data, tag) in data_loader:
        if is_gpu:
            data = data.cuda()
            tag = tag.cuda()
        output = net(data)
        loss += net.loss_function(output, tag).detach().cpu()
    loss /= len(data_loader)
    losses.append(loss)

    if vis:
        vis_win[0] = vis.line(torch.tensor(losses),
                              torch.arange(len(losses)),
                              win=vis_win[0],
                              name='test',
                              opts=vis_options)

    net = net.cpu()
