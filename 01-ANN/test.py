import torch
from matplotlib import pyplot as plt


def test(net, is_gpu, dataset: torch.utils.data.TensorDataset, batch_size, num_workers: int, vis, vis_win: list, vis_options: dict):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if is_gpu:
        net = net.cuda()
    else:
        net = net.cpu()

    net.eval()

    for (data, tag) in data_loader:
        if is_gpu:
            data = data.cuda()
            tag = tag.cuda()
        output = net(data)

        if vis:
            plt.close()
            plt.plot(data.detach().cpu(),
                     tag.detach().cpu(), '.', markersize=1)
            plt.plot(data.detach().cpu(),
                     output.detach().cpu(), '.', markersize=1)

            vis_win[0] = vis.matplot(plt, win=vis_win[0], opts=vis_options)

            # if not vis_win[0]:
            #     vis_win[0] = vis.scatter(X=torch.cat((data, output), 1).detach().cpu(),
            #                              name='output',
            #                              opts=vis_options)
            #     vis.scatter(X=torch.cat((data, tag), 1).detach().cpu(),
            #                 win=vis_win[0], name='tag',
            #                 update='append',
            #                 opts=vis_options)
            # else:
            #     vis.scatter(X=torch.cat((data, output), 1).detach().cpu(),
            #                 win=vis_win[0], name='output',
            #                 update='append',
            #                 opts=vis_options)

    net = net.cpu()
