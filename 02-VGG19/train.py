import math
import torch
import os


def train(net: torch.nn.Module, is_gpu: bool,
          dataset: torch.utils.data.TensorDataset,
          batch_size: int, num_workers: int, n_epoch: int,
          vis, vis_win: list, vis_options: dict,
          model_file='latest_model.pt',
          on_test=None):

    if model_file and os.path.exists(model_file):
        net.load_state_dict(torch.load(model_file))
        print("Model loaded.")
    else:
        print("No model to load.")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_batch = n_epoch * math.ceil(len(dataset) / batch_size)
    losses = torch.zeros(size=(n_batch,))
    indexes = torch.arange(n_batch)
    batch_index = 0

    try:
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
    except KeyboardInterrupt:
        pass

    net = net.cpu()
    torch.save(net.state_dict(), model_file)
    print('Model saved.')
