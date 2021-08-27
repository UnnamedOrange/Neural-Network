import torch


def make_dataset(f, size=10000, seed=None, x_range=3):
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.rand(size=(size, 1), dtype=torch.float)
    x = x * x_range + 0.05
    y = f(x)

    return torch.utils.data.TensorDataset(x, y)
