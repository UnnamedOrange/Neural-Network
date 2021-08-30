def make_dataset(filenames: list):
    '''Load CIFAR-10 to make a dataset.'''

    import numpy as np
    import torch
    import pickle

    data = []
    labels = []

    for filename in filenames:
        with open(filename, 'rb') as file:
            d = pickle.load(file, encoding='bytes')
        data.append(d[b'data'].reshape(
            [-1, 3, 32, 32]))
        labels.append(d[b'labels'])

    for i in range(len(data)):
        data[i] = np.array(data[i], dtype=np.float32)
        for j in range(3):
            data[i][:, j, :, :] = (
                data[i][:, j, :, :] - np.mean(data[i][:, j, :, :])) / np.std(data[i][:, j, :, :])
        data[i] = torch.from_numpy(data[i])
    data = torch.cat(data, dim=0)

    for i in range(len(labels)):
        labels[i] = torch.tensor(labels[i], dtype=torch.long)
    labels = torch.cat(labels, dim=0)

    return torch.utils.data.TensorDataset(data, labels)
