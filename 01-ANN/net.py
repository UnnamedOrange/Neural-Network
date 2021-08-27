from collections import OrderedDict
import torch


class ann(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(1, 32)),
            ('relu1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(32, 120)),
            ('relu2', torch.nn.ReLU()),
            ('fc3', torch.nn.Linear(120, 120)),
            ('relu3', torch.nn.ReLU()),
            ('fc4', torch.nn.Linear(120, 84)),
            ('relu4', torch.nn.ReLU()),
            ('fc5', torch.nn.Linear(84, 1)),
        ]))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=4e-3)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, data_in):
        output = self.fc(data_in)
        return output
