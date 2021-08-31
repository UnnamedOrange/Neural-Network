from collections import OrderedDict
import torch


class vgg19(torch.nn.Module):
    def __init__(self, in_width=32, in_height=32):
        super().__init__()

        self.in_width = in_width
        self.in_height = in_height
        self.factor = 2**4  # 4 for the number of MaxPool.
        self.out_width = in_width // self.factor
        self.out_height = in_height // self.factor

        self.block1 = torch.nn.Sequential(OrderedDict([
            ('conv1-1', torch.nn.Conv2d(3, 64, (3, 3), padding='same')),
            ('norm1-1', torch.nn.BatchNorm2d(64)),
            ('relu1-1', torch.nn.ReLU()),

            ('conv1-2', torch.nn.Conv2d(64, 64, (3, 3), padding='same')),
            ('norm1-2', torch.nn.BatchNorm2d(64)),
            ('relu1-2', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d((2, 2), stride=(2, 2))),
        ]))
        self.block2 = torch.nn.Sequential(OrderedDict([
            ('conv2-1', torch.nn.Conv2d(64, 128, (3, 3), padding='same')),
            ('norm2-1', torch.nn.BatchNorm2d(128)),
            ('relu2-1', torch.nn.ReLU()),

            ('conv2-2', torch.nn.Conv2d(128, 128, (3, 3), padding='same')),
            ('norm2-2', torch.nn.BatchNorm2d(128)),
            ('relu2-2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool2d((2, 2), stride=(2, 2))),
        ]))
        self.block3 = torch.nn.Sequential(OrderedDict([
            ('conv3-1', torch.nn.Conv2d(128, 256, (3, 3), padding='same')),
            ('norm3-1', torch.nn.BatchNorm2d(256)),
            ('relu3-1', torch.nn.ReLU()),

            ('conv3-2', torch.nn.Conv2d(256, 256, (3, 3), padding='same')),
            ('norm3-2', torch.nn.BatchNorm2d(256)),
            ('relu3-2', torch.nn.ReLU()),

            ('conv3-3', torch.nn.Conv2d(256, 256, (3, 3), padding='same')),
            ('norm3-3', torch.nn.BatchNorm2d(256)),
            ('relu3-3', torch.nn.ReLU()),

            ('conv3-4', torch.nn.Conv2d(256, 256, (3, 3), padding='same')),
            ('norm3-4', torch.nn.BatchNorm2d(256)),
            ('relu3-4', torch.nn.ReLU()),

            ('pool3', torch.nn.MaxPool2d((2, 2), stride=(2, 2))),
        ]))
        self.block4 = torch.nn.Sequential(OrderedDict([
            ('conv4-1', torch.nn.Conv2d(256, 512, (3, 3), padding='same')),
            ('norm4-1', torch.nn.BatchNorm2d(512)),
            ('relu4-1', torch.nn.ReLU()),

            ('conv4-2', torch.nn.Conv2d(512, 512, (3, 3), padding='same')),
            ('norm4-2', torch.nn.BatchNorm2d(512)),
            ('relu4-2', torch.nn.ReLU()),

            ('conv4-3', torch.nn.Conv2d(512, 512, (3, 3), padding='same')),
            ('norm4-3', torch.nn.BatchNorm2d(512)),
            ('relu4-3', torch.nn.ReLU()),

            ('conv4-4', torch.nn.Conv2d(512, 512, (3, 3), padding='same')),
            ('norm4-4', torch.nn.BatchNorm2d(512)),
            ('relu4-4', torch.nn.ReLU()),

            ('pool4', torch.nn.MaxPool2d((2, 2), stride=(2, 2))),
        ]))
        self.block5 = torch.nn.Sequential(OrderedDict([
            ('conv5-1', torch.nn.Conv2d(512, 512, (3, 3), padding='same')),
            ('norm5-1', torch.nn.BatchNorm2d(512)),
            ('relu5-1', torch.nn.ReLU()),

            ('conv5-2', torch.nn.Conv2d(512, 512, (3, 3), padding='same')),
            ('norm5-2', torch.nn.BatchNorm2d(512)),
            ('relu5-2', torch.nn.ReLU()),

            ('conv5-3', torch.nn.Conv2d(512, 512, (3, 3), padding='same')),
            ('norm5-3', torch.nn.BatchNorm2d(512)),
            ('relu5-3', torch.nn.ReLU()),

            ('conv5-4', torch.nn.Conv2d(512, 512, (3, 3), padding='same')),
            ('norm5-4', torch.nn.BatchNorm2d(512)),
            ('relu5-4', torch.nn.ReLU()),

            # ('pool5', torch.nn.MaxPool2d((2, 2), stride=(2, 2))),
        ]))
        self.block6 = torch.nn.Sequential(OrderedDict([
            ('flatten6-1', torch.nn.Flatten()),
            ('fc6-1', torch.nn.Linear(self.out_width * self.out_width * 512, 4096)),
            ('norm6-1', torch.nn.BatchNorm1d(4096)),
            ('relu6-1', torch.nn.ReLU()),
            ('dropout6-1', torch.nn.Dropout()),

            ('fc6-2', torch.nn.Linear(4096, 4096)),
            ('norm6-2', torch.nn.BatchNorm1d(4096)),
            ('relu6-2', torch.nn.ReLU()),
            ('dropout6-2', torch.nn.Dropout()),

            ('fc6-3', torch.nn.Linear(4096, 10)),
            ('norm6-3', torch.nn.BatchNorm1d(10)),
            ('relu6-3', torch.nn.ReLU()),
            ('softmax6-3', torch.nn.Softmax(dim=-1)),
        ]))  # Transfer train block6.

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20)

    def forward(self, data_in):
        output = self.block1(data_in)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.block6(output)
        return output
