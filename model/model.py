import torch
from torch import nn

from utils import get_device


class CarColorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            # input - [3, 224, 224]
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # output - [32, 112, 112]

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # output - [64, 56, 56]

            nn.Conv2d(64, 128, 5, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # output - [128, 27, 27]
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 27 * 27, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 9)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)

        return x

device = get_device()

model = CarColorModel().to(device)

        