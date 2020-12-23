import torch
import torch.nn as nn

from environment import ActionType


class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, len(ActionType)),
        )
        #     nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 8, kernel_size=2, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 8, kernel_size=2, stride=1),
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        self.c3 =  nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(ActionType)),
        )

    def forward(self, x):
        # result = self.conv(x)
        y1 = self.c1(x)
        y2 = self.c2(y1)
        y3 = self.c3(y2)
        y4 = self.flatten(y3)
        return y3
