import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # BLOCK 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # BLOCK 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # BLOCK 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # BLOCK 4 (głębiej)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 200)

    def forward(self, x):

        # BLOCK 1
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.pool(x)

        # BLOCK 2
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.pool(x)

        # BLOCK 3
        x = self.conv5(x).relu()
        x = self.conv6(x).relu()
        x = self.pool(x)

        # BLOCK 4
        x = self.conv7(x).relu()
        x = self.pool(x)

        # HEAD
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x