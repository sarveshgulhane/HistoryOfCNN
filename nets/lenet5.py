import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # input: (N,1,32,32)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)  # -> (N,6,28,28)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # -> (N,6,14,14)

        self.conv2 = nn.Conv2d(
            6, 16, kernel_size=5, stride=1
        )  # -> (N,16,10,10)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # -> (N,16,5,5)

        # classic LeNet: flatten -> 16*5*5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x
