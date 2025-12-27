import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# Spatial Transformer (STN)
# ================================
class STN(nn.Module):
    def __init__(self):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        # compute output size dynamically
        loc_out = self.localization_output_size()

        self.fc_loc = nn.Sequential(
            nn.Linear(loc_out, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # initialize affine transform as identity
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def localization_output_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 32, 32)  # your MNIST input after padding
            x = self.localization(x)
            return x.numel()

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


# ================================
# STN + LeNet (Q2 Model)
# ================================
class STNLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stn = STN()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # Spatial transformer
        x = self.stn(x)

        # LeNet path
        x = F.tanh(self.conv1(x))
        x = self.pool(x)

        x = F.tanh(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x