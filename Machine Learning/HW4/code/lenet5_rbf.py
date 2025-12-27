import torch
import torch.nn as nn
import numpy as np
import os

def scaled_tanh(x):
    return 1.7159 * torch.tanh((2.0/3.0) * x)


class SubsamplingLayer(nn.Module):
    def __init__(self, num_maps):
        super().__init__()
        self.a = nn.Parameter(torch.ones(num_maps))
        self.b = nn.Parameter(torch.zeros(num_maps))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.a.view(1, -1, 1, 1) * x + self.b.view(1, -1, 1, 1)
        return scaled_tanh(x)


class LeNet5RBF(nn.Module):
    def __init__(self, rbf_codes_path, num_classes=10, j=0.1):
        super().__init__()

        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = SubsamplingLayer(6)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = SubsamplingLayer(16)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)

        if not os.path.exists(rbf_codes_path):
            raise FileNotFoundError("RBF codes not found at " + rbf_codes_path)

        codes = np.load(rbf_codes_path).astype(np.float32)
        self.register_buffer("rbf_centers", torch.from_numpy(codes))
        self.j = j

    def forward_features(self, x):
        x = scaled_tanh(self.c1(x))
        x = self.s2(x)
        x = scaled_tanh(self.c3(x))
        x = self.s4(x)
        x = scaled_tanh(self.c5(x))
        x = x.view(x.size(0), -1)
        x = scaled_tanh(self.f6(x))
        return x

    def forward(self, x):
        f = self.forward_features(x)
        diff = f.unsqueeze(1) - self.rbf_centers.unsqueeze(0)
        dist2 = (diff**2).sum(dim=2)
        return dist2


def lenet_rbf_loss(y, target, j=0.1):
    y_correct = y.gather(1, target.unsqueeze(1)).squeeze(1)
    neg_y = -y

    max_val, _ = torch.max(torch.cat([neg_y, torch.full_like(neg_y[:, :1], -j)], dim=1),
                           dim=1, keepdim=True)

    sum_exp = torch.exp(neg_y - max_val).sum(dim=1)
    sum_exp += torch.exp(-j - max_val.squeeze(1))

    log_term = max_val.squeeze(1) + torch.log(sum_exp + 1e-12)
    return (y_correct + log_term).mean()
