from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import mnist
import torch
import numpy as np
import torchvision
from lenet5_rbf import LeNet5RBF
 
def test(dataloader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.float()
            y = y.long()

            penalties = model(x)
            pred = torch.argmin(penalties, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    print("test accuracy:", correct / total)

def main():
    pad = transforms.Pad(2)

    mnist_test = MNIST(
        root="C:/Users/rajsh/OneDrive/Desktop/HW4/data/",
        train=False,
        transform=transforms.Compose([pad, transforms.ToTensor()]),
        download=False,
    )

    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    model = LeNet5RBF("C:/Users/rajsh/OneDrive/Desktop/HW4/models/rbf_codes.npy")
    model.load_state_dict(torch.load(
    "C:/Users/rajsh/OneDrive/Desktop/HW4/models/LeNet1.pth",
    map_location="cpu"
	))

    test(test_loader, model)

if __name__ == "__main__":
    main()