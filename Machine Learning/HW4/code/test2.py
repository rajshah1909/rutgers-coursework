from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from lenet2 import STNLeNet

def test(dataloader, model):
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.float()
            y = y.long()

            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    print("test accuracy:", correct / total)


def main():
    # --- SAME preprocessing as train2.py (Pad to 32×32 and ×255) ---
    test_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    # --- Load MNIST test set ---
    mnist_test = MNIST(
        root="C:/Users/rajsh/OneDrive/Desktop/HW4/data/",
        train=False,
        transform=test_transform,
        download=False
    )

    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    # --- Load model ---
    model = STNLeNet()
    model.load_state_dict(torch.load(
        "C:/Users/rajsh/OneDrive/Desktop/HW4/models/LeNet2.pth",
        map_location="cpu"
    ))

    test(test_loader, model)


if __name__ == "__main__":
    main()