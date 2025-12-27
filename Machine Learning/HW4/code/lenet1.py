import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim

from lenet5_rbf import LeNet5RBF, lenet_rbf_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RBF_PATH = "C:/Users/rajsh/OneDrive/Desktop/HW4/models/rbf_codes.npy"
MODEL_PATH = "C:/Users/rajsh/OneDrive/Desktop/HW4/models/LeNet1.pth"
FIG_DIR = "C:/Users/rajsh/OneDrive/Desktop/HW4/figures"
LOG_DIR = "C:/Users/rajsh/OneDrive/Desktop/HW4/log"
LOG_FILE = os.path.join(LOG_DIR, "lenet1_train_log.txt")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------------
# Logging function (NEW)
# ------------------------------
def log_write(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def load_mnist(batch_size=1):
    tfm = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)
    ])
    train = datasets.MNIST(root="../data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="../data", train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size, shuffle=True), \
           DataLoader(test, batch_size, shuffle=False)


def compute_confusion_matrix(model, test_loader):
    cm = np.zeros((10, 10), dtype=int)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            penalties = model(x)
            pred = torch.argmin(penalties, dim=1)

            cm[y.item(), pred.item()] += 1

    return cm


def main():
    train_loader, test_loader = load_mnist()

    model = LeNet5RBF(RBF_PATH).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=0.001)

    train_errs = []
    test_errs = []

    log_write("===== Starting LeNet1 (RBF) Training =====")

    for ep in range(1, 21):
        model.train()
        total = correct = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            penalties = model(x)
            loss = lenet_rbf_loss(penalties, y)
            loss.backward()
            opt.step()

            pred = penalties.argmin(dim=1)
            correct += (pred == y).sum().item()
            total += 1

        train_err = 1 - correct / total
        train_errs.append(train_err)

        # testing
        model.eval()
        total = correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                penalties = model(x)
                pred = torch.argmin(penalties, dim=1)
                correct += (pred == y).sum().item()
                total += 1

        test_err = 1 - correct / total
        test_errs.append(test_err)

        # -------- log epoch results --------
        log_write(f"Epoch {ep}: train_err={train_err:.4f}, test_err={test_err:.4f}")

    # ---------- SAVE MODEL ----------
    torch.save(model.state_dict(), MODEL_PATH)
    log_write(f"Saved model to {MODEL_PATH}")

    # ---------- SAVE ERROR CURVE ----------
    plt.figure()
    plt.plot(train_errs, label="Train Error")
    plt.plot(test_errs, label="Test Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.title("LeNet-5 Error Curve")
    plt.legend()
    curve_path = os.path.join(FIG_DIR, "lenet1_error_curve.png")
    plt.savefig(curve_path)
    plt.close()
    log_write(f"Saved error curve to {curve_path}")

    # ---------- SAVE CONFUSION MATRIX ----------
    cm = compute_confusion_matrix(model, test_loader)
    cm_path = os.path.join(LOG_DIR, "lenet1_confusion.txt")
    np.savetxt(cm_path, cm, fmt="%d")
    log_write(f"Saved confusion matrix to {cm_path}")

    log_write("===== Training Complete =====")


if __name__ == "__main__":   # FIXED BUG ✔
    main()