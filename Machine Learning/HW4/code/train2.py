import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from lenet2 import STNLeNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# PATHS (MATCHING YOUR FOLDER)
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "LeNet2.pth")
FIG_DIR = os.path.join(BASE_DIR, "figures")

LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "lenet2_train_log.txt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ===============================
# LOGGING FUNCTION
# ===============================
def log_write(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


# ===============================
# LOAD DATA
# ===============================
def get_dataloaders(batch_size=64, val_ratio=0.1):

    train_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomAffine(
            degrees=25,
            translate=(0.15, 0.15),
            scale=(0.8, 1.2),
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    test_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    full_train = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    val_ds.dataset.transform = test_transform

    test_ds = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ===============================
# EVALUATION
# ===============================
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc


# ===============================
# CONFUSION MATRIX FUNCTION
# ===============================
def compute_confusion_matrix(model, loader):
    model.eval()
    cm = np.zeros((10, 10), dtype=int)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            lbl = y.cpu().numpy()
            cm[lbl[0], preds[0]] += 1

    return cm


# ===============================
# TRAINING FUNCTION
# ===============================
def train():
    train_loader, val_loader, test_loader = get_dataloaders()
    model = STNLeNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    best_val_acc = 0.0

    train_errors = []
    val_errors = []

    log_write("===== Starting STN-LeNet Training =====")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_err = 1.0 - train_acc

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_err = 1.0 - val_acc

        train_errors.append(train_err)
        val_errors.append(val_err)

        # -------- logging --------
        log_write(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_err={train_err:.4f}, val_loss={val_loss:.4f}, val_err={val_err:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            log_write(f"Saved new best model (val_acc={val_acc:.4f})")

    log_write(f"Training finished. Best val_acc = {best_val_acc:.4f}")

    # ---------- SAVE ERROR CURVE ----------
    epochs = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_errors, label="Train Error")
    plt.plot(epochs, val_errors, label="Val Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.title("LeNet-5 (STN) Error Curve")
    plt.legend()
    fig_path = os.path.join(FIG_DIR, "lenet2_error_curve.png")
    plt.savefig(fig_path)
    plt.close()
    log_write(f"Saved error curve to: {fig_path}")

    # ---------- CONFUSION MATRIX ----------
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    cm = compute_confusion_matrix(model, test_loader)
    cm_path = os.path.join(LOG_DIR, "lenet2_confusion.txt")
    np.savetxt(cm_path, cm, fmt="%d")
    log_write(f"Saved confusion matrix to: {cm_path}")

    log_write("===== STN-LeNet Training Complete =====")


# ===============================
# RUN TRAINING
# ===============================
if __name__ == "__main__":
    train()