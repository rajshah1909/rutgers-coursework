import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lenet5_rbf import LeNet5RBF
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "data")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "LeNet1.pth")
RBF_CODES_PATH = os.path.join(BASE_DIR, "..", "models", "rbf_codes.npy")
FIG_DIR = os.path.join(BASE_DIR, "..", "figures")
LOG_DIR = os.path.join(BASE_DIR, "..", "log")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------------------------------
# Load MNIST test set (same preprocessing as Q1)
# -------------------------------------------------
def load_mnist_test():
    tfm = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)
    ])

    test_ds = datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=tfm
    )
    return DataLoader(test_ds, batch_size=1, shuffle=False)


# -------------------------------------------------
# Find confusion matrix + most confusing examples
# -------------------------------------------------
def analyze_most_confusing(model, test_loader):
    model.eval()

    cm = np.zeros((10, 10), dtype=int)
    best_conf = [-1] * 10
    best_pred = [-1] * 10
    best_img = [None] * 10
    best_index = [-1] * 10

    idx_global = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).long()

            penalties = model(x)
            pred = torch.argmin(penalties, dim=1).item()
            true = y.item()

            cm[true][pred] += 1

            if pred != true:

                probs = torch.softmax(-penalties, dim=1)
                conf_wrong = probs[0][pred].item()

                if conf_wrong > best_conf[true]:
                    best_conf[true] = conf_wrong
                    best_pred[true] = pred
                    best_img[true] = x.cpu().clone().squeeze(0)
                    best_index[true] = idx_global

            idx_global += 1

    # Save confusion matrix
    cm_path = os.path.join(LOG_DIR, "lenet1_confusion.txt")
    np.savetxt(cm_path, cm, fmt="%d")
    print("Saved confusion matrix to:", cm_path)

    # Save summary
    summary_path = os.path.join(LOG_DIR, "lenet1_most_confusing.txt")
    summary = []

    for d in range(10):
        if best_img[d] is None:
            summary.append(f"Digit {d}: No misclassifications.\n")
            continue

        filename = f"most_confusing_true{d}_pred{best_pred[d]}.png"
        out_path = os.path.join(FIG_DIR, filename)

        img = best_img[d].numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img[0]).save(out_path)

        summary.append(
            f"Digit {d}: predicted as {best_pred[d]}, "
            f"confidence={best_conf[d]:.4f}, index={best_index[d]}, file={filename}\n"
        )

    with open(summary_path, "w") as f:
        f.writelines(summary)

    print("Saved most confusing summary to:", summary_path)


def main():
    test_loader = load_mnist_test()

    model = LeNet5RBF(RBF_CODES_PATH).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    print("Model loaded:", MODEL_PATH)

    analyze_most_confusing(model, test_loader)


if __name__ == "__main__":
    main()