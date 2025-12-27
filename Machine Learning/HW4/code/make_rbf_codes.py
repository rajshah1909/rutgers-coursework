import argparse
import os
import glob
import numpy as np
from PIL import Image

def build_digit_prototype(class_dir, target_size=(12, 7), max_images=500):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(class_dir, ext)))

    if len(paths) == 0:
        raise RuntimeError(f"No images found in {class_dir}")

    paths = paths[:max_images]
    w, h = target_size
    acc = np.zeros((h, w), dtype=np.float64)

    for p in paths:
        img = Image.open(p).convert("L")
        img = img.resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float64)
        acc += arr

    avg = acc / len(paths)
    thr = avg.mean()
    bitmap = np.where(avg > thr, 1.0, -1.0)

    return bitmap.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits-root", type=str, required=True)
    parser.add_argument("--output", type=str,
        default="C:/Users/rajsh/OneDrive/Desktop/HW4/models/rbf_codes.npy")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    codes = np.zeros((10, 84), dtype=np.float32)
    for d in range(10):
        print(f"Processing digit {d}...")
        class_dir = os.path.join(args.digits_root, str(d))
        codes[d] = build_digit_prototype(class_dir)

    np.save(args.output, codes)
    print("Saved RBF codes to:", args.output)


if __name__ == "__main__":
    main()
