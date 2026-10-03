"""Normalize the raw pixels and produce stratified train, validation and test sets."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


def normalize(images):
    return images.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    with open(args.params, encoding="utf-8") as stream:
        params = yaml.safe_load(stream)["preprocess"]
    with np.load("data/raw/train.npz") as raw:
        images, labels = normalize(raw["images"]), raw["labels"]
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=params["validation_size"],
        random_state=params["seed"], stratify=labels,
    )
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "train.npz", images=x_train, labels=y_train)
    np.savez_compressed(output / "val.npz", images=x_val, labels=y_val)
    with np.load("data/raw/test.npz") as raw:
        np.savez_compressed(output / "test.npz", images=normalize(raw["images"]), labels=raw["labels"])
    print(f"Normalized to [0, 1]: train={len(y_train)}, val={len(y_val)}, test=10000.")


if __name__ == "__main__":
    main()
