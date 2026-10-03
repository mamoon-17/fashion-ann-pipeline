"""Download Fashion MNIST and persist the original uint8 arrays."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import numpy as np
from tensorflow import keras


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    output = Path("data/raw")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "train.npz", images=x_train, labels=y_train)
    np.savez_compressed(output / "test.npz", images=x_test, labels=y_test)
    print(f"Prepared {len(y_train)} training and {len(y_test)} test images (28 x 28).")


if __name__ == "__main__":
    main()
