"""Evaluate the saved model and export metrics and a labeled confusion matrix."""
import argparse
import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
import tensorflow as tf
import yaml

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    with open(args.params, encoding="utf-8") as stream:
        params = yaml.safe_load(stream)["train"]
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with np.load("data/processed/test.npz") as data:
        images, labels = data["images"], data["labels"]
    model = tf.keras.models.load_model("models/model.h5", compile=False)
    probabilities = np.concatenate([
        model(images[start:start + params["batch_size"]], training=False).numpy()
        for start in range(0, len(labels), params["batch_size"])
    ])
    predictions = probabilities.argmax(axis=1)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(10))
    metrics = {
        "test_loss": float(tf.reduce_mean(loss).numpy()),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "test_samples": int(len(labels)),
        "target_accuracy_met": bool(accuracy_score(labels, predictions) >= 0.85),
    }
    output = Path("reports/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    content = json.dumps(metrics, indent=2) + "\n"
    Path("metrics.json").write_text(content, encoding="utf-8")
    (output / "metrics.json").write_text(content, encoding="utf-8")
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Fashion MNIST test set confusion matrix")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    print(content)


if __name__ == "__main__":
    main()
