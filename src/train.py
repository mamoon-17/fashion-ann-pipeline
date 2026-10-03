"""Train the fully connected ANN configured in params.yaml."""
import argparse
import csv
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    with open(args.params, encoding="utf-8") as stream:
        params = yaml.safe_load(stream)["train"]
    tf.keras.utils.set_random_seed(params["seed"])
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    def dataset(split, shuffle=False):
        with np.load(f"data/processed/{split}.npz") as data:
            images, labels = data["images"], data["labels"]
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            ds = ds.shuffle(len(labels), seed=params["seed"], reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return ds.batch(params["batch_size"]).with_options(options).prefetch(1)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(params["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(dataset("train", shuffle=True), validation_data=dataset("val"),
                        epochs=params["epochs"], verbose=2)
    output = Path("models")
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.h5")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", *history.history])
        for i in range(len(history.epoch)):
            writer.writerow([i + 1, *(history.history[key][i] for key in history.history)])
    print("Saved models/model.h5 and models/history.csv.")


if __name__ == "__main__":
    main()
