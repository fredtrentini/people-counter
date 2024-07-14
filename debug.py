from __future__ import annotations

import os
import json

import keras
import tensorflow as tf

from dataset import Dataset
import models
from utils import (
    setup,
)

RESULTS_FOLDER = "results"

def main():
    setup()
    print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    dataset = Dataset.create_dataset()
    
    pretrained_model_data = models.get_pretrained_model_data()
    dataset.create_dataset_annotations(pretrained_model_data)
    
    model_datas = models.get_model_datas()

    for n, model_data in enumerate(model_datas, start=1):
        model: keras.Model = model_data.model

        (train_dataset, test_dataset) = dataset.load_data(model_data.target_class)  
        print(f"\n\nModel {n}/{len(model_datas)}: {model.name}\n")

        raw_results = model.evaluate(test_dataset)
        results = dict(zip(model.metrics_names, raw_results))
        path = os.path.join(RESULTS_FOLDER, f"{model.name}_evaluate.json")

        with open(path, "w") as file:
            json.dump(results, file, indent=4)
            print(f"Results written to {path}")

        for layer in model.layers:
            layer.trainable = False

        model.compile(
            classification_loss='binary_crossentropy',
            box_loss='ciou',
            optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
            jit_compile=False,
        )

        history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
        path = os.path.join(RESULTS_FOLDER, f"{model.name}_fit.json")

        with open(path, "w") as file:
            json.dump(history, file, indent=4)
            print(f"Results written to {path}")

if __name__ == "__main__":
    main()
