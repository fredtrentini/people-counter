import argparse
import os

import tensorflow as tf

from config import (
    DATASET_ANNOTATIONS_FOLDER
)
from dataset import Dataset
import models
from utils import setup

def main():
    setup()
    print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")

    parser = argparse.ArgumentParser(description="Visualize annotations")
    parser.add_argument(
        "-file", 
        type=str, 
        required=False, 
        help="Get annotations from given .json file"
    )
    args = parser.parse_args()
    annotations_filename = args.file

    if annotations_filename is not None:
        annotations_filename = os.path.join(DATASET_ANNOTATIONS_FOLDER, annotations_filename)
        assert os.path.exists(annotations_filename), "Annotations filename not found"

    model_data = models.get_pretrained_model_data()
    dataset = Dataset.create_dataset()
    dataset.create_dataset_annotations(model_data)
    prediction_batches = dataset.get_prediction_batches(annotations_filename)

    for img_batch, predictions in zip(dataset.iterate_img_batches(model_data), prediction_batches):
        dataset.visualize(img_batch, predictions)

if __name__ == "__main__":
    main()
