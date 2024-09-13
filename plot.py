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
    
    model_name_to_model_data_function_map = {
        "pascalvoc": models._get_yolov8_pascalvoc_model_data,
        "yolov8s": models._get_yolov8s_ultralytics_model_data,
        "yolov8s_trained": models._get_yolov8s_ultralytics_model_data_trained,
    }

    parser = argparse.ArgumentParser(description="Visualize annotations")
    parser.add_argument(
        "-file", 
        type=str, 
        required=False, 
        help="Get annotations from given .json file"
    )
    parser.add_argument(
        "-model", 
        type=str, 
        required=False, 
        choices=[*model_name_to_model_data_function_map.keys()],
        help="Model to generate annotations in runtime"
    )
    args = parser.parse_args()
    annotations_filename = args.file
    model_arg = args.model

    if annotations_filename is not None and model_arg is not None:
        raise RuntimeError("-file and -model can't be used simultaneously")

    if annotations_filename is not None:
        annotations_filename = os.path.join(DATASET_ANNOTATIONS_FOLDER, annotations_filename)
        assert os.path.exists(annotations_filename), "Annotations filename not found"

    model_data = models.get_pretrained_model_data()
    dataset = Dataset.create_dataset()
    dataset.create_dataset_annotations(model_data)
    prediction_batches = dataset.get_prediction_batches(annotations_filename)

    if model_arg is None:
        for img_batch, predictions in zip(dataset.iterate_img_batches(model_data), prediction_batches):
            dataset.visualize(img_batch, predictions)
        
        return
    
    model_data = model_name_to_model_data_function_map[model_arg]()

    for img_batch in dataset.iterate_img_batches(model_data):
        predictions = dataset.predict_img_batch(model_data, img_batch)
        dataset.visualize(img_batch, predictions)

if __name__ == "__main__":
    main()
