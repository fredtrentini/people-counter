import argparse
import os

import tensorflow as tf

from config import (
    DATASET_ANNOTATIONS_FOLDER
)
from dataset import Dataset
import models
from utils import (
    setup,
    Predictions,
    ModelData,
)

def run_plot_mode(dataset: Dataset, model_data: ModelData, correct_prediction_batches: list[Predictions]) -> None:
    if model_data is None:
        for img_batch, predictions in zip(dataset.iterate_img_batches(model_data), correct_prediction_batches):
            dataset.visualize(img_batch, predictions)
        
        return

    for img_batch in dataset.iterate_img_batches(model_data):
        predictions = dataset.predict_img_batch(model_data, img_batch)
        dataset.visualize(img_batch, predictions)

def run_benchmark_mode(dataset: Dataset, model_data: ModelData, correct_prediction_batches: list[Predictions]) -> None:
    person_count = 0
    success_count = 0
    correct_person_count = 0
    img_count = len(correct_prediction_batches) * len(correct_prediction_batches[0]["classes"])

    for img_batch_i, img_batch in enumerate(dataset.iterate_img_batches(model_data)):
        predictions = dataset.predict_img_batch(model_data, img_batch)
        img_batch_classes = predictions["classes"]

        for i, img in enumerate(img_batch):
            img_classes = img_batch_classes[i]
            img_person_count = len(img_classes[img_classes != -1])
            
            correct_img_classes = correct_prediction_batches[img_batch_i]["classes"][i]
            img_correct_person_count = len(correct_img_classes[correct_img_classes != -1])

            correct_person_count += img_correct_person_count
            person_count += img_person_count

            if img_person_count == img_correct_person_count:
                success_count += 1
            
            print(f"[{img_batch_i * 4 + i + 1}/{img_count}] | {img_person_count}/{img_correct_person_count}")
    
    print()
    print(f"Total person count: {person_count}/{correct_person_count}")
    print(f"Accuracy: {round((success_count / img_count) * 100, 2)}%")

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
    parser.add_argument(
        "-mode", 
        type=str, 
        required=False, 
        choices="plot benchmark".split(),
        default="plot",
        help="Script behavior"
    )
    
    args = parser.parse_args()
    annotations_filename = args.file
    model_arg = args.model
    mode = args.mode

    if annotations_filename is not None:
        if model_arg is not None:
            raise RuntimeError("-file and -model can't be used simultaneously")

        if mode == "benchmark":
            raise RuntimeError("-file can't be used with benchmark mode")

    if annotations_filename is not None:
        annotations_filename = os.path.join(DATASET_ANNOTATIONS_FOLDER, annotations_filename)
        assert os.path.exists(annotations_filename), "Annotations filename not found"
    
    model_data = model_name_to_model_data_function_map.get(
        model_arg,
        models._get_yolov8_pascalvoc_model_data
    )()

    dataset = Dataset.create_dataset()
    dataset.create_dataset_annotations(models.get_pretrained_model_data())
    prediction_batches = dataset.get_prediction_batches(annotations_filename)

    if mode == "plot":
        run_plot_mode(dataset, model_data, prediction_batches)
    else:
        run_benchmark_mode(dataset, model_data, prediction_batches)

if __name__ == "__main__":
    main()
