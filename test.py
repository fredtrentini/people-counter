import argparse
import os

import keras
import numpy as np
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

from config import (
    BATCH_SIZE,
    DATASET_ANNOTATIONS_FOLDER,
    MAIN_MODEL_PATH_PASCALVOC,
    TRAIN_RATIO,
)
from dataset import Dataset
import models
from utils import (
    setup,
    Predictions,
    ModelData,
)

def normalized_mean_absolute_error(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    y_true_mean = np.mean(np.abs(y_true))

    return mae / y_true_mean

def run_plot_mode(dataset: Dataset, model_data: ModelData, correct_prediction_batches: list[Predictions]) -> None:
    if model_data is None:
        for img_batch, predictions in zip(dataset.iterate_img_batches(model_data, test_only=True), correct_prediction_batches):
            dataset.visualize(img_batch, predictions)
        
        return

    for img_batch in dataset.iterate_img_batches(model_data, test_only=True):
        predictions = dataset.predict_img_batch(model_data, img_batch)
        dataset.visualize(img_batch, predictions)

def run_benchmark_mode(dataset: Dataset, model_data: ModelData, correct_prediction_batches: list[Predictions]) -> None:
    person_count = 0
    success_count = 0
    correct_person_count = 0
    tested_img_count = 0
    img_count = len(correct_prediction_batches) * len(correct_prediction_batches[0]["classes"])
    img_person_counts = []
    img_correct_person_counts = []
    
    TEST_RATIO = 1 - TRAIN_RATIO
    test_img_count = int(img_count * TEST_RATIO)

    for batch_i, img_batch in enumerate(dataset.iterate_img_batches(model_data)):
        predictions = dataset.predict_img_batch(model_data, img_batch)
        img_batch_classes = predictions["classes"]

        for batch_img_i, img in enumerate(img_batch):
            shallow_img_i = batch_i * BATCH_SIZE + batch_img_i
            
            if shallow_img_i not in dataset.test_indexes:
                continue

            img_classes = img_batch_classes[batch_img_i]
            img_person_count = len(img_classes[img_classes != -1])
            
            correct_img_classes = correct_prediction_batches[batch_i]["classes"][batch_img_i]
            img_correct_person_count = len(correct_img_classes[correct_img_classes != -1])

            correct_person_count += img_correct_person_count
            person_count += img_person_count

            img_person_counts.append(img_person_count)
            img_correct_person_counts.append(img_correct_person_count)

            if img_person_count == img_correct_person_count:
                success_count += 1
            
            tested_img_count += 1
            print(f"[{tested_img_count}/{test_img_count}] | {img_person_count}/{img_correct_person_count}")
    
    print()
    print(f"Total person count: {person_count}/{correct_person_count}")
    print(f"Accuracy: {round((success_count / test_img_count) * 100, 2)}%")
    print(f"MAE: {round(mean_absolute_error(img_correct_person_counts, img_person_counts), 3)}")
    print(f"MAEN: {round(normalized_mean_absolute_error(img_correct_person_counts, img_person_counts), 3)}")

def main():
    setup()
    print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")

    model_data_pascalvoc = models._get_yolov8_pascalvoc_model_data()
    model_data_pascalvoc.model = keras.models.load_model(MAIN_MODEL_PATH_PASCALVOC)
    
    model_name_to_model_data_function_map = {
        "pascalvoc": models._get_yolov8_pascalvoc_model_data,
        "ultralytics": models._get_yolov8s_ultralytics_model_data,
        "pascalvoc_trained": model_data_pascalvoc,
        "ultralytics_trained": models._get_yolov8s_ultralytics_model_data_trained,
    }

    parser = argparse.ArgumentParser(description="Visualize results")
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
