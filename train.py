import argparse
import shutil

import keras
import tensorflow as tf
from ultralytics import YOLO

from config import (
    BATCH_SIZE,
    MAIN_MODEL_PATH_PASCALVOC,
    MAIN_MODEL_PATH_ULTRALYTICS,
    IMG_RESIZE,
    CONFIDENCE,
)
from dataset import Dataset
import models
from utils import (
    setup,
)

def train_pascalvoc(dataset: Dataset) -> None:
    model_data = models._get_yolov8_pascalvoc_model_data()
    model: keras.Model = model_data.model

    for layer in model.layers[:-20]:
        layer.trainable = False

    model.compile(
        classification_loss='binary_crossentropy',
        box_loss='ciou',
        optimizer=keras.optimizers.Adam(0.001),
        jit_compile=False,
    )

    (train_dataset, test_dataset) = dataset.load_data_as_keras(model_data.target_class)
    model.fit(train_dataset, epochs=10, validation_data=test_dataset)
    model.save(MAIN_MODEL_PATH_PASCALVOC)

    print(f"Trained model saved: {MAIN_MODEL_PATH_PASCALVOC}")

def train_ultralytics(dataset: Dataset) -> None:
    model_data = models._get_yolov8s_ultralytics_model_data()
    model: YOLO = model_data.model

    dataset.generate_ultralytics_files(model_data.target_class)
    model.train(
        data="dataset.yaml",
        epochs=10,
        freeze=[*range(5)],
        patience=8,
        batch=BATCH_SIZE,
        imgsz=IMG_RESIZE[1],
        workers=8,
        pretrained=True,
        resume=False,
        single_cls=False,
        box=5,
        cls=0.3,
        dfl=1,
    )
    results = model.val(
        imgsz=IMG_RESIZE[1],
        batch=BATCH_SIZE,
        conf=CONFIDENCE,
        iou=0.5,
        save_json=False,
        save_hybrid=False,
        split="val"
    )

    exported_path = model.export(format="onnx")
    shutil.copy(exported_path, MAIN_MODEL_PATH_ULTRALYTICS)
    print(f"Trained model saved: {MAIN_MODEL_PATH_ULTRALYTICS}")

def main():
    setup()
    print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")

    model_to_function_map = {
        "pascalvoc": train_pascalvoc,
        "ultralytics": train_ultralytics,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", 
        type=str, 
        required=False, 
        choices=[*model_to_function_map.keys()],
        default="pascalvoc",
        help="Choose model to train"
    )
    
    args = parser.parse_args()
    model_arg = args.model

    print("Step 1/4: Prepare dataset\n")
    dataset = Dataset.create_dataset()
    
    print("\nStep 2/4: Build dataset annotations\n")
    pretrained_model_data = models.get_pretrained_model_data()
    dataset.create_dataset_annotations(pretrained_model_data)
    
    print("\nStep 3/4: Train + test\n")
    print(f"Model: {model_arg}")
    model_to_function_map[model_arg](dataset)

if __name__ == "__main__":
    main()
