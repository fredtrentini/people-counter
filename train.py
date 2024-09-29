import shutil

import tensorflow as tf
from ultralytics import YOLO

from config import (
    BATCH_SIZE,
    MAIN_MODEL_PATH,
    IMG_RESIZE,
    CONFIDENCE,
)
from dataset import Dataset
import models
from utils import (
    setup,
)

def main():
    setup()
    print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")

    print("Step 1/4: Prepare dataset\n")
    dataset = Dataset.create_dataset()
    
    print("\nStep 2/4: Build dataset annotations\n")
    pretrained_model_data = models.get_pretrained_model_data()
    dataset.create_dataset_annotations(pretrained_model_data)
    
    print("\nStep 3/4: Train + test\n")
    model_data = models.get_model_data_to_train()
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
    shutil.copy(exported_path, MAIN_MODEL_PATH)

if __name__ == "__main__":
    main()
