import os
import time

import cv2
import keras
import tensorflow as tf
from ultralytics import YOLO, settings

from config import (
    BATCH_SIZE,
    EXIT_KEY,
    IMG_RESIZE,
    YOLO_DATASET_NAME,
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
    model_data = models.get_main_model_data()
    model: YOLO = model_data.model
    
    dataset.generate_ultralytics_files(model_data.target_class)
    settings.update({
        "runs_dir": "runs",
        "datasets_dir": YOLO_DATASET_NAME,
    })
    model.train(
        data=os.path.join(YOLO_DATASET_NAME, "dataset.yaml"),
        epochs=10,
        patience=8,
        batch=BATCH_SIZE,
        imgsz=IMG_RESIZE[1],
        workers=8,
        pretrained=True,
        resume=False,
        single_cls=False,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    results = model.val(
        imgsz=IMG_RESIZE[1],
        batch=BATCH_SIZE,
        conf=0.001,
        iou=0.7,
        save_json=False,
        save_hybrid=False,
        split="val"
    )

    from IPython import embed
    embed()
    
    exit()
    print("\nStep 4/4: Real time inference\n")
    count = 0
    camera = cv2.VideoCapture(0)

    while True:
        count += 1
        start = time.time()

        _, img = camera.read()

        
        end = time.time()
        ms_spent = int((end - start) * 1000)
        print(f"Frame {count}: {ms_spent}ms")
        


        if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
            break

if __name__ == "__main__":
    main()
