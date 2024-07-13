from __future__ import annotations

import time

import cv2
import keras
import tensorflow as tf

from config import (
    EXIT_KEY,
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
    model: keras.Model = model_data.model
    (train_dataset, test_dataset) = dataset.load_data(model_data.target_class)

    for layer in model.layers:
        layer.trainable = False

    model.compile(
        classification_loss='binary_crossentropy',
        box_loss='ciou',
        optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
        jit_compile=False,
    )

    history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
    
    exit()
    print("\nStep 4/4: Real time inference\n")
    count = 0
    camera = cv2.VideoCapture(0)

    while True:
        count += 1
        start = time.time()

        _, img = camera.read()
        # TODO: Predict img with domain model
        
        end = time.time()
        ms_spent = int((end - start) * 1000)
        print(f"Frame {count}: {ms_spent}ms")
        
        # TODO: Plot image with highlighted objects in non-blocking way (pure cv2 probably)

        if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
            break

if __name__ == "__main__":
    main()
