import time

import cv2
import tensorflow as tf
from ultralytics import YOLO

from config import (
    EXIT_KEY,
)
from dataset import Dataset
import models
from utils import (
    setup,
)

CONFIDENCE = 0.3

def main():
    setup()
    print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")
    print("\nStep 4/4: Real time inference\n")

    model_data = models.get_main_model_data()
    model: YOLO = model_data.model
    count = 0
    camera = cv2.VideoCapture(0)

    while True:
        count += 1
        start = time.time()

        _, img = camera.read()
        
        end = time.time()
        ms_spent = int((end - start) * 1000)
        print(f"Frame {count}: {ms_spent}ms")

        is_person_array = (results.boxes.cls == model_data.target_class) & (results.boxes.conf >= CONFIDENCE)
        results.boxes = results.boxes[is_person_array]

        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=frame, detections=sv.Detections.from_ultralytics(results))

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
            break

if __name__ == "__main__":
    main()
