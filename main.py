import cv2
import tensorflow as tf
import supervision as sv
from ultralytics import YOLO

from config import (
    IMG_RESIZE,
    EXIT_KEY,
)
import models
from utils import (
    setup,
)

CONFIDENCE = 0.3

def main():
    setup()
    print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")
    print("Step 4/4: Real time inference\n")

    model_data = models.get_main_model_data()
    model: YOLO = model_data.model
    camera = cv2.VideoCapture(0)

    while True:
        _, img = camera.read()
        results = model.predict(img, imgsz=IMG_RESIZE[1])[0]

        is_person_array = (results.boxes.cls == model_data.target_class) & (results.boxes.conf >= CONFIDENCE)
        results.boxes = results.boxes[is_person_array]

        box_annotator = sv.BoxAnnotator()
        img = box_annotator.annotate(scene=img, detections=sv.Detections.from_ultralytics(results))

        cv2.imshow("Frame", img)

        if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
            break

if __name__ == "__main__":
    main()
