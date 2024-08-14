import glob
import os

import config
import cv2
import tensorflow as tf
import supervision as sv
from ultralytics import YOLO

PERSON_CLASS = 0
CONFIDENCE = 0.3

print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")

model = YOLO("yolov8s.pt")
video_path = video_path = glob.glob(os.path.join(config.INPUT_DIR, "**"))[2]
video_info = sv.VideoInfo.from_video_path(video_path)
i = 0

model.train(data="./dataset-annotations/current.json", epochs=1)
