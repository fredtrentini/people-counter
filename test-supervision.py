from __future__ import annotations

import glob
import os
import pathlib

import cv2
import tensorflow as tf
import torch
import supervision as sv
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np

PRETRAINED_MODEL_PATH = "./yolo.pt"
INPUT_DIR = "data"
EXIT_KEY = "q"

# Step 2
FORCE_DATASET_LABELS_REBUILD = True
VISUALIZE_LABELED_IMAGE = True
BATCH_SIZE = 1
CONFIDENCE = 0.3
IMG_RESIZE = (480, 640)
PERSON_CLASS = 0

def main():
    os.chdir(pathlib.Path(__file__).parent.resolve())
    assert os.path.exists(INPUT_DIR), "Videos input folder not found"
    
    print(f"Devices: {[device.device_type for device in tf.config.list_physical_devices()]}\n")

    model = YOLO("yolov8s.pt")

    video_paths = glob.glob(os.path.join(INPUT_DIR, "**"))

    for video_i, video_path in enumerate(video_paths, start=1):
        print(f"Video {video_i}/{len(video_paths)}")
        video_info = sv.VideoInfo.from_video_path(video_path)
        i = 0

        for frame in sv.get_video_frames_generator(video_path):
            i += 1
            print(f"Frame {i}/{video_info.total_frames}")
            results: Results = model(frame, imgsz=1280)[0]
            detections = sv.Detections.from_yolov8(results)
            from IPython import embed
            embed()

            is_person_array = (detections.class_id == PERSON_CLASS) & (detections.confidence >= CONFIDENCE)
            detections.xyxy = detections.xyxy[is_person_array]
            detections.class_id = detections.class_id[is_person_array]
            detections.confidence = detections.confidence[is_person_array]

            box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
            frame = box_annotator.annotate(scene=frame, detections=detections)

            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
