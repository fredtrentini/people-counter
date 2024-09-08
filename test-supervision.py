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
video_path = glob.glob(os.path.join(config.VIDEOS_DIR, "**"))[2]
video_info = sv.VideoInfo.from_video_path(video_path)
i = 0

for frame in sv.get_video_frames_generator(video_path):
    # from IPython import embed
    # embed()
    # exit()
    i += 1
    print(f"Frame {i}/{video_info.total_frames}")
    results = model.predict(frame, imgsz=config.IMG_RESIZE[1])[0]

    is_person_array = (results.boxes.cls == PERSON_CLASS) & (results.boxes.conf >= CONFIDENCE)
    results.boxes = results.boxes[is_person_array]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=frame, detections=sv.Detections.from_ultralytics(results))

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
