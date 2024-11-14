import glob
import os

import config
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import supervision as sv

PERSON_CLASS = 0
CONFIDENCE = 0.5
REGION_POINTS = [(700, 150), (1250, 150), (1250, 0), (700, 0)]

model = YOLO("yolov8s.pt")
video_path = video_path = glob.glob(os.path.join(config.VIDEOS_DIR, "**"))[2]
frame_generator = sv.get_video_frames_generator(video_path)

counter = object_counter.ObjectCounter(
    view_img=True,
    reg_pts=REGION_POINTS,
    names=model.names,
    draw_tracks=True,
)

for img in frame_generator:
    tracks = model.track(img, persist=True, show=False)
    
    for track in tracks:
        is_person_array = (track.boxes.cls == PERSON_CLASS) & (track.boxes.conf >= CONFIDENCE)
        track.boxes = track.boxes[is_person_array]

    counter.start_counting(img, tracks)
