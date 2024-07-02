import datetime
import math
import os

import numpy as np

from config import (
    IMG_RESIZE,
    PERSON_CLASS,
    BATCH_SIZE
)
from utils import (
    Predictions,
)

class Coco:
    coco_annotation_structure = {
        "info": {
            "description": "People detection",
            "year": datetime.datetime.now().year,
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d")
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": PERSON_CLASS, "name": "person", "supercategory": "person"}
        ]
    }

    @staticmethod
    def to_coco(prediction_batches: list[Predictions], img_paths: list[str]) -> dict:
        img_id = 0
        annotation_id = 0
        annotations = Coco.coco_annotation_structure.copy()
        
        for prediction_batch in prediction_batches:
            batch_boxes = prediction_batch["boxes"]
            batch_classes = prediction_batch["classes"]
            
            for boxes, classes in zip(batch_boxes, batch_classes):
                img_id += 1
                height, width = IMG_RESIZE

                annotations["images"].append({
                    "id": img_id,
                    "width": width,
                    "height": height,
                    "file_name": os.path.basename(img_paths[img_id - 1]),
                })

                for box, class_ in zip(boxes.tolist(), classes.tolist()):
                    if class_ == -1:
                        continue

                    annotation_id += 1
                    x, y, w, h = box
                    annotations["annotations"].append({
                        "id": annotation_id,
                        "category_id": class_,
                        "image_id": img_id,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                        "segmentation": [[
                            x, y,
                            x + w, y,
                            x + w, y + h,
                            x, y + h 
                        ]],
                    })
        
        return annotations

    @staticmethod
    def from_coco(annotations: dict) -> list[Predictions]:
        image_count = len(annotations["images"])
        assert image_count % BATCH_SIZE == 0, f"Image count {image_count} must be divisible by BATCH_SIZE {BATCH_SIZE}"

        chunk_amount = math.ceil(image_count / BATCH_SIZE)
        prediction_batches = [{"boxes": [], "classes": []} for _ in range(chunk_amount)]

        for i in range(image_count):
            batch_i = i // BATCH_SIZE
            prediction_batches[batch_i]["boxes"].append([])
            prediction_batches[batch_i]["classes"].append([])
        
        for annotation in annotations["annotations"]:
            batch_i = (annotation["image_id"] - 1) // BATCH_SIZE
            img_i = (annotation["image_id"] - 1) % BATCH_SIZE

            prediction_batches[batch_i]["boxes"][img_i].append(annotation["bbox"])
            prediction_batches[batch_i]["classes"][img_i].append(PERSON_CLASS)
        
        for batch_i, predictions in enumerate(prediction_batches):
            for img_i, img_value in enumerate(predictions["boxes"]):
                while len(img_value) < 100:
                    prediction_batches[batch_i]["boxes"][img_i].append([-1, -1, -1, -1])
                    prediction_batches[batch_i]["classes"][img_i].append(-1)
        
        for batch_i, predictions in enumerate(prediction_batches):
            for batch_key, batch_data in predictions.items():
                prediction_batches[batch_i][batch_key] = np.array(batch_data)

        return prediction_batches
