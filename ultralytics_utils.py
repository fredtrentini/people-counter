import shutil
import os

import numpy as np

from config import (
    IMG_RESIZE,
    TRAIN_RATIO,
    SEED,
)
from utils import (
    Labels,
)

def generate_files(img_paths_: list[str], labels: Labels, target_class: int) -> None:
    """
    Generated file structure:

    data/
        train/
            images/
                [*img.jpg]
            labels/
                [*img.jpg.txt]
        test/
            images/
                [*img.jpg]
            labels/
                [*img.jpg.txt]
    
    dataset.yaml
    """
    img_paths = np.array(img_paths_)
    img_count = len(img_paths)
    train_count = int(img_count * TRAIN_RATIO)
    labels = {k: v.copy() for k, v in labels.items()}
    _shuffle_deterministically(img_paths, labels["boxes"], labels["classes"])

    train_labels = {k: v.copy() for k, v in labels.items()}
    train_labels["boxes"] = train_labels["boxes"][:train_count]
    train_labels["classes"] = train_labels["classes"][:train_count]

    test_labels = {k: v.copy() for k, v in labels.items()}
    test_labels["boxes"] = test_labels["boxes"][train_count:]
    test_labels["classes"] = test_labels["classes"][train_count:]

    label_map = {
        "train": {
            "images": img_paths[:train_count],
            "labels": train_labels,
        },
        "test": {
            "images": img_paths[train_count:],
            "labels": test_labels,
        },
    }

    for folder in ["train", "test"]:
        os.makedirs(os.path.join("data", folder, "images"), exist_ok=True)
        os.makedirs(os.path.join("data", folder, "labels"), exist_ok=True)
        
        images = label_map[folder]["images"]
        labels = label_map[folder]["labels"]
        
        for i, image in enumerate(images):
            image_src_filename = os.path.basename(image)
            image_dst_filename = os.path.join("data", folder, "images", image_src_filename)
            label_filename = f"{image_dst_filename}.txt".replace("images", "labels", 1)
            shutil.copyfile(image, image_dst_filename)
            
            boxes = labels["boxes"][i]
            classes = labels["classes"][i]
            label_lines = []

            for box, class_ in zip(boxes, classes):
                if box[0] == -1:
                    break

                yolo_box = _coco_to_yolo_box(box, IMG_RESIZE[1], IMG_RESIZE[0])
                label_line = f"{class_} {' '.join(f'{v}' for v in yolo_box)}"
                label_lines.append(label_line)

            with open(label_filename, "w") as file:
                file.write("\n".join(label_lines))
    
    with open("dataset.yaml", "w") as file:
        lines = [
            f"path: ./data",
            "train: train",
            "val: test",
            "",
            "names:",
            f"  {target_class}: person",
        ]

        file.write("\n".join(lines))

def _shuffle_deterministically(*arrays: np.ndarray) -> None:
    np.random.seed(SEED)
    length = len(arrays[0])
    indexes = np.random.permutation(length)

    for i, array in enumerate(arrays):
        assert len(array) == length, f"Expected numpy array {i} to have length {length}, got {len(array)}"

    for array in arrays:
        np.copyto(array, array[indexes])

def _coco_to_yolo_box(bbox, img_width, img_height) -> tuple[int, int, int, int]:
    x_min, y_min, width, height = bbox
    
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    x_center_norm = round(x_center_norm, 4)
    y_center_norm = round(y_center_norm, 4)
    width_norm = round(width_norm, 4)
    height_norm = round(height_norm, 4)
    
    return x_center_norm, y_center_norm, width_norm, height_norm