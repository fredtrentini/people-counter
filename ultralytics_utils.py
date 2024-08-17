import shutil
import os

import numpy as np

from config import (
    TRAIN_RATIO,
    SEED,
)
from utils import (
    Labels,
)

def generate_files(img_paths_: list[str], labels: Labels) -> None:
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
    from IPython import embed
    embed()
    exit()

    for folder in ["train", "test"]:
        os.makedirs(os.path.join("data", folder, "images"), exist_ok=True)
        os.makedirs(os.path.join("data", folder, "labels"), exist_ok=True)
        
        images = label_map[folder]["images"]
        labels = label_map[folder]["labels"]
        
        for i, image in enumerate(images):
            image_src_filename = os.path.basename(image)
            image_dst_filename = os.path.join("data", folder, "images", image_src_filename)
            label_filename = f"{image_dst_filename}.txt"

            shutil.copytree(image, image_dst_filename)
            
            boxes = labels["boxes"][i]
            classes = labels["classes"][i]
            label_lines = []

            for box, class_ in zip(boxes, classes):
                if box[0][0] == -1:
                    break

                label_line = f"{class_} {box}"
                label_lines.append(label_line)

            with open(label_filename, "w") as file:
                file.write("\n".join(label_lines))

def _shuffle_deterministically(*arrays: np.ndarray) -> None:
    np.random.seed(100)
    length = len(arrays[0])
    indexes = np.random.permutation(length)

    for i, array in enumerate(arrays):
        assert len(array) == length, f"Expected numpy array {i} to have length {length}, got {len(array)}"

    for array in arrays:
        np.copyto(array, array[indexes])
