from __future__ import annotations as _future_annotations

from dataclasses import dataclass, field
import glob
import itertools
import json
import math
import os
import shutil
from typing import Iterator

import cv2
import imutils
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from coco import Coco
from config import (
    INPUT_DIR,
    DATASET_DIR,
    DATASET_ANNOTATIONS_FOLDER,
    DATASET_ANNOTATIONS_FILE,
    FORCE_DATASET_REBUILD,
    FRAMES_PER_VIDEO,
    BOUNDING_BOX_FORMAT,
    FORCE_DATASET_ANNOTATIONS_REBUILD,
    BATCH_SIZE,
    CONFIDENCE,
    IMG_RESIZE,
    PERSON_CLASS,
    TRAIN_RATIO,
)
from utils import (
    ModelData,
    Predictions,
    Labels,
)

def pad(text: int | str, width: int, fill_char: str = " ") -> str:
    return str(text).rjust(width, fill_char)

@dataclass
class Dataset:
    video_paths: list[str]
    img_paths: list[str] = field(default_factory=list)

    @staticmethod
    def create_dataset() -> Dataset:
        if FORCE_DATASET_REBUILD:
            if os.path.exists(DATASET_DIR):
                shutil.rmtree(DATASET_DIR)
        
        assert os.path.exists(INPUT_DIR), "Videos input folder not found"
        video_paths = glob.glob(os.path.join(INPUT_DIR, "**"))
        dataset = Dataset(video_paths)

        if os.path.exists(DATASET_DIR):
            print(f"Dataset already exists, skipping build")
        else:
            assert len(video_paths) > 0, f"Input folder {INPUT_DIR} is empty"
            os.mkdir(DATASET_DIR)

            for i, video_path in enumerate(video_paths, start=1):
                justify_width = len(str(len(video_paths)))
                print(f"Video {pad(i, justify_width)}/{len(video_paths)}: Extracting {FRAMES_PER_VIDEO} frames... ({video_path})")
                dataset._extract_imgs(video_path, i)

        print("Reading images...")
        dataset.img_paths = glob.glob(os.path.join(DATASET_DIR, "**", "*.jpg"), recursive=True)
        
        return dataset

    def create_dataset_annotations(self, model_data: ModelData) -> None:
        annotations_path = Dataset.get_annotations_path()

        if FORCE_DATASET_ANNOTATIONS_REBUILD:
            if os.path.exists(annotations_path):
                os.remove(annotations_path)

        if os.path.exists(annotations_path):
            print(f"Dataset annotations already exist, skipping build")
        else:
            print(f"Image batches to annotate: {math.ceil(len(self.img_paths) / BATCH_SIZE)} ({len(self.img_paths)} images)")
            self._generate_dataset_annotations(model_data)
    
    def iterate_img_batches(self, model_data: ModelData) -> Iterator[np.ndarray]:
        img_filename_chunks = self._batch(self.img_paths, BATCH_SIZE)

        for batch_id, img_paths_chunk in enumerate(img_filename_chunks, start=1):
            print(f"Iterating batch {batch_id}/{len(img_filename_chunks)}...")
            imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_paths_chunk]
            img_batch = np.array(imgs)

            if model_data.preprocess_model is not None:
                img_batch = model_data.preprocess_model(img_batch)

            assert img_batch.shape == (BATCH_SIZE, *IMG_RESIZE, 3), f"Expected image shape {(BATCH_SIZE, *IMG_RESIZE, 3)}, got {img_batch.shape}"
            yield img_batch

    def get_prediction_batches(self, filename: str | None = None) -> list[Predictions]:
        if filename is None:
            filename = Dataset.get_annotations_path()

        with open(filename) as file:
            content = file.read()
            annotations = json.loads(content)

        prediction_batches = Coco.from_coco(annotations)
        
        return prediction_batches
    
    def predict_img_batch(self, model_data: ModelData, img_batch: np.ndarray) -> Predictions:
        predictions = model_data.model.predict(img_batch)
        
        boxes = predictions["boxes"]
        classes = predictions["classes"]
        confidences = predictions["confidence"]

        class_mask = (classes == model_data.target_class) & (confidences >= CONFIDENCE)
        box_mask = np.expand_dims(class_mask, axis=-1).repeat(4, axis=-1)

        filtered_boxes = np.full_like(boxes, -1, dtype=float)
        filtered_classes = np.full_like(classes, -1, dtype=int)
        filtered_confidences = np.full_like(confidences, -1, dtype=float)
        
        filtered_boxes[box_mask] = boxes[box_mask]
        filtered_classes[class_mask] = classes[class_mask]
        filtered_confidences[class_mask] = confidences[class_mask]

        filtered_classes = np.where(filtered_classes == -1, filtered_classes, PERSON_CLASS)

        filtered_predictions = {
            "boxes": np.array(filtered_boxes),
            "classes": np.array(filtered_classes),
            "confidence": np.array(filtered_confidences),
        }

        return filtered_predictions
    
    def visualize(self, img_batch: np.ndarray, predictions: Predictions) -> None:
        img_count = img_batch.shape[0]
        rows = math.ceil(img_count ** 0.5)
        cols = math.ceil(img_count ** 0.5)

        while rows * (cols - 1) >= img_count:
            cols -= 1

        keras_cv.visualization.plot_bounding_box_gallery(
            img_batch,
            value_range=(0, 255),
            rows=rows,
            cols=cols,
            y_pred=predictions,
            scale=5,
            font_scale=0.7,
            bounding_box_format=BOUNDING_BOX_FORMAT,
        )
        plt.show()
    
    @staticmethod
    def get_annotations_path() -> str:
        return os.path.join(DATASET_ANNOTATIONS_FOLDER, DATASET_ANNOTATIONS_FILE)
    
    def load_data(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        img_paths = self.img_paths
        prediction_batches = self.get_prediction_batches()
        labels = self._parse_prediction_batches(prediction_batches)
        
        assert TRAIN_RATIO > 0 and TRAIN_RATIO < 1, "Invalid train ratio percentage"
        TEST_RATIO = 1 - TRAIN_RATIO
        img_count = len(img_paths)
        
        assert (img_count * TRAIN_RATIO) % 1 == 0, f"Expected {TRAIN_RATIO * 100}% of {img_count}"\
                                                   f" to be an integer, got {img_count * TRAIN_RATIO}"
        assert (img_count * TEST_RATIO) % 1 == 0, f"Expected {TEST_RATIO * 100}% of {img_count}"\
                                                  f" to be an integer, got {img_count * TEST_RATIO}"
        train_count = int(img_count * TRAIN_RATIO)

        def preprocess_img(img_path, labels):
            img = tf.io.read_file(img_path)
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_RESIZE)
            
            return (img, labels)

        dataset = tf.data.Dataset\
        .from_tensor_slices((img_paths, labels))\
        .shuffle(buffer_size=img_count, seed=100)\
        .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)\

        train_dataset = dataset\
        .take(train_count)\
        .batch(BATCH_SIZE, drop_remainder=True)\
        .prefetch(buffer_size=tf.data.AUTOTUNE)\

        test_dataset = dataset\
        .skip(train_count)\
        .batch(BATCH_SIZE, drop_remainder=True)\
        .prefetch(buffer_size=tf.data.AUTOTUNE)\

        return (train_dataset, test_dataset)

    def _extract_imgs(self, input_video_path: str, video_i: int) -> None:
        cap = cv2.VideoCapture(input_video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = int(total_frames / FRAMES_PER_VIDEO)

        for i in range(FRAMES_PER_VIDEO):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()

            if not ret:
                break
            
            img_path = os.path.join(DATASET_DIR, f"{video_i}-{frame_count:05d}.jpg")
            height, width = IMG_RESIZE
            resized_frame = imutils.resize(frame, width, height)
            cv2.imwrite(img_path, resized_frame)
            frame_count += max(frame_skip, 1)
        
        cap.release()
    
    def _generate_dataset_annotations(self, model_data: ModelData) -> None:
        prediction_batches: list[Predictions] = []

        for img_batch in self.iterate_img_batches(model_data):
            predictions = self.predict_img_batch(model_data, img_batch)
            prediction_batches.append(predictions)
        
        annotations = Coco.to_coco(prediction_batches, self.img_paths)

        with open(Dataset.get_annotations_path(), "w") as file:
            file.write(json.dumps(annotations, indent=4))
    
    def _batch(self, items: list, size: int) -> list[list]:
        return list(list(batch) for batch in itertools.batched(items, size))

    def _parse_prediction_batches(self, prediction_batches: list[Predictions]) -> Labels:
        boxes = []
        classes = []
        img_i = -1
        preserve_homogeneous_shape = True

        for predictions in prediction_batches:
            batch_boxes = predictions["boxes"]
            batch_classes = predictions["classes"]

            for i in range(BATCH_SIZE):
                img_i += 1
                img_boxes = batch_boxes[i]
                img_classes = batch_classes[i]
                
                if not preserve_homogeneous_shape:
                    img_boxes = img_boxes[img_boxes != -1]
                    img_boxes = img_boxes.reshape(img_boxes.shape[0] // 4, 4)
                    img_classes = img_classes[img_classes != -1]
                
                boxes.append(img_boxes.tolist())
                classes.append(img_classes.tolist())

        boxes = np.array(boxes)
        classes = np.array(classes)
        labels = {
            "boxes": boxes,
            "classes": classes,
        }

        return labels
