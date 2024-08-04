from dataclasses import dataclass
import json
import os
import pathlib

import keras
import numpy as np

from typing import Callable, TypedDict

@dataclass
class ModelData:
    model: keras.Model
    preprocess_model: keras.Model | None
    target_class: int
    _prepare_to_train: Callable[[keras.Model], keras.Model]

    def prepare_to_train(self) -> keras.Model:
        return self._prepare_to_train(self.model)

class Predictions(TypedDict):
    """ Represent boxes and classes for a batch of images, thus having first dimension of length BATCH_SIZE. """
    # (img, box, 4)
    boxes: np.ndarray
    # (img, class)
    classes: np.ndarray

class Labels(TypedDict):
    """ Represent boxes and classes for all images in a tf.data.Dataset. """
    # (img, box, 4)
    boxes: np.ndarray
    # (img, class)
    classes: np.ndarray

def setup() -> None:
    os.chdir(pathlib.Path(__file__).parent.resolve())
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def predictions_to_list(prediction_data: Predictions | list[Predictions]) -> list:
    predictions_list = []

    if not isinstance(prediction_data, list):
        prediction_data = [prediction_data]

    for predictions in prediction_data:
        predictions_item = {
            "boxes": predictions["boxes"].tolist(),
            "classes": predictions["classes"].tolist(),
        }
        predictions_list.append(predictions_item)
    
    return predictions_list

def main():
    predictions = {
        "boxes": np.array([
            [
                [20, 40, 10, 5]
            ]
        ]),
        "classes": np.array([
            [
                1
            ]
        ])
    }
    predictions_list = predictions_to_list(predictions)

    print(json.dumps(predictions_list, indent=4))

if __name__ == "__main__":
    main()
