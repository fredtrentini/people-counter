from __future__ import annotations as _future_annotations

import keras
import keras_cv

from config import (
    BOUNDING_BOX_FORMAT,
    IMG_RESIZE,
)
from utils import (
    ModelData
)

PASCALVOC_PERSON_CLASS = 14

preprocess_model = keras.Sequential([
    keras.layers.Input(shape=(None, None, 3)),
    keras_cv.layers.Resizing(*IMG_RESIZE, pad_to_aspect_ratio=True, bounding_box_format=BOUNDING_BOX_FORMAT),
])

def get_pretrained_model_data() -> ModelData:
    return ModelData(
        _get_yolov8_pascalvoc_model(),
        preprocess_model,
        PASCALVOC_PERSON_CLASS,
    )

def get_main_model_data() -> ModelData:
    return ModelData(
        _get_yolov8_pascalvoc_model(),
        None,
        PASCALVOC_PERSON_CLASS,
    )

def get_model_datas() -> list[ModelData]:
    return [
        _get_yolov8_pascalvoc_model(),
    ]

def _get_yolov8_pascalvoc_model() -> keras.Model:
    return keras_cv.models.YOLOV8Detector.from_preset(
        "yolo_v8_m_pascalvoc",
        bounding_box_format=BOUNDING_BOX_FORMAT,
        num_classes=20
    )

def main():
    model: keras.Model = _get_yolov8_pascalvoc_model()
    print(model.name)
    model.summary()

if __name__ == "__main__":
    main()
