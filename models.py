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

preprocess_model = keras.Sequential([
    keras.layers.Input(shape=(None, None, 3)),
    keras_cv.layers.Resizing(*IMG_RESIZE, pad_to_aspect_ratio=True, bounding_box_format=BOUNDING_BOX_FORMAT),
])

def get_pretrained_model_data() -> ModelData:
    model_data = ModelData(
        _get_yolov8_pascalvoc_model(),
        preprocess_model,
        14
    )
    
    return model_data

def get_main_model() -> keras.Model:
    return _get_yolov8_pascalvoc_model()

def _get_yolov8_pascalvoc_model() -> keras.Model:
    return keras_cv.models.YOLOV8Detector.from_preset(
        "yolo_v8_m_pascalvoc",
        bounding_box_format=BOUNDING_BOX_FORMAT,
        num_classes=20
    )

def main():
    model: keras.Model = _get_yolov8_pascalvoc_model()
    print(model)
    model.summary()

if __name__ == "__main__":
    main()
