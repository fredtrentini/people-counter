import keras
import keras_cv
from ultralytics import YOLO

from config import (
    BOUNDING_BOX_FORMAT,
    IMG_RESIZE,
    MAIN_MODEL_PATH_PASCALVOC,
    MAIN_MODEL_PATH_ULTRALYTICS,
)
from utils import (
    ModelData
)

preprocess_model = keras.Sequential([
    keras.layers.Input(shape=(None, None, 3)),
    keras_cv.layers.Resizing(*IMG_RESIZE, pad_to_aspect_ratio=True, bounding_box_format=BOUNDING_BOX_FORMAT),
])

def get_pretrained_model_data() -> ModelData:
    return _get_yolov8_pascalvoc_model_data()

def get_main_model_data() -> ModelData:
    return _get_yolov8s_ultralytics_model_data_trained()

def _get_yolov8_pascalvoc_model_data() -> ModelData:
    return ModelData(
        keras_cv.models.YOLOV8Detector.from_preset(
            "yolo_v8_m_pascalvoc",
            bounding_box_format=BOUNDING_BOX_FORMAT,
            num_classes=20
        ),
        preprocess_model,
        14,
    )

def _get_yolov8s_ultralytics_model_data() -> ModelData:
    return ModelData(
        YOLO("yolov8s.pt", task="detect"),
        preprocess_model,
        0,
    )

def _get_yolov8s_pascalvoc_model_data_trained() -> ModelData:
    model_data = _get_yolov8_pascalvoc_model_data()
    model_data.model = keras.models.load_model(MAIN_MODEL_PATH_PASCALVOC)

    return model_data

def _get_yolov8s_ultralytics_model_data_trained() -> ModelData:
    return ModelData(
        YOLO(MAIN_MODEL_PATH_ULTRALYTICS, task="detect"),
        preprocess_model,
        0,
    )

def main():
    model_data = _get_yolov8s_ultralytics_model_data()
    model = model_data.model
    model.summary()

if __name__ == "__main__":
    main()
