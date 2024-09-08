import keras
import keras_cv
from ultralytics import YOLO

from config import (
    BOUNDING_BOX_FORMAT,
    IMG_RESIZE,
    MAIN_MODEL_PATH,
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

def get_model_data_to_train() -> ModelData:
    return _get_yolov8s_ultralytics_model_data()

def get_main_model_data() -> ModelData:
    return _get_yolov8s_ultralytics_model_data_trained()

def _get_yolov8_pascalvoc_model_data() -> ModelData:
    def prepare_to_train(model: keras.Model) -> keras.Model:
        for layer in model.layers[:-20]:
            layer.trainable = False

        model.compile(
            classification_loss='binary_crossentropy',
            box_loss='ciou',
            optimizer=keras.optimizers.Adam(0.001),
            jit_compile=False,
        )

        return model

    return ModelData(
        keras_cv.models.YOLOV8Detector.from_preset(
            "yolo_v8_m_pascalvoc",
            bounding_box_format=BOUNDING_BOX_FORMAT,
            num_classes=20
        ),
        preprocess_model,
        14,
        prepare_to_train
    )

def _get_yolov8s_ultralytics_model_data() -> ModelData:
    return ModelData(
        YOLO("yolov8s.pt", task="detect"),
        preprocess_model,
        0,
        None
    )

def _get_yolov8s_ultralytics_model_data_trained() -> ModelData:
    return ModelData(
        YOLO(MAIN_MODEL_PATH, task="detect"),
        preprocess_model,
        0,
        None
    )

def main():
    model_data = _get_yolov8s_ultralytics_model_data()
    model = model_data.model

    model.summary()
    model_data.prepare_to_train()

if __name__ == "__main__":
    main()
