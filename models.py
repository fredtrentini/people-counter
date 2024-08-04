import keras
from keras import layers
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
    return _get_yolov8_pascalvoc_model_data()

def get_main_model_data() -> ModelData:
    return _get_yolov8_pascalvoc_model_data()

def get_model_datas() -> list[ModelData]:
    return [
        _get_yolov8_pascalvoc_model_data(),
        # _get_another_model_data(),
    ]

def _get_yolov8_pascalvoc_model_data() -> ModelData:
    def prepare_to_train(base_model: keras.Model) -> keras.Model:
        domain_model_layers = [
            keras.Input(shape=(IMG_RESIZE[0], IMG_RESIZE[1], 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Dropout(0.2),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(2, activation='softmax')
        ]

        # base_model.compile(
        #     classification_loss='binary_crossentropy',
        #     box_loss='ciou',
        #     optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
        #     jit_compile=False,
        # )

        base_model.trainable = False
        model = base_model

        for domain_model_layer in domain_model_layers:
            model = domain_model_layer(model)

        model.compile(
            loss='binary_crossentropy',
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
        PASCALVOC_PERSON_CLASS,
        prepare_to_train
    )

def _get_another_model_data() -> ModelData:
    return ModelData(
        keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_xs_backbone_coco",
        ),
        preprocess_model,
        0,
        None
    )

def main():
    model_datas = get_model_datas()
    model_data = model_datas[0]
    model = model_data.model

    model.summary()
    model_data.prepare_to_train()
    print(len(model_data.model.layers))

if __name__ == "__main__":
    main()
