import keras
from keras import layers


def get_efficientNetB0_model(input_shape, num_classes):
    # Load the pretained model
    pretrained_model = keras.applications.efficientnet.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='max'
    )

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = layers.Dense(128, activation='relu')(pretrained_model.output)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.45)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
