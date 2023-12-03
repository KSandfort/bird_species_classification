import keras
from keras import layers
from keras import regularizers


def get_efficientNetB0_model(input_shape, num_classes):
    base_model = keras.applications.efficientnet.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling='max'
    )

    base_model.trainable = False

    model = keras.models.Sequential([
        base_model,
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        layers.Dense(
            256,
            kernel_regularizer=regularizers.l2(l=0.016),
            activity_regularizer=regularizers.l1(0.006),
            bias_regularizer=regularizers.l1(0.006), activation='relu'),
        keras.layers.Dropout(rate=0.45, seed=123),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def get_efficientNetB0_model_old(input_shape, num_classes):
    # Load the pretained model
    pretrained_model = keras.applications.efficientnet.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='max'
    )

    #pretrained_model.trainable = True

    inputs = pretrained_model.input
    x = layers.Dense(128, activation='relu')(pretrained_model.output)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.45)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
