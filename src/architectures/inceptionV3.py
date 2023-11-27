import keras
from keras import layers


def get_inceptionV3_model(input_shape, num_classes):
    # shape should be (224, 224, 3)
    base_model = keras.applications.InceptionV3(include_top=False, )
    # Freeze the base model
    base_model.trainable = False

    inputs = keras.layers.Input(shape=input_shape, name="input-layer")
    x = base_model(inputs)
    print(f"Shape after passing inputs through base model: {x.shape}")

    # Average pool the outputs of the base model
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    print(f"Shape after GlobalAveragePooling2D: {x.shape}")

    # Create the output activation layer
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output-layer")(x)

    # Combine model
    model = keras.Model(inputs, outputs)

    return model
