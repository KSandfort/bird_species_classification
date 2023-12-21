import keras
import tensorflow as tf


def add_gaussian_noise(image, stddev=0.1):
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
    noisy_image = tf.clip_by_value(image + noise, 0.0, 1.0)  # Clip values between 0 and 1
    return noisy_image


def get_inceptionV3_model(input_shape, num_classes, add_noise=False):

    # Define input layer (224, 224, 3)
    inputs = keras.layers.Input(shape=input_shape, name="input-layer")

    # If the add_noise flag is activated, add this after the input layer
    if add_noise:
        inputs = keras.layers.Lambda(lambda x: add_gaussian_noise(x, stddev=0.1))(inputs)

    # Load pre-trained base model
    base_model = keras.applications.InceptionV3(include_top=False)
    base_model.trainable = False

    # Add inputs in front of the base model
    x = base_model(inputs)

    # Average pool the outputs of the base model
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

    # Create output layer
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output-layer")(x)

    # Combine model inputs and outputs
    model = keras.Model(inputs, outputs)

    return model
