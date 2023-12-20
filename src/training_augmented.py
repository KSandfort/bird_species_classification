from keras.callbacks import CSVLogger
from datetime import datetime

import tensorflow as tf
import data_loader
from architectures.inceptionV3 import get_inceptionV3_model
from architectures.alexNet import get_alexNet_model
from architectures.efficientNetB0 import get_efficientNetB0_model
import keras
import os


def add_gaussian_noise(image, stddev=0.1):
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
    noisy_image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return noisy_image


if __name__ == '__main__':

    num_epochs = 50  # Number of training epochs
    input_path = 'tf/input'  # Input path (of the images)
    output_path = 'tf/output'

    input_shape = (224, 224, 3)
    num_classes = 525

    # Load data
    train_data, test_data, val_data = data_loader.get_data(input_path, augmented=False)
    #train_data = train_data.map(lambda img: add_gaussian_noise(img, stddev=0.1))

    model_identifiers = [
        'inceptionV3',
        # 'alexNet'
        # 'efficientNetB0'
    ]

    # Run training for different models specified in model_identifiers
    for identifier in model_identifiers:

        if identifier == 'inceptionV3':
            model = get_inceptionV3_model(input_shape, num_classes, add_noise=True)
        elif identifier == 'alexNet':
            model = get_alexNet_model(input_shape, num_classes)
        elif identifier == 'efficientNetB0':
            model = get_efficientNetB0_model(input_shape, num_classes)

        augment = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),

        ])

        # Compile the model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"])

        # Get time as a part of the log file name
        now = datetime.now()
        log_name = f'training_{identifier}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.log'
        model_name = f'model_{identifier}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.keras'

        # Add logger
        if not os.path.exists(os.path.join(output_path, 'logs')):
            os.makedirs(os.path.join(output_path, 'logs'))

        log_path = os.path.join(output_path, 'logs')
        csv_logger = CSVLogger(os.path.join(log_path, log_name))

        if not os.path.exists(os.path.join(output_path, 'models')):
            os.makedirs(os.path.join(output_path, 'models'))

        history = model.fit(
            train_data,
            epochs=num_epochs,
            steps_per_epoch=len(train_data),
            validation_data=val_data,
            validation_steps=len(val_data),
            callbacks=[csv_logger]
        )

        model_path = os.path.join(output_path, 'models')
        model.save(os.path.join(model_path, model_name))
