import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator


def get_data(directory, augmented=False):
    train_dir = f'{directory}/train'
    test_dir = f'{directory}/test'
    val_dir = f'{directory}/valid'

    # Rescale
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # data transfer from directories to batches
    if augmented:
        # Data Pre-processing

        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_data = data_generator.flow_from_directory(
            directory=train_dir,
            batch_size=32,
            target_size=(224, 224),
            class_mode='categorical'  # Adjust according to your task (binary, categorical, etc.)
        )
    else:
        train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   batch_size=32,
                                                   target_size=(224, 224),
                                                   class_mode="categorical")

    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 batch_size=32,
                                                 target_size=(224, 224),
                                                 class_mode="categorical")

    val_data = valid_datagen.flow_from_directory(directory=val_dir,
                                                 batch_size=32,
                                                 target_size=(224, 224),
                                                 class_mode="categorical")

    return train_data, test_data, val_data


def get_data_unbatched(directory):
    train_dir = f'{directory}/train'
    test_dir = f'{directory}/test'
    val_dir = f'{directory}/valid'

    # Rescale
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # data transfer from directories to batches
    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   batch_size=1,
                                                   target_size=(224, 224),
                                                   class_mode="categorical")

    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 batch_size=1,
                                                 target_size=(224, 224),
                                                 class_mode="categorical")

    val_data = valid_datagen.flow_from_directory(directory=val_dir,
                                                 batch_size=1,
                                                 target_size=(224, 224),
                                                 class_mode="categorical")

    return train_data, test_data, val_data
