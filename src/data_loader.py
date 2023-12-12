from keras_preprocessing.image import ImageDataGenerator

def get_data(dir):
    train_dir = f'{dir}/train'
    test_dir = f'{dir}/test'
    val_dir = f'{dir}/valid'

    # Rescale
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # data transfer from directories to batches
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


def get_data_unbatched(dir):
    train_dir = f'{dir}/train'
    test_dir = f'{dir}/test'
    val_dir = f'{dir}/valid'

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
