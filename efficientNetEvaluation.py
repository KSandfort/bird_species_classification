import keras

"""
This class tests the provided EfficientNet model on the dataset.
"""
if __name__ == '__main__':


    # Import data
    train_data = keras.preprocessing.image_dataset_from_directory(
        "archive/train",
        batch_size=64,
        image_size=(224, 224), shuffle=True
    )

    val_data = keras.preprocessing.image_dataset_from_directory(
        "archive/valid",
        batch_size=64,
        image_size=(224, 224), shuffle=True
    )

    test_data = keras.preprocessing.image_dataset_from_directory(
        "archive/test",
        batch_size=64,
        image_size=(224, 224), shuffle=True
    )

    #print(train_data.class_names)

    model = keras.models.load_model('models/EfficientNetB0-525-(224 X 224)- 98.97.h5', custom_objects={'F1_score': 'F1_score'})
