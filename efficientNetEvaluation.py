import keras

"""
This class tests the provided EfficientNet model on the dataset.
"""
if __name__ == '__main__':


    # Import data
    train_data = keras.preprocessing.image_dataset_from_directory(
        'input/train',
        batch_size=64,
        image_size=(224, 224),
        shuffle=True,
        labels='inferred'
    )

    val_data = keras.preprocessing.image_dataset_from_directory(
        'input/valid',
        batch_size=64,
        image_size=(224, 224),
        shuffle=True,
        labels='inferred'
    )

    test_data = keras.preprocessing.image_dataset_from_directory(
        'input/test',
        batch_size=64,
        image_size=(224, 224),
        shuffle=True,
        labels='inferred'
    )

    #print(test_data)

    #print(train_data.class_names)

    model = keras.models.load_model('models/EfficientNetB0-525-(224 X 224)- 98.97.h5', custom_objects={'F1_score': 'F1_score'})

    predictions = model.predict(test_data)
    print(predictions)
    print(predictions.shape)
