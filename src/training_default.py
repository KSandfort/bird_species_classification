import data_loader
from architectures.inceptionV3 import get_inceptionV3_model
from architectures.alexNet import get_alexNet_model
import keras

if __name__ == '__main__':

    # Load data
    train_data, test_data, val_data = data_loader.get_data('tf/input')

    model_identifiers = [
        'inceptionV3',
        #'alexNet'
    ]

    input_shape = (224, 224, 3)
    num_classes = 525

    # Run training for different models specified in model_identifiers
    for identifier in model_identifiers:

        if identifier == 'inceptionV3':
            model = get_inceptionV3_model(input_shape, num_classes)
        elif identifier == 'alexNet':
            model = get_alexNet_model(input_shape, num_classes)

        # Compile the model
        model.compile(loss="categorical_crossentropy",
                        optimizer=keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["accuracy"])

        history = model.fit(train_data,
                              epochs=10,
                              steps_per_epoch=len(train_data),
                              validation_data=val_data,
                              validation_steps=int(0.25 * len(val_data)), )

