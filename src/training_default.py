from keras.callbacks import CSVLogger
from datetime import datetime

import data_loader
from architectures.inceptionV3 import get_inceptionV3_model
from architectures.alexNet import get_alexNet_model
from architectures.efficientNetB0 import get_efficientNetB0_model
import keras
import os


if __name__ == '__main__':

    num_epochs = 50                 # Number of training epochs
    input_path = 'tf/input'         # Input path (of the
    output_path = 'tf/output'

    input_shape = (224, 224, 3)
    num_classes = 525

    # Load data
    train_data, test_data, val_data = data_loader.get_data(input_path)

    model_identifiers = [
        #'inceptionV3',
        #'alexNet'
        'efficientNetB0'
    ]

    # Run training for different models specified in model_identifiers
    for identifier in model_identifiers:

        if identifier == 'inceptionV3':
            model = get_inceptionV3_model(input_shape, num_classes)
        elif identifier == 'alexNet':
            model = get_alexNet_model(input_shape, num_classes)
        elif identifier == 'efficientNetB0':
            model = get_efficientNetB0_model(input_shape, num_classes)

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
