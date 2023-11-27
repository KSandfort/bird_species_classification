import data_loader
import keras

if __name__ == '__main__':

    # Load data
    train_data, test_data, val_data = data_loader.get_data('input')

    base_model = keras.applications.InceptionV3(include_top=False, )

    # 2. Freeze the base model
    base_model.trainable = False

    # 3. Create inputs into models
    inputs = keras.layers.Input(shape=(224, 224, 3), name="input-layer")

    # 4. Rescaling
    # x = tf.keras.layers.experimental.preprocessing.Rescaling(1/255.)(inputs)

    # 5. Pass the inputs
    x = base_model(inputs)
    print(f"Shape after passing inputs through base model: {x.shape}")

    # 6. Average pool the outputs of the base model
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    print(f"Shape after GlobalAveragePooling2D: {x.shape}")

    # 7. Create the output activation layer
    outputs = keras.layers.Dense(525, activation="softmax", name="output-layer")(x)

    # 8. Combine the inputs with outputs into a model
    model_0 = keras.Model(inputs, outputs)

    # 9. Compile the model
    model_0.compile(loss="categorical_crossentropy",
                    optimizer=keras.optimizers.Adam(learning_rate=0.01),
                    metrics=["accuracy"])

    history = model_0.fit(train_data,
                          epochs=2,
                          steps_per_epoch=len(train_data),
                          validation_data=val_data,
                          validation_steps=int(0.25 * len(val_data)), )