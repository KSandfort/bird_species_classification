from tensorflow.keras.models import load_model
import data_loader

if __name__ == '__main__':
    # Load the Saved Model
    model_path = 'tf/output/models/model_inceptionV3_2023_12_21_0_7_7_augmented.keras'
    model = load_model(model_path)

    # Load Data
    input_path = 'tf/input'
    train_data, test_data, val_data = data_loader.get_data_unbatched(input_path)

    # Evaluate the Model on Test Data
    evaluation = model.evaluate(test_data)

    # Print Test Accuracy and Loss
    print("Test Loss:", evaluation[0])
    print("Test Accuracy:", evaluation[1])

