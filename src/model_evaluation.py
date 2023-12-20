import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import data_loader
from sklearn.metrics import f1_score

if __name__ == '__main__':
    # Load the saved model
    model_path = 'tf/output/models/model_inceptionV3_2023_12_11_9_0_25.keras'
    model = load_model(model_path)

    # Load data
    input_path = 'tf/input'
    train_data, test_data, val_data = data_loader.get_data_unbatched(input_path)

    # Preprocess your test data if needed (e.g., normalization)

    # Evaluate the model on test data
    evaluation = model.evaluate(test_data)

    # Print the evaluation metrics (e.g., loss and accuracy)
    print("Test Loss:", evaluation[0])
    print("Test Accuracy:", evaluation[1])

    y_pred = model.predict(test_data)
    y_true = np.array(test_data)[:, 1, 0]

    f1score = f1_score(y_true, y_pred)
    print("F1 Score:", f1score)
