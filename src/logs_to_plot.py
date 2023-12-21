import os
import numpy as np
import matplotlib.pyplot as plt


def create_logs():

    # Change path to save it to your desired log file (will be created during runtime)
    log_path = 'tf/output/logs/training_inceptionV3_2023_12_21_0_7_7_augmented.log'

    # Read data from log file
    with open(log_path, 'r') as f:
        f.readline()  # Remove header
        lines = [line.rstrip() for line in f]
        rows = []
        for line in lines:
            rows.append(line.split(','))
        log_values = np.array(rows, dtype=np.float64)
        print(log_values)

        # Define output path
        output_path = '../results/plots'

        # Extract model name
        name_info = log_path.split('_')
        model_name = name_info[1]

        # Create Accuracy Plot
        plt.plot(log_values[:, 0], log_values[:, 1], label='Training Accuracy')
        plt.plot(log_values[:, 0], log_values[:, 3], label='Validation Accuracy')
        plt.legend()
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(output_path, f'accuracy_{model_name}.png'))

        # Create Loss Plot
        plt.figure()
        plt.plot(log_values[:, 0], log_values[:, 2], label='Training Loss')
        plt.plot(log_values[:, 0], log_values[:, 4], label='Validation Loss')
        plt.legend()
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(output_path, f'loss_{model_name}.png'))


if __name__ == '__main__':
    create_logs()
