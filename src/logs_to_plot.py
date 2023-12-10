import os
import numpy as np
import matplotlib.pyplot as plt

def create_logs():

    log_path = 'tf/output/logs/training_inceptionV3_2023_12_4_12_59_49.log'

    with open(log_path, 'r') as f:
        # Read data from log file
        f.readline()  # Remove first line (header)
        lines = [line.rstrip() for line in f]
        rows = []
        for line in lines:
            rows.append(line.split(','))
        log_values = np.array(rows, dtype=np.float64)
        print(log_values)

        # Create plots
        output_path = '../results/plots'

        # Extract model name
        name_info = log_path.split('_')
        model_name = name_info[1]

        # Accuracy
        plt.plot(log_values[:, 0], log_values[:, 1], label='Training Accuracy')
        plt.plot(log_values[:, 0], log_values[:, 3], label='Validation Accuracy')
        plt.legend()
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(output_path, f'accuracy_{model_name}.png'))

        # Loss
        plt.figure()
        plt.plot(log_values[:, 0], log_values[:, 2], label='Training Loss')
        plt.plot(log_values[:, 0], log_values[:, 4], label='Validation Loss')
        plt.legend()
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(output_path, f'loss{model_name}.png'))


if __name__ == '__main__':
    create_logs()


