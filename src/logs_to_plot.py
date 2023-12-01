import os
import numpy as np
import matplotlib.pyplot as plt

def create_logs():
    with open('tf/output/logs/training_alexNet_2023_11_27_16_16_10.log', 'r') as f:
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
        # Accuracy

        plt.plot(log_values[:, 0], log_values[:, 1], label='Training Accuracy')
        plt.plot(log_values[:, 0], log_values[:, 3], label='Validation Accuracy')
        plt.legend()
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(output_path, 'accuracy.png'))

        plt.figure()
        plt.plot(log_values[:, 0], log_values[:, 2], label='Training Loss')
        plt.plot(log_values[:, 0], log_values[:, 4], label='Validation Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(output_path, 'loss.png'))


if __name__ == '__main__':
    create_logs()


