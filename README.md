# Bird Species Classification

## Introduction
- The model training is supposed to run inside a docker container environment
- An extensive amount of speed-up is possible with a designated graphics card
- You need a supported NVIDIA GPU for multithreaded training

To start the container, run:

```docker-compose up --build```

To free space of old containers, run:

```docker system prune --volumes```

## How to use the code:

**Important** Before you get started, make sure you download the images of the dataset
and place the min the corresponding target folder. (tf/input)
The dataset is way too large for GitHub.

There are several files in this project serving different purposes:

- training_default.py -> Execute for training without image pre-processing and augmentation
- training_augmented.py -> Execute for training with image pre-processing and augmentation
- logs_to_plot.py -> Creates plots based off model training logs
- model_evaluation.py -> Evaluates test accuracy and loss on a model
- clustering.ipynb -> Interactive clustering jupyter notebook