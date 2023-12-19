import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.manifold import TSNE
from PIL import Image
import os

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

import data_loader

def extract_features(image):
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)
    print(f'Image shape: {len(image)}')
    print(f'Image shape: {len(image[0])}')
    print(f'Image shape: {len(image[0][0])}')
    features = model.predict(image[0])
    return features.flatten()

if __name__ == '__main__':
    # Step 1: Load Data
    input_path = 'tf/input'

    #input_path = 'home/konstantin/Documents/bird_class'  # Change for container version
    train_data, test_data, val_data = data_loader.get_data_unbatched(input_path)

    x_by_species = []

    # Step 2: Combine Test-, Training-, Validation data into one set


    # Step 3: Image pre-processing for all images

    # Step 4: Extract features in a pre-trained VGG model
    all_features = []
    #for img in train_data:
    for i in range(5):
        features = extract_features(train_data[i])
        all_features.append(features)



    # Convert to np array
    all_features = np.array(all_features)

    print(all_features)

    # Step 4.1 Scale (standardise) all features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)

    # Step 5: Run PCA on all features
    num_components = 2
    pca = PCA(n_components=num_components)
    reduced_features = pca.fit_transform(scaled_features)
    print(len(reduced_features))
    print(len(reduced_features[0]))

    #plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    #plt.show()

    # Step 6: Apply KMeans clustering to each class

    clusters = KMeans(n_clusters=2, n_init="auto").fit(reduced_features)
    y_kmeans = clusters.predict(reduced_features)
    # Retrieve scores (SSE)
    score = clusters.score(reduced_features)
    # Retrieve cluster centers
    cntr = clusters.cluster_centers_
    # Create subplots
    X = reduced_features
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=5, cmap='magma')
    plt.scatter(cntr[:, 0], cntr[:, 1], c='black', s=80, alpha=0.8)
    plt.show()

    # Step 7: Explore clustering distances
    # Step 7.1: Determine appropriate clustering distance to determine if male/female split is possible

