import data_loader

if __name__ == '__main__':
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


    # --- All garbage from here ---
    """
    # Define a function to extract features from images using the VGG model
    def extract_features_vgg(image_path):
        img = image.load_img(image_path, target_size=(224, 224, 3))  # VGG's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocess the input

        # Get the features from a specific layer (adjust layer_name as needed)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)
        features = model.predict(img_array)
        return features.flatten()  # Flatten the features to a 1D array


    # Assuming you have a list of paths to your image files
    image_paths = ['path_to_image_1.jpg', 'path_to_image_2.jpg', ...]

    # Extract features from images
    all_features = []
    for path in image_paths:
        features = extract_features(path)
        all_features.append(features)

    # Convert the list of feature vectors to a numpy array
    all_features = np.array(all_features)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)

    # Reduce dimensionality using PCA
    pca = PCA(n_components=50)  # Adjust the number of components as needed
    reduced_features = pca.fit_transform(scaled_features)

    # Alternatively, use t-SNE for dimensionality reduction
    # tsne = TSNE(n_components=2)
    # reduced_features = tsne.fit_transform(scaled_features)

    # Apply K-means clustering
    num_clusters = 2  # Assuming you want to explore dividing into male and female
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(reduced_features)

    # Explore the clustering results
    for i in range(num_clusters):
        # Display images from each cluster for manual inspection
        cluster_i_indices = np.where(clusters == i)[0]
        # Show or analyze images in each cluster to assess if they resemble male or female
    """

    # --- Garbage ends here ---

    # --- OWN STRUCTURE, START FROM HERE!!! ----------------------------------------------------------------------------

    # Step 1: Load Data
    input_path = 'tf/input'
    #input_path = 'home/konstantin/Documents/bird_class'  # Change for container version
    train_data, test_data, val_data = data_loader.get_data(input_path)

    # Step 2: Combine Test-, Training-, Validation data into one set
    #train_unbatched = train_data.unbatch()
    #test_unbatched = test_data.unbatch()
    #val_unbatched = val_data.unbatch()

    #combined_unbatched = tf.data.Dataset.concatenate(train_data, test_data)
    #combined_unbatched = tf.data.Dataset.concatenate(combined_unbatched, val_data)

    dataset_cardinality = tf.data.experimental.cardinality(train_data).numpy()
    print("Dataset Cardinality:", dataset_cardinality)

    #combined_batched = combined_unbatched.batch(32)

    # Step 2.1: Divide set into species by class label

    # Step 3: Image pre-processing for all images

    # Step 4: Extract features in a pre-trained VGG model
    # Step 4.1 Scale, all features

    # Step 5: Run PCA on all features

    # Step 6: Apply KMeans clustering to each class

    # Step 7: Explore clustering distances
    # Step 7.1: Determine appropriate clustering distance to determine if male/female split is possible

