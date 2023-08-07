# Autores: Allan Rodrigo Remedi Amantino e João Vitor Silva Gomes

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_image(img_array):
    """
    Preprocess the input image array.

    Args:
        img_array (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    # Normalization
    img_array = img_array.astype('float32')
    img_array /= 255.0

    return img_array

def load_images_and_labels(data_folder):
    """
    Load images and labels from a specified data folder.

    Args:
        data_folder (str): Path to the data folder.

    Returns:
        tuple: A tuple containing preprocessed images and labels as NumPy arrays.
    """
    images = []
    labels = []
    for person_name in os.listdir(data_folder):
        person_folder = os.path.join(data_folder, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(person_folder, filename)
                    img = load_img(img_path, target_size=(64, 64))
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)  # Correção aqui
                    images.append(img_array)
                    labels.append(person_name)
    return np.array(images), np.array(labels)

def create_model(input_shape, num_classes):
    """
    Create a VGG16-based model.

    Args:
        input_shape (tuple): Input shape of the model.
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.models.Model: The created model.
    """
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=500):
    """
    Train the model.

    Args:
        model (tensorflow.keras.models.Model): The model to train.
        X_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation images.
        y_val (numpy.ndarray): Validation labels.
        epochs (int): Number of epochs to train.

    Returns:
        tuple: A tuple containing the trained model and training history.
    """
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
    return model, history

def plot_results(history, y_test, label_encoder, best_model, X_test):
    """
    Plot the results including accuracy, predicted labels, and sample images.

    Args:
        history (tensorflow.keras.callbacks.History): Training history.
        y_test (numpy.ndarray): Test labels.
        label_encoder (sklearn.preprocessing.LabelEncoder): Label encoder.
        best_model (tensorflow.keras.models.Model): Best model.
        X_test (numpy.ndarray): Test images.
    """
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print("Accuracy of the best model:", accuracy)

    results = grid_search.cv_results_
    print("Grid Search results:")
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"Hyperparameter Combination: {params}, Mean Accuracy: {mean_score}")

    n_test_images = 3
    fig, axes = plt.subplots(1, n_test_images, figsize=(12, 4))
    for i, (test_image, test_label) in enumerate(zip(X_test[:n_test_images], y_test[:n_test_images])):
        test_label = label_encoder.inverse_transform([test_label])[0]
        predicted_label = label_encoder.inverse_transform([best_model.predict(np.expand_dims(test_image, axis=0))[0]])[0]

        axes[i].imshow(test_image.astype(np.uint8))
        axes[i].set_title(f"True: {test_label}\n Predicted: {predicted_label}")
        axes[i].axis('off')
    plt.show()

if __name__ == '__main__':
    data_folder = "data"

    print("Loading images and labels...")
    images, labels = load_images_and_labels(data_folder)
    print("Loaded images:", images.shape)
    print("Loaded labels:", labels.shape)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    print("Creating the classifier model using transfer learning...")
    input_shape = X_train.shape[1:]
    model = KerasClassifier(build_fn=create_model, input_shape=input_shape, num_classes=num_classes)

    param_grid = {'num_classes': [num_classes]}

    print("Training the complete model...")
    best_model, history = train_model(model, X_train, y_train, X_val, y_val)

    print("Performing grid search with cross-validation...")
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    plot_results(history, y_test, label_encoder, best_model, X_test)
