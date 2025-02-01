# Classifier Code 

import os
import nibabel as nib
import numpy as np
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras import layers, models

def classify_subfolders(main_directory):
    """Classify subfolders based on the presence of l_a1.nii.gz file."""
    classifications = {}  # Dictionary to store classifications
    for subfolder in os.listdir(main_directory):
        subfolder_path = os.path.join(main_directory, subfolder)
        if os.path.isdir(subfolder_path):
            label = 1 if "l_a1.nii.gz" in os.listdir(subfolder_path) else 0
            classifications[subfolder] = label
    return classifications

def normalize_mri_zscore(volume):
    """Normalize the volume using Z-score normalization."""
    mean = np.mean(volume)
    std = np.std(volume)
    return ((volume - mean) / std).astype("float32")

def resize_volume(img):
    """Resize across z-axis to (16, 256, 256)."""
    depth_factor = 16 / img.shape[-1]
    width_factor = 256 / img.shape[0]
    height_factor = 256 / img.shape[1]
    img = ndimage.rotate(img, 90, reshape=False)
    return ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

def load_and_preprocess_data(main_directory):
    """Load and preprocess images from the train, validation, and test sets."""
    data, labels, subfolder_names = [], [], []
    classifications = classify_subfolders(main_directory)  # Get classifications for all subfolders

    # Load and preprocess images
    for subfolder, label in classifications.items():
        subfolder_path = os.path.join(main_directory, subfolder)
        image_files = [f for f in os.listdir(subfolder_path) if f.endswith('.nii.gz')]
        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            try:
                image = nib.load(image_path).get_fdata()
                image = normalize_mri_zscore(image)
                processed_image = resize_volume(image) #crop the image here randomly
                data.append(processed_image)
                labels.append(label)
                subfolder_names.append(subfolder)
            except nib.filebasedimages.ImageFileError:
                continue  # Skip invalid NIfTI images

    data = np.expand_dims(np.array(data), axis=-1)
    labels = np.array(labels)
    return data, labels, subfolder_names

def train_and_save_model(main_directory):
    """Train the CNN model on train, validate using validation set, and save the model periodically."""
    # Load data for train, validation, and test
    train_dir = os.path.join(main_directory, 'train')
    val_dir = os.path.join(main_directory, 'validation')
    test_dir = os.path.join(main_directory, 'test')

    # Preprocess train, validation, and test data
    X_train, y_train, _ = load_and_preprocess_data(train_dir)
    X_val, y_val, _ = load_and_preprocess_data(val_dir)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16)

    # Build a simple Convolutional Neural Network (CNN)
    model = models.Sequential([
        layers.Input(shape=(16, 256, 256, 1)),
        layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Custom training loop with model saving
    epochs = 20
    save_epochs = [1, 10, 20]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.fit(train_dataset, validation_data=val_dataset, epochs=1)

        # Save the model periodically
        if (epoch + 1) in save_epochs:
            model_save_path = os.path.expanduser(f"~/Desktop/saved_models/model_epoch_{epoch + 1}.keras")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            model.save(model_save_path)
            print(f"Model saved at {model_save_path}")

    # Predict on the validation set
    predictions = model.predict(val_dataset)

    # Print predictions for each subfolder in the validation set
    subfolder_predictions = {}
    _, _, val_subfolders = load_and_preprocess_data(val_dir)
    for subfolder, true_label, predicted_prob in zip(val_subfolders, y_val, predictions.flatten()):
        if subfolder not in subfolder_predictions:
            subfolder_predictions[subfolder] = {"true_labels": [], "predicted_probs": []}
        subfolder_predictions[subfolder]["true_labels"].append(true_label)
        subfolder_predictions[subfolder]["predicted_probs"].append(predicted_prob)

    # Calculate final predicted label for each subfolder based on the average prediction probability
    for subfolder, values in subfolder_predictions.items():
        avg_predicted_prob = np.mean(values["predicted_probs"])
        predicted_label = 1 if avg_predicted_prob > 0.5 else 0
        true_label = int(np.mean(values["true_labels"]))  # Since all true labels should be the same, take the average
        print(f"Subfolder: {subfolder}, True Label: {true_label}, Predicted Label: {predicted_label}")

# Example usage of the function
split_data_path = os.path.expanduser("/Users/lynnkaram/Desktop/split_data")
train_and_save_model(split_data_path)
