import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image

_DATASET_FOLDER = "neural/dataset"
_class_folders = {
    "0" : 0x0,
    "1" : 0x1,
    "2" : 0x2,
    "3" : 0x3,
    "4" : 0x4,
    "5" : 0x5,
    "6" : 0x6,
    "7" : 0x7,
    "8" : 0x8,
    "9" : 0x9,
    "add" : -0xa,
    "dec" : -0xb,
    "div" : -0xc,
    "eq" : -0xd,
    "mul" : -0xe,
    "sub" : -0xf,
}
_images = []
_labels = []
for class_folder in _class_folders.items():
    print(class_folder)
    class_folder_path = os.path.join(_DATASET_FOLDER, class_folder[0])
    for file_name in os.listdir(class_folder_path):
        file_path = os.path.join(class_folder_path, file_name)

        # Open and preprocess the image
        image = Image.open(file_path)
        image = image.resize(
            (28, 28)
        )  # Resize the image to match your model's input shape
        image = np.array(image.convert("L")) / 255.0  # Normalize pixel values between 0 and 1
        _images.append(image)
        _labels.append(
            class_folder[1]
        )  # You can assign class labels based on the folder names or any other criteria

# Convert the lists to NumPy arrays
_images = np.array(_images)
_labels = np.array(_labels)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_combined_dataset = np.concatenate((x_train, _images), axis=0)
y_combined_dataset = np.concatenate((y_train, _labels), axis=0)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_combined_dataset, y_combined_dataset, test_size=0.2, random_state=42
)

# Normalize the pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape the images to match the model's input shape
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert the labels to categorical (one-hot encoding)
num_classes = 16
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Create TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Batch and shuffle the datasets
batch_size = 32
train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
