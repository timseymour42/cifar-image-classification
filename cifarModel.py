import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
import ssl
import numpy as np
from scipy import stats
ssl._create_default_https_context = ssl._create_unverified_context

# Refer to cifarDataPrep notebook for step-by-step explanation of data preparation
# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert the target labels to one-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build the baseline model
base_model = tf.keras.Sequential([
    # Convolutional layers are crucial for capturing spatial patterns in images. The chosen configuration 
    # consists of two pairs of Conv2D layers. The first pair has 32 filters, and the second pair has 64 filters. 
    # Each Conv2D layer uses a 3x3 filter size, which is a common choice. These layers apply convolution operations 
    # to the input image, extracting relevant features and learning representations that are useful for classification.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # Max pooling is used to downsample the feature maps obtained from the convolutional layers. It helps reduce the
    # spatial dimensions while retaining the most important information. In this model, MaxPooling2D layers with a 
    # 2x2 pool size are added after each pair of Conv2D layers to progressively reduce the spatial dimensions.
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Dropout layers help prevent overfitting by randomly setting a fraction of input units to 0 during training. 
    # The dropout rate of 0.25 is chosen in this model, meaning 25% of the units are randomly dropped during each training step. 
    # Dropout acts as a regularization technique, promoting the model's ability to generalize and reducing the likelihood of overfitting.
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    # The Flatten layer is used to convert the 2D feature maps from the previous layer into a 1D feature vector. 
    # This flattening operation is necessary to connect the convolutional layers to the subsequent dense (fully connected) layers.
    tf.keras.layers.Flatten(),
    # Dense layers are fully connected layers where each neuron is connected to every neuron in the previous layer. In this model, 
    # there is one dense layer are added with 512 units. This layer provides higher-level abstractions by combining features learned 
    # from the convolutional layers. The use of ReLU activation in the dense layer helps introduce non-linearity to the model.
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # The final dense layer has the number of units equal to the number of classes in the dataset (10 in this case). 
    # The softmax activation function is used to produce probability scores for each class, enabling the model to make predictions.
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Building a transfer learning model to improve upon the baseline model
# Load the VGG16 model pre-trained on ImageNet
# By setting include_top=False, we exclude the fully connected layers at the top of the VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the pre-trained layers
for layer in vgg_model.layers:
    layer.trainable = False

vgg16_model = tf.keras.Sequential([
    vgg_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

def run_model(model, batch_size=64, epochs=10):
    # Compile the model
    # adam optimizer: 
    # - adaptive learning rate: adjusts the step size for each parameter based on that parameter's performance
    # - momentum optimization: helps avoid local minima, achieve faster convergence
    # categorical crossentropy: 
    # - takes the predicted probabilities and the true class labels and calculates a single number that 
    #   represents the dissimilarity between them
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    # batch_size:
    # - batches refer to subsets of the training data that are processed together during the training of a neural network. 
    #   Instead of feeding all the training examples at once, the data is divided into smaller groups or batches, and the 
    #   model is updated based on the average gradients calculated from each batch.
    # - Standard practice is to choose a power-of-2 batch size
    # epoch:
    # - a complete iteration over the entire training dataset
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    return model

    

base_model_trained = run_model(base_model)
vgg16_model_trained = run_model(vgg16_model)

# Evaluate base model
base_model_loss, base_model_accuracy = base_model_trained.evaluate(X_test, y_test)
print("Base Model:")
print(f"Loss: {base_model_loss:.4f}")
print(f"Accuracy: {base_model_accuracy:.4f}")

# Evaluate VGG16 model
vgg16_model_loss, vgg16_model_accuracy = vgg16_model_trained.evaluate(X_test, y_test)
print("VGG16 Model:")
print(f"Loss: {vgg16_model_loss:.4f}")
print(f"Accuracy: {vgg16_model_accuracy:.4f}")

# Compare model performances
print("\nModel Performance Comparison:")
if base_model_accuracy > vgg16_model_accuracy:
    print("Base model performs better.")
elif base_model_accuracy < vgg16_model_accuracy:
    print("VGG16 model performs better.")
else:
    print("Both models have the same accuracy.")