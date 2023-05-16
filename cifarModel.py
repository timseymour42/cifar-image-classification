import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert the target labels to one-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build the model
model = tf.keras.Sequential([
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
    # two dense layers are added with 512 units each. These layers provide higher-level abstractions by combining features learned 
    # from the convolutional layers. The use of ReLU activation in the dense layers helps introduce non-linearity to the model.
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # The final dense layer has the number of units equal to the number of classes in the dataset (10 in this case). 
    # The softmax activation function is used to produce probability scores for each class, enabling the model to make predictions.
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
