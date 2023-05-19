# CIFAR-10 Image Classification with TensorFlow

This repository contains code and resources for an image classification project using TensorFlow on the CIFAR-10 dataset. The goal of this project is to explore different model architectures and techniques for image classification and provide a tutorial for beginners to learn from.

## Project Overview

The CIFAR-10 dataset is a popular benchmark dataset for image classification tasks. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes include objects such as airplanes, automobiles, birds, cats, and more.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/timseymour42/cifarClassification.git


Install the required dependencies. 

## Model Architectures

This project explores two main model architectures: a baseline model and a transfer learning model using the VGG-16 network.

- The baseline model is built using TensorFlow's Sequential API and consists of convolutional, pooling, dropout, and dense layers. It serves as a starting point for understanding the fundamentals of image classification models.

- The transfer learning model utilizes the pre-trained VGG-16 model as a feature extractor. The earlier layers of the VGG-16 network are frozen, while the later layers are fine-tuned for the CIFAR-10 dataset. This approach leverages the pre-trained model's ability to capture high-level features and helps improve performance.

## Results and Comparison

The performance of the baseline model and the transfer learning model are compared based on metrics such as loss and accuracy. The results are presented in a table in the accompanying Medium article (link below).

## Resources

- Medium Article: [CIFAR-10 Image Classification with TensorFlow](link-to-medium-article)
- CIFAR-10 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

## Contributing

Contributions to this project are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to open an issue or submit a pull request.
