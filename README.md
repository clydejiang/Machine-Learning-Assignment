# Assignment 1: Linear Regression Project

## Overview

This project implements a simple linear regression model from scratch to predict the profit of a restaurant based on the population size of a city. It includes Python functions to load data, compute the cost function, and compute gradients for the model parameters using gradient descent.

### Features

- Load dataset from text files.
- Compute the cost function for linear regression.
- Compute the gradient for parameters `w` (weight) and `b` (bias).
- Implement gradient descent to minimize the cost function.


# Assignment 2: Neural Networks for Handwritten Digit Recognition, Multiclass


## Overview

This repository contains the code for a machine learning project that uses neural networks to recognize handwritten digits (0-9). The project is structured as an interactive practice lab, where users can understand and implement the key concepts of neural networks, including the ReLU activation function, the softmax function, and multiclass classification.

### Features
- Neural Network Implementation: Step-by-step guide to implementing a neural network using TensorFlow, a powerful and widely-used machine learning library.
- ReLU Activation: In-depth explanation and application of the Rectified Linear Unit (ReLU) activation function, which introduces non-linearity to the model.
- Softmax Function: Utilization of the softmax function for the output layer to interpret neural network outputs as probabilities.
- Multiclass Classification: Detailed approach to handling multiple classes in neural network outputs, allowing for the recognition of different handwritten digits.
- Interactive Exercises: The lab includes hands-on exercises that reinforce learning and understanding of neural network concepts.

# Assignment 3: Decision Tree Helpers for Mushroom Classification

## Overview

This repository contains Python functions designed to assist in the construction of a decision tree for classifying mushrooms as either edible or poisonous. The key components include entropy calculation, dataset splitting based on a feature, and information gain computation to determine the best feature for splitting at each node. These utilities form the foundation for building a decision tree from scratch, emphasizing a clear understanding of the mechanics behind decision trees.

### Features
- Entropy Calculation: A function to compute the entropy at a node, providing a measure of impurity based on the proportion of edible and poisonous examples. This is crucial for evaluating the quality of a split.
- Dataset Splitting: A utility that divides the dataset into left and right branches based on a specified feature. This is essential for recursively constructing the branches of the decision tree.
- Information Gain Computation: A mechanism to calculate the information gain from splitting on each feature, guiding the selection of the most informative feature at each node.
- Best Feature Identification: An algorithm to iterate through all features, applying the information gain computation to identify the feature that maximally reduces uncertainty (or entropy) upon splitting.

  
# Assignment 4: K-means Image Compression

## Overview

This project demonstrates the implementation and application of the K-means clustering algorithm for the purpose of image compression. By reducing the number of colors in an image to a select few that best represent the original palette, we achieve significant compression with minimal loss of quality. This repository contains Python code that guides you through the process of K-means clustering from the basics of finding the closest centroids and computing new centroid means, to applying the algorithm for compressing an image.

### Features
- K-means Clustering Implementation: Step-by-step Python implementation of the K-means algorithm, including finding the closest centroids and updating centroids based on cluster assignments.
- Visual Understanding: Code to visualize the clustering process on a 2D dataset, providing an intuitive grasp of K-means algorithm iterations and convergence.
- Random Initialization: Utilizes random initialization of centroids to enhance the algorithm's efficiency and effectiveness, avoiding the pitfalls of poor initial centroid placement.
- Image Compression Application: Demonstrates the application of K-means clustering for reducing the color palette of an image, effectively compressing the image without significant loss of visual fidelity.
- Efficient Data Representation: Compresses images by reducing the 24-bit color representation to a manageable number of colors, demonstrating a practical application of cluster analysis in data compression.

# Assignment 5: Collaborative Filtering Recommender Systems

## Overview

Collaborative filtering stands at the core of modern recommender systems, predicting user ratings for items based on the preferences of similar users and item characteristics. This implementation focuses on learning parameter vectors for both users and movies, along with user bias terms, to predict how a user might rate movies they haven't seen.

### Features
Custom Training Loop: Implements a bespoke training loop using TensorFlow's GradientTape for flexible and efficient gradient computation and parameter updates.
Dataset: Utilizes the "ml-latest-small" dataset from MovieLens, focusing on movies released since the year 2000, comprising ratings from 443 users across 4,778 movies.
Cost Function: Includes a custom cost function that incorporates both prediction error and regularization to prevent overfitting, optimizing for a balance between accuracy and model generalization.
Parameter Learning: Simultaneously learns movie feature vectors, user parameter vectors, and user biases to predict ratings, capturing the nuanced interplay between user preferences and movie features.
TensorFlow Integration: Leverages TensorFlow for its powerful automatic differentiation capabilities, streamlining the implementation of the learning algorithm.
