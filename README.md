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
