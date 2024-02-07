Linear Regression Project
Overview
This project implements a simple linear regression model from scratch to predict the profit of a restaurant based on the population size of a city. It includes Python functions to load data, compute the cost function, and compute gradients for the model parameters using gradient descent.

Features
Load dataset from text files.
Compute the cost function for linear regression.
Compute the gradient for parameters w (weight) and b (bias).
Implement gradient descent to minimize the cost function.
Getting Started
Prerequisites
Python 3.x
NumPy library
Ensure you have Python installed on your machine. NumPy can be installed using pip:

bash
Copy code
pip install numpy
Installation
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/your-github-username/linear-regression-project.git
cd linear-regression-project
Dataset
The dataset is stored in the data folder:

ex1data1.txt - Dataset containing the population of various cities and the profit of a restaurant in each city.
ex1data2.txt - (Optional) Another dataset, if your project extends to multiple variables.
Running the Code
Start Jupyter Notebook in the project directory:
bash
Copy code
jupyter notebook
Open the notebook file (e.g., Linear_Regression.ipynb) and run the cells to load the data, compute the cost, and perform gradient descent.
Project Structure
load_data(file_path): Function to load the dataset from a file.
compute_cost(x, y, w, b): Function to compute the cost function for linear regression.
compute_gradient(x, y, w, b): Function to compute the gradient of the cost function with respect to parameters w and b.
Linear_Regression.ipynb: Jupyter Notebook containing the implementation and demonstration of the linear regression model.
Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
