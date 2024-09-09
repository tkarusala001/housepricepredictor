
House Price Prediction

This project demonstrates a basic house price prediction model using linear regression. It utilizes the Boston Housing dataset to predict house prices based on various features such as the number of rooms, crime rate, and more.

Project Overview

In this repository, you'll find a Python script that performs the following tasks:

Load and Explore Data: Loads the Boston Housing dataset, performs exploratory data analysis (EDA), and visualizes the data.
Preprocess Data: Prepares the data for training by splitting it into features and target variables and then into training and test sets.
Build and Train Model: Initializes a linear regression model, trains it on the training data, and makes predictions.
Evaluate Model: Evaluates the model's performance using metrics like Mean Squared Error (MSE) and R-squared, and visualizes predictions vs. true values.
Save and Load Model: Saves the trained model to a file and demonstrates how to load it.
Getting Started

Prerequisites
Python 3.x
Pip (Python package installer)
Installation
Clone this repository to your local machine:


git clone https://github.com/tkarusala001/housepricepredictor.git
cd housepricepredictor

Install the required Python libraries:

pip install numpy pandas scikit-learn matplotlib seaborn joblib

Running the Script
To run the house price prediction script, execute the following command in your terminal or command prompt:


python house_price_prediction.py

This script will:
Load the Boston Housing dataset.
Perform exploratory data analysis.
Train a linear regression model.
Evaluate the model's performance.
Save the model to a file named house_price_model.pkl.
Model Loading
To load the saved model and make predictions, use the following code snippet:


import joblib

# Load the model
model = joblib.load('house_price_model.pkl')
print("Model loaded successfully!")
Files in This Repository

house_price_prediction.py: The main script that performs all the tasks described.


README.md: This file.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

The Boston Housing dataset is part of the Scikit-learn library.
This project is intended for educational purposes.
