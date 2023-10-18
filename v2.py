import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

#===========================================================================
#===========================================================================
# Change directory
path = "/Users/hagenraasch/Documents/Python/Kaggle/Housing/"
os.chdir(path)

# Get the current working directory
cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))

#===========================================================================
# Read data
#===========================================================================
dataset = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")

#print(dataset.head())
#print(dataset.info())
# house price distribution
#print(dataset['SalePrice'].describe())
#===========================================================================
# Plot data
#===========================================================================
# Create a histogram of 'SalePrice'
# plt.hist(dataset['SalePrice'], bins=30)
# plt.xlabel('Sale Price')
# plt.ylabel('Frequency')
# plt.title('Distribution of Sale Price')
# plt.show()
# Create a box plot of 'SalePrice'
# sns.boxplot(data=dataset, y='SalePrice')
# plt.xlabel('Sale Price')
# plt.title('Box Plot of Sale Price')
# plt.show()
# Interactive Visualization Libraries:
#fig = px.histogram(dataset, x='SalePrice')
#fig.show()

# Create a histogram of numbers
#dataset_numbers.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
#dataset_numbers.hist()
# dataset_numbers = dataset.select_dtypes(include= ['int64', 'float64'])
# correlation_matrix = dataset_numbers.corr()
# dataset_numbers.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
# plt.show()
# # Create a heatmap
# sns.set(font_scale=1.0)  # Adjust the font size if needed
# plt.figure(figsize=(10, 8))  # Adjust the figure size if needed
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

# # Show the plot
# plt.show()

#dataset["SalePrice"].plot()
#dataset.plot.box()
#plt.show()
#===========================================================================
# Drop ID -> not necessary
#===========================================================================
dataset = dataset.drop('Id', axis=1)

#===========================================================================
# Prepare dataset
#===========================================================================
numeric_cols = dataset.select_dtypes(include=['int64', 'float64'])
categorical_cols = dataset.select_dtypes(include=['object']).columns

dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())
dataset[categorical_cols] = dataset[categorical_cols].fillna(dataset[categorical_cols].mode().iloc[0])
print(dataset.info())
dataset = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True)
print(dataset.info())
y = dataset['SalePrice']
X = dataset.drop('SalePrice', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train.info())
#print(X_test.info())
# Use encodings for category columns

#===========================================================================
# Create and train random forest regressor
#===========================================================================
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

#===========================================================================
# Make predictions on the test data and evaluate the model using Mean Squared Error (MSE)
#===========================================================================
y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)