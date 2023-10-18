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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

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
numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = dataset.select_dtypes(include=['object']).columns

#print(dataset.info())
dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean(), axis=0)
#print(dataset.info())
dataset[categorical_cols] = dataset[categorical_cols].fillna(dataset[categorical_cols].mode().iloc[0])
#print(dataset.info())
dataset = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True)
#print(dataset.info())
y = dataset['SalePrice']
X = dataset.drop('SalePrice', axis=1)
#X = X[top_feature_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#print(X_train.info())
#print(X_test.info())
#===========================================================================
# Create and train random forest regressor
#===========================================================================
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=42)
#rf_regressor = RandomForestRegressor(max_depth=10, min_samples_split=10, n_estimators=300, random_state=42)
rf_regressor.fit(X_train, y_train)
#===========================================================================
# perform a scikit-learn Recursive Feature Elimination (RFE)
#===========================================================================
# here we want only 50 final feature, we do this to produce a ranking
# n_features_to_select = 50
# rfe = RFE(rf_regressor, n_features_to_select=n_features_to_select)

# rfe.fit(X_train, y_train)
#===========================================================================
# Use grid search for hyperparameter optimation
#===========================================================================
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    #'max_features': ['auto', 'sqrt', 'log2', None],
    #'bootstrap': [True, False],
    #'oob_score': [True, False],
    #'random_state': [42],  # A specific random state for reproducibility
    #'criterion': ['mse', 'mae'],
    #'n_jobs': [-1],  # Use all available CPU cores
    #'warm_start': [True, False]
    # Add other hyperparameters and their parameter space
}

grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rf_regressor = grid_search.best_estimator_

print(best_params)
print(best_rf_regressor)
#===========================================================================
# Make predictions on the test data and evaluate the model using Mean Squared Error (MSE)
#===========================================================================
y_pred = rf_regressor.predict(X_test)
# y_pred = rfe.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# sns.regplot(x=y_test, y=y_pred)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Regression Plot")
# plt.show()

#===========================================================================
# Check if some features can be omitted
#===========================================================================
# Get feature importances from the trained RandomForestRegressor
# feature_importances = rf_regressor.feature_importances_

# Get the names of the features (column names from your DataFrame)
# feature_names = X_train.columns

# Sort features by their importance
# sorted_idx = feature_importances.argsort()[::-1]
#sorted_idx.tofile("sorted_idx.csv", delimeter=',')
#sorted_idx = np.array(sorted_idx)
#np.savetxt("sorted_idx.csv",sorted_idx, delimiter=',')
#print(sorted_idx)
#print(feature_importances)
# Create a bar plot of feature importances
# plt.figure(figsize=(12, 6))
# plt.title("Feature Importances")
# plt.bar(range(X_train.shape[1]), feature_importances[sorted_idx], align="center")
# plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in sorted_idx], rotation=90)
# plt.tight_layout()
# plt.show()