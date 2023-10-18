import numpy as np
import os
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from operator import itemgetter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder


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
features = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'LotShape',
    'LandContour', 'LotConfig', 'Neighborhood', 'HouseStyle', 'OverallQual',
    'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'BsmtQual',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
    'CentralAir', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
    'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    'ScreenPorch', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition',
    'LandSlope'
]

dataset = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")
X_train = dataset.loc[:, features]
#X_train = X_train.drop(columns=["Id", "SalePrice"])

print(X_train)
column_types = X_train.dtypes
print(column_types)
#X_test = dataset_test.drop(columns=["Id"])
#X_test = X_test[features]
X_test = dataset_test.loc[:, features]
y_train = dataset["SalePrice"].astype('int')
print(X_test)
#===========================================================================
# simple preprocessing: imputation; substitute any 'NaN' with mean value
#===========================================================================
# Calculate the mean for numeric columns
numeric_columns = X_train.select_dtypes(include=['number'])
numeric_columns_test = X_train.select_dtypes(include=['number'])
numeric_column_means = numeric_columns.mean()
numeric_column_means_test = numeric_columns.mean()

# Replace missing values in numeric columns with the mean
X_train[numeric_columns.columns] = X_train[numeric_columns.columns].fillna(numeric_column_means)
X_test[numeric_columns.columns] = X_test[numeric_columns.columns].fillna(numeric_column_means)
#X_train = X_train.fillna(X_train.mean())
#X_test  = X_test.fillna(X_test.mean())

#===========================================================================
# further preprocessing - Encode categorical columns using Label Encoding
#===========================================================================
#encoder = OneHotEncoder()
#X_train_encoded = encoder.fit_transform(X_train[['CategoricalFeature']])
label_encoder = LabelEncoder()
#columns_to_encode = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
columns_to_encode = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'BsmtExposure', 'RoofStyle', 'BsmtFinType1', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'BsmtQual', 'SaleType', 'SaleCondition']

#X_train_encoded = encoder.fit_transform(X_train[['CategoricalFeature']])

#X["categorical_column_name"] = label_encoder.fit_transform(X["categorical_column_name"])
# Encode the string values in the specified columns
for column in columns_to_encode:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.fit_transform(X_test[column])

#    X_train[column] = X_train[column].astype('int')

# Check for missing values in the entire DataFrame
missing_values = X_train.isna().sum()
missing_valuesY = y_train.isna().sum()
#print(missing_valuesY)
# Print the missing values count for each column
# Loop through each column and print its name and missing value count
#for column_name, missing_count in missing_values.items():
#    print(f'{column_name}: {missing_count}')
# Calculate the mode for categorical columns
#categorical_columns = X_train.select_dtypes(include=['object'])
#categorical_column_modes = categorical_columns.mode().iloc[0]

# Replace missing values in categorical columns with the mode
#X_train[categorical_columns.columns] = X_train[categorical_columns.columns].fillna(categorical_column_modes)


# Verify that all missing values have been replaced
#missing_values_after_fill = X_train.isna().sum()
#for column_name, missing_count in missing_values_after_fill.items():
#    print(f'{column_name}: {missing_count}')

X_train['LotFrontage'] = X_train['LotFrontage'].astype('int')
X_train['MasVnrArea'] = X_train['MasVnrArea'].astype('int')
X_train['GarageYrBlt'] = X_train['GarageYrBlt'].astype('int')

X_test['LotFrontage'] = X_test['LotFrontage'].astype('int')
X_test['MasVnrArea'] = X_test['MasVnrArea'].astype('int')
X_test['GarageYrBlt'] = X_test['GarageYrBlt'].astype('int')

#===========================================================================
# Scale features
#===========================================================================
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#print(X_train_scaled)
min_max_scaler = preprocessing.MinMaxScaler()
print(X_train)
#names = X_train.columns.to_list()
#for name in names:
#    print(name)
#column_types = X_train.dtypes
#print(column_types)
min_max_scaler.fit(X_train)
#X_train_scaled = min_max_scaler.fit_transform(X_train)
#X_test_scaled = min_max_scaler.fit_transform(X_test)
X_train_scaled = min_max_scaler.transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)
#===========================================================================
# RandomForestRegressor 
#===========================================================================
regressor = RandomForestRegressor(n_estimators=100, max_depth=10)

#===========================================================================
# perform a scikit-learn Recursive Feature Elimination (RFE)
#===========================================================================
# here we want only 50 final feature, we do this to produce a ranking
n_features_to_select = 50
rfe = RFE(regressor, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

#===========================================================================
# now print out the features in order of ranking
#===========================================================================
#features = X_train.columns.to_list()
#feature_list = []
#for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):
#    print(x, y)
#    feature_list.append(y)
#print(type(feature_list))
#===========================================================================
# use the model to predict the prices for the test data
#===========================================================================
predictions = rfe.predict(X_test)
#print(predictions)

#===========================================================================
# define the neural network
#===========================================================================
model = tf.keras.Sequential([
    # input layer
    tf.keras.layers.Input(shape=51),
    tf.keras.layers.Dense(51, activation='relu'), #vllt sigmoid besser
    tf.keras.layers.Dense(51, activation='relu'),
    tf.keras.layers.Dense(51, activation='relu'),
    tf.keras.layers.Dense(51, activation='relu'),
 #   tf.keras.layers.Dense(1000, activation='relu'),
#    tf.keras.layers.Dense(1000, activation='relu'),
 #   tf.keras.layers.Dense(1, activation='sigmoid')
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
#model.compile(optimizer='sgd',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#===========================================================================
# Convert pandas array into numpy array
#===========================================================================
#X_train  = X_train_scaled[features]
#X_train = X_train_scaled.loc[:, features]

X_train = np.array(X_train_scaled)
y_train = np.array(y_train)
#X_test = X_test_scaled[features]
X_test_submission = np.array(X_test)
#print(type(X))
#print(type(y))
#print(X_test_submission)
#X_train = X[:10]
#y_train = y[:10]
#X_train = X_train[:,0:10]

#===========================================================================
# Generate model
#===========================================================================
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_train, y_train, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
#print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#hist = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.3)  # Anzahl der Epochen und Batch-Größe anpassen
hist = model.fit(X_train, Y_train,
          batch_size=128, epochs=1000,
          validation_data=(X_val, Y_val))
model.summary()
#===========================================================================
# use the model to predict the prices for the test data
#===========================================================================
#predictions2 = model.predict(X_test)
predictions3 = model.predict(X_test_submission)
#print(predictions2)
print(predictions3)
print(predictions)
#===========================================================================
# write out CSV submission file
#===========================================================================
#output = pd.DataFrame({"SalePrice1":predictions.flatten(), "SalePrice2":predictions2.flatten()})
#output.to_csv('submission.csv', index=False)

#print('MAE:', metrics.mean_absolute_error(predictions, predictions2))  
#print('MSE:', metrics.mean_squared_error(predictions, predictions2))  
#print('RMSE:', np.sqrt(metrics.mean_squared_error(predictions, predictions2)))

# print('MAE:', metrics.mean_absolute_error(predictions3, predictions))  
# print('MSE:', metrics.mean_squared_error(predictions3, predictions))  
# print('RMSE:', np.sqrt(metrics.mean_squared_error(predictions3, predictions)))

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper right')
# plt.show()