# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
feature_matrix = dataset.iloc[:, :-1].values
dependant_variable = dataset.iloc[:, 3].values 

# Taking care of the missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(feature_matrix[:, 1:3])
feature_matrix[:, 1:3] = imputer.transform(feature_matrix[:, 1:3])

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_feature_matrix = LabelEncoder()
feature_matrix[:, 0] = labelencoder_feature_matrix.fit_transform(feature_matrix[:, 0])

# Dummy Encoding
onehotencoder = OneHotEncoder(categorical_features= [0])
feature_matrix = onehotencoder.fit_transform(feature_matrix).toarray()

# Encoding the dependant variable
labelencoder_dependant_variable = LabelEncoder()
dependant_variable = labelencoder_dependant_variable.fit_transform(dependant_variable)

# Splitting the data into the Training set and Test set
from sklearn.cross_validation import train_test_split
feature_matrix_train, feature_matrix_test, dependant_variable_train, dependant_variable_test = train_test_split(feature_matrix, dependant_variable, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler() 
feature_matrix_train = standard_scaler.fit_transform(feature_matrix_train)
feature_matrix_test = standard_scaler.transform(feature_matrix_test)


