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