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
