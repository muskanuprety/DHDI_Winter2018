import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset= pd.read_csv('Data.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

# replacing missing data; we put the mean of the column where data is null
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN", strategy="mean", axis= 0)
imputer= imputer.fit(x[:, 1:3])
x[:, 1:3]= imputer.transform(x[:, 1:3])

# changing string variables to integers through encoding
# encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelincoder_x = LabelEncoder()
x[:,0] = labelincoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features= [0])
x = onehotencoder.fit_transform(x).toarray()

labelincoder_y= LabelEncoder()
y= labelincoder_y.fit_transform(y)

