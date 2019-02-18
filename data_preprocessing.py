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

#we do this because python will give numbers to each country if we dont do this
# and integers can be ranked based on their values which would not be helpful here

labelincoder_x = LabelEncoder()
x[:,0] = labelincoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features= [0])
x = onehotencoder.fit_transform(x).toarray()

# since its only yes/no, binary will work here

labelincoder_y= LabelEncoder()
y= labelincoder_y.fit_transform(y)

# splitting dataset into training and test set

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:5] = sc.fit_transform(x_train[:,3:5])
x_test[:,3:5] = sc.transform(x_test[:,3:5])

