
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_csv('kc_house_data.csv')
data.head()


# In[2]:


data.columns


# In[3]:


cont = data[['price','bathrooms','waterfront','view','condition','grade','sqft_living','bedrooms','sqft_lot', 'floors','sqft_above', 'sqft_basement','sqft_living15', 'sqft_lot15']]
cont.head()
y = data['price']
x_bathroom = np.linspace(0,10,11)
x_sqftliving = np.linspace(0,20000,21)


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt


model = LinearRegression()
model.fit(data['bathrooms'][:,np.newaxis], y)

yfit = model.predict(x_bathroom[:, np.newaxis])


plt.scatter(data['bathrooms'], y)
plt.plot(x_bathroom, yfit, color = 'green')


cross_val_score(model, data['bathrooms'][:,np.newaxis], y, cv = 10).mean()


# In[5]:


from sklearn.linear_model import Ridge

model1 =Ridge(alpha= 3)
model1.fit(data['bathrooms'][:,np.newaxis], y)
yfit1 = model1.predict(x_bathroom[:, np.newaxis])
cross_val_score(model1, data['bathrooms'][:,np.newaxis], y, cv = 10).mean()


# In[6]:


from sklearn.linear_model import Lasso

model2 =Lasso(alpha= 5)
model2.fit(data['bathrooms'][:,np.newaxis], y)
yfit2 = model2.predict(x_bathroom[:, np.newaxis])
cross_val_score(model2, data['bathrooms'][:,np.newaxis], y, cv = 10).mean()


# In[7]:


model3 =Ridge(alpha= 2)
model3.fit(data['sqft_living'][:,np.newaxis], y)
yfit3 = model3.predict(x_sqftliving[:, np.newaxis])

cross_val_score(model3, data['sqft_living'][:,np.newaxis], y, cv = 10).mean()


# In[8]:


model4 =Lasso(alpha= 0.0001)
model4.fit(data['sqft_living'][:,np.newaxis], y)
yfit4 = model4.predict(x_sqftliving[:, np.newaxis])
cross_val_score(model4, data['sqft_living'][:,np.newaxis], y, cv = 10).mean()


# In[9]:


model5 =LinearRegression()
model5.fit(data['sqft_living'][:,np.newaxis], y)
yfit5 = model4.predict(x_sqftliving[:, np.newaxis])
cross_val_score(model4, data['sqft_living'][:,np.newaxis], y, cv = 10).mean()


# In[10]:


x_sqft_above = np.linspace(0,10000,11)
model.fit(data['sqft_above'][:,np.newaxis],y)
yfit_sqft_above = model.predict(x_sqft_above[:, np.newaxis])

plt.scatter(data['sqft_above'],y)
plt.plot(x_sqft_above, yfit_sqft_above, color = 'green')

cross_val_score(model,data['sqft_above'][:,np.newaxis],y, cv = 10).mean()


# In[11]:


data['sqft_living'].corr(data['sqft_living15'])


# In[12]:


x_sqftliving15 = np.linspace(0,7000, 8)

model.fit(data['sqft_living15'][:,np.newaxis], y)
yfit_sqft_living15 = model.predict(x_sqftliving15[:, np.newaxis])

plt.scatter(data['sqft_living15'],y)
plt.plot(x_sqftliving15, yfit_sqft_living15, color = 'green')
cross_val_score(model, data['sqft_living15'][:,np.newaxis], y, cv = 10).mean()


# In[13]:


x_bedroom = np.linspace(0,40, 15)

model.fit(data['bedrooms'][:,np.newaxis], y)
yfit_bedroom = model.predict(x_bedroom[:, np.newaxis])

plt.scatter(data['bedrooms'],y)
plt.plot(x_bedroom, yfit_bedroom, color = 'green')
cross_val_score(model, data['bedrooms'][:,np.newaxis], y, cv = 10).mean()


# In[55]:


import seaborn as sps

plt.figure(figsize = (16,5))
sps.heatmap(cont.corr(), annot = True,fmt= '0.2f')


# In[15]:


from sklearn.preprocessing import OneHotEncoder
O = OneHotEncoder()

zipp = data['zipcode']
zipp = O.fit_transform(zipp[:, np.newaxis]).toarray()
model.fit(zipp, y)

yfit_zip = model.predict(zipp)

cross_val_score(model, yfit_zip[:, np.newaxis], y, cv= 10).mean()


# In[27]:


from sklearn.metrics import r2_score

r2_score(data['price'].values, yfit_zip, multioutput= 'variance_weighted')


# In[42]:


final = data[['bathrooms','grade','sqft_living','sqft_above','sqft_living15']]
model = LinearRegression()
model.fit(final, y)
yfit_final = model.predict(final)
cross_val_score(model, yfit_final[:, np.newaxis],y, cv = 10).mean()


# In[54]:


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 6)
final11 = poly.fit_transform(final)
model11 = LinearRegression().fit(final11,y)
yfit_final2 = model11.predict(final11)
cross_val_score(model11, yfit_final2[:, np.newaxis], y, cv =10).mean()

