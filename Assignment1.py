
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data", header = None, names=["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price",])
data.to_csv("data.csv")
data = pd.read_csv("data.csv")

data.replace(to_replace="?", value= np.NaN, inplace= True)
data[["symboling","normalized-losses","wheel-base","length","width","height","curb-weight","engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price",]] = data[["symboling","normalized-losses","wheel-base","length","width","height","curb-weight","engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price",]].astype('float')
#data.replace(to_replace=-9999, value= np.NaN, inplace= True)

data['bore'].fillna(value= data['bore'].mean(), inplace= True )
data['stroke'].fillna(value= data['stroke'].mean(), inplace= True )
data['horsepower'].fillna(value= data['horsepower'].mean(), inplace= True )
data['peak-rpm'].fillna(value= data['peak-rpm'].mean(), inplace= True )
data['normalized-losses'].fillna(value= data['normalized-losses'].mean(), inplace= True )

data.dropna(axis=0, subset= ['num-of-doors'],inplace= True)


data.head(10)
    




# plt.hist(x,binn)
# data["horsepower"].value_counts()
# categories = pd.cut(data["horsepower"],binn)
# categories = categories.to_frame()


# In[2]:


type_price = pd.DataFrame()

type_price[['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda','isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
                               'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche',
                               'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo',]] = pd.get_dummies(data['make'])

price= data['price']
type_price['price'] = price

type_price.head(10)


# In[3]:


data['price'].describe()


# In[19]:


make_df = pd.DataFrame({'make': data['make'],'price': data['price']})
make_df.head(10)
group = make_df.groupby('make').mean()
group


# In[5]:


plt.plot(group, group.index)


# In[6]:


data['fuel-type'].unique()


# In[7]:


data[['gas','diesel']] = pd.get_dummies(data['fuel-type'])
data.drop(columns= 'fuel-type', inplace=True)


# In[8]:


data["aspiration"].unique()


# In[9]:


data[["aspiration -std",'aspiration -turbo']] = pd.get_dummies(data["aspiration"])
data.drop(columns= 'aspiration', inplace=True)
data.head()


# In[10]:


data[["Two-doors",'Four-doors']] = pd.get_dummies(data["num-of-doors"])
data[['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible']] = pd.get_dummies(data["body-style"])
data[['4wd', 'fwd', 'rwd']] = pd.get_dummies(data["drive-wheels"])
data[['front-engine', 'rear-engine']] = pd.get_dummies(data["engine-location"])
data[['engine-type-dohc', 'engine-type-dohcv', 'engine-type-l', 'engine-type-ohc', 'engine-type-ohcf', 'engine-type-ohcv', 'engine-type-rotor']] = pd.get_dummies(data["engine-type"])
data['num-of-cylinders'].replace(to_replace = ['four','six','five','three','twelve','two','eight'], value=[4,6,5,3,12,2,8], inplace= True)


data.drop(columns= 'num-of-doors', inplace=True)
data.drop(columns= 'body-style', inplace=True)
data.drop(columns= 'drive-wheels', inplace=True)
data.drop(columns= 'engine-location', inplace=True)
data.drop(columns= 'engine-type', inplace=True)

data.head()


# In[11]:


data['make'].value_counts()


# In[12]:


power = data.groupby('make')['horsepower','engine-size','highway-mpg','city-mpg'].median()
power


# In[13]:


plt.plot(power, power.index, label = power.index)


# In[14]:


maxx = data.groupby('make')['price'].max()
minn = data.groupby('make')['price'].min()

tablee = pd.DataFrame({'make':data['make'].unique(),'maximum price':maxx, 'minimum price':minn,})
tablee


# In[24]:


binn=np.linspace(0,60000.0,9)
binn
# make_df['price'] = pd.cut(make_df['price'], binn)
# make_df
analysis = make_df.groupby('make')
# analysis['price'] = pd.cut(analysis['price'],binn)
analysis

