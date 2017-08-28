
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

import datetime as dt
from sklearn import linear_model
import math

#%%
import xgboost as xgb


# In[2]:

train = pd.read_csv('train.csv')


# In[3]:

test = pd.read_csv('test.csv')


# In[4]:
# Convert datetimes
test.pickup_datetime = pd.to_datetime(test.pickup_datetime, errors='coerce')
train.pickup_datetime = pd.to_datetime(train.pickup_datetime, errors='coerce')
train.dropoff_datetime = pd.to_datetime(train.dropoff_datetime, errors='coerce')

# In[ ]:

train.dtypes

# In[ ]:

train.trip_duration.hist()
plt.show()


# In[ ]:

train.loc[train.trip_duration > 50000,]
#%%
# Remove rows with excessive drip_duration
train = train.loc[train.trip_duration < 50000,]

# In[ ]:
# Check NA's.
sum(train.isnull().any(axis=1))

# In[6]:

for df in (train, test):
    
    # Since year is 2016 for ALL records in test and train, exclude it
    # df["year"] = df.pickup_datetime.dt.year
    df["month"] = df.pickup_datetime.dt.month
    df["day"] =  df.pickup_datetime.dt.day
    df["hr"]  = df.pickup_datetime.dt.hour
    df["dayofweek"] = df.pickup_datetime.dt.weekday
    # Convert Y/N to 1/0 so that it's numeric.
    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')
    # df['minute'] = df.pickup_datetime.dt.minute
    # To prevent negative predictions convert longitudes to absolute numbers... Didn't help
    # df['pickup_longitude'] = -1 * df['pickup_longitude']
    # df['dropoff_longitude'] = -1 * df['dropoff_longitude']
    # Convert angles to radians
    df.pickup_longitude = np.radians(df.pickup_longitude) + 2*math.pi
    df.pickup_latitude = np.radians(df.pickup_latitude)
    df.dropoff_longitude = np.radians(df.dropoff_longitude) + 2*math.pi
    df.dropoff_latitude = np.radians(df.dropoff_latitude)
    
    # Add new feature "time_bin" - which 15-min bin did time occur.
    df["time_bin"] = df["hr"] * 4 + df.pickup_datetime.dt.minute // 15 + 1


# In[20]:

train.head()


# In[7]:

# Slice out the duration as our response variable
y_train = train.trip_duration


# In[8]:

y_train.head()


# In[9]:

# Remove id and trip_duration from training data
# Also had to remove pickup_datetime and dropoff_datetime because linear regression didn't like datetime stamps
X_train = train.drop(labels=['id','trip_duration', 'pickup_datetime', 'dropoff_datetime', 'hr'], axis=1)


# In[ ]:

train.head()


# In[ ]:

test.head()


# In[ ]:

X_train.head()


# In[ ]:

test.dtypes


# In[10]:

# Slice and remove id's from test data. 
test_ids = test.id


# In[11]:

test_ids.shape


# In[12]:

test_ids.head()


# In[13]:

# Also remove pickup_datetime from test data.
X_test = test.drop(labels=['id', 'pickup_datetime', 'hr'], axis=1)


# In[ ]:


model = linear_model.LinearRegression()


# LinearRegression resulted in negative predictions. Try Ridge Regression (whatever the hell that is)

# In[14]:

model = linear_model.RidgeCV()


# In[15]:

model.fit(X_train, y_train)

#%%
coeffs = model.coef_
#%%
X_train.dtypes[coeffs<0]

#%%
model.alpha_
# In[16]:

predictions = np.round(model.predict(X_test), 0 )


# In[17]:

y_predict = pd.Series(predictions,name='trip_duration')


# In[18]:

y_predict


# In[19]:

# Check if any negative predictions
sum(y_predict < 0)

#%%
neg_predictions = X_test[y_predict < 0]

# In[ ]:

results = pd.concat([test_ids, y_predict], axis=1)


# In[ ]:

results.head()


# In[ ]:

# Write results to file
results.to_csv('NSG001.csv', index=None)


# 

