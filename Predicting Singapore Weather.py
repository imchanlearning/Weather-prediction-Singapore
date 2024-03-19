#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:/Users/chant/Downloads/M890081new.csv')


# In[3]:


df.head()


# In[4]:


df.columns = [x.lower() for x in df.columns]


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


# since minimum relative humidity dtypes is object, change it to numeric.zdxs
df['minimum relative humidity'] = pd.to_numeric(df['minimum relative humidity'], errors='coerce')


# In[8]:


#fill missing columns with average values
# Fill null values in 'minimum relative humidity' column with the rounded mean value (to two decimal places)
df['minimum relative humidity'] = df['minimum relative humidity'].fillna(round(df['minimum relative humidity'].mean(), 2))


# In[9]:


df['minimum relative humidity']


# In[10]:


plt.figure(figsize=(10, 6))
corr=df.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')


# In[11]:


plt.figure(figsize=(12, 6))
df['t_absolute_maximum'].plot(marker='o',linestyle='',label="t_absolute_maximum")
df['t_absolute_minimum'].plot(marker='o',linestyle='',label="t_absolute_minimum")
plt.grid(True)
plt.ylabel('temperature')
plt.title('maximum and minimum temperature')
plt.legend()
plt.show()


# In[12]:


plt.figure(figsize=(10, 6))
plt.title('Relationship betweenrainy_days and average sunshine')
sns.lineplot(df,y='number_of_rainy_days',x='bright_sunshine_daily_mean')


# In[13]:


# our target column
df['t_mean']=(df['t_absolute_maximum']+df['t_absolute_minimum'])/2


# In[14]:


def features_engineer(df):
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df['year'] = df['date'].dt.year
    
    return df


# In[15]:


df = features_engineer(df)
df = df.drop('date',axis=1)
df.head()


# In[16]:


int_cols = ["number_of_rainy_days","year","month"]
numeric_cols = [x for x in df.columns if x not in int_cols +["t_mean"]]
target = "t_mean"


# In[17]:


plt.figure(figsize=(15, 8))
sns.lineplot(x='year',y='t_mean',data=df)
plt.title("average temperature vs year")
plt.show()


# In[18]:


#scaling numeric columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(df[numeric_cols])
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# In[20]:


df[numeric_cols].head()


# In[21]:


X = df[numeric_cols + int_cols]
y = df[target]


# In[23]:


from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:


model=XGBRegressor(n_estimators=200,early_stopping_rounds=50,eval_metric='mae')
model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)],  # Validation data
          verbose=True)


# In[26]:


features_data=pd.DataFrame(data=model.feature_importances_,
             index=model.feature_names_in_,
             columns=['importances'])


# In[27]:


features_data.sort_values('importances').plot(kind='barh',title='feature importance')


# In[28]:


predictions=model.predict(X_test)


# In[29]:


from sklearn.metrics import mean_absolute_error


# In[30]:


mean_absolute_error(y_test,predictions)


# In[31]:


from sklearn.ensemble import RandomForestRegressor


# In[32]:


rf_model=RandomForestRegressor(n_estimators=100,random_state=42)


# In[33]:


rf_model.fit(X_train,y_train)


# In[34]:


new_predictions=rf_model.predict(X_test)


# In[35]:


mean_absolute_error(y_test,new_predictions)


# In[36]:


plt.figure(figsize=(12,6))
plt.scatter(y_test,predictions,alpha=0.5)
plt.plot(y_test,y_test,'r')
plt.title('Actual vs. Predicted Values for RandomForestRegressor model')
plt.grid(True)


# In[37]:


plt.figure(figsize=(12,6))
plt.scatter(y_test,new_predictions,alpha=0.5)
plt.plot(y_test,y_test,'r')
plt.title('Actual vs. Predicted Values for XGBoost Model')
plt.grid(True)

