#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# In[16]:


df = pd.read_csv('tesla.csv')
df.head()


# In[17]:


df.shape


# In[18]:


df.describe()


# In[19]:


df.info()


# In[20]:


plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close Price', fontsize=15)
plt.ylabel('Price in dollars')
plt.show()


# In[21]:


df[df['Close'] == df['Adj Close']].shape


# In[22]:


df = df.drop(['Adj Close'],axis=1)


# In[23]:


df.isnull().sum()


# In[24]:


features = ['Open','High','Low','Close','Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.distplot(df[col])
plt.show()


# In[25]:


features = ['Open','High','Low','Close','Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.boxplot(df[col])
plt.show()


# In[26]:


splitted = df['Date'].str.split('/',expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')
df.head()


# In[27]:


df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()


# In[30]:


data_grouped = df.drop('Date',axis=1).groupby('year').mean()
plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open','High','Low','Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
    plt.show()


# In[33]:


df.drop('Date',axis=1).groupby('is_quarter_end').mean()


# In[34]:


df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# In[35]:


plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()


# In[36]:


plt.figure(figsize=(10, 10)) 
 
sb.heatmap(df.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()


# In[37]:


features = df[['open-close','low-high','is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train,X_valid,Y_train,Y_valid = train_test_split(features,target,test_size=0.1,random_state=2022)
print(X_train.shape,X_valid.shape)


# In[40]:


models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()


# In[41]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.show()


# In[ ]:




