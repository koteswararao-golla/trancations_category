#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


df = pd.read_csv('cards_data.csv')
print(df.shape)
#from pandas_profiling import ProfileReport
#ProfileReport(df)


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
cor = df.corr()
plt.figure(figsize=(12,6))
sns.heatmap(cor,cmap='Set1',annot=True)


# In[4]:


df.columns


# In[5]:


df = df.fillna('-999')


# In[6]:


df = df.drop(['acct_type','merchant_id','as_of_date','posting_date'],axis=1)
df.shape


# In[7]:


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

X_tr, X_eval = train_test_split(df, test_size=0.3)


y_tr = X_tr.category
y_eval = X_eval.category

X_tr = X_tr.drop(columns=['category'])
X_eval = X_eval.drop(columns=['category'])

features = [col_name for col_name in X_tr.columns if col_name != 'category']
cat_features = [col_name for col_name in features if X_tr[col_name].dtype == 'object']


train_dataset = Pool(X_tr, y_tr, feature_names=list(X_tr.columns), cat_features=cat_features)

model_params = {
    'iterations': 20, 
    'loss_function': 'MultiClass', 
    'train_dir': 'crossentropy',
    'allow_writing_files': False,
    'random_seed': 42,
    'learning_rate': 0.01
}

model = CatBoostClassifier(**model_params)
model.fit(train_dataset, verbose=True, plot=True)


# In[8]:


from sklearn.metrics import roc_auc_score, roc_curve,classification_report
predict = model.predict(X_eval)
print('Test data f1 score',classification_report(y_eval,predict))


# In[9]:


df_test = pd.read_csv('cards_data_test.csv')
print(df_test.shape)


# In[10]:


df_test = df_test.fillna('-999')
df_test = df_test.drop(['acct_type','merchant_id','as_of_date','posting_date'],axis=1)

X_test = df_test.drop(columns=['category'])
X_label = df_test.category


# In[11]:


predict_test = model.predict(X_test)
print('Test data metric scores',classification_report(X_label,predict_test))

