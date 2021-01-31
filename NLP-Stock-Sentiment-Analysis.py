#!/usr/bin/env python
# coding: utf-8

# ### Stock Sentiment Analysis using News Headlines

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
import re


# In[2]:


os.chdir('E:\\prasad\\practice\\NLP\\dataset')


# In[3]:


df=pd.read_csv('Data.csv',encoding='ISO-8859-1')


# In[4]:


df.shape


# In[5]:


df.isnull().sum().sum()


# In[6]:


df=df.dropna()


# In[7]:


df.isnull().sum().sum()


# In[8]:


df.head(1)


# In[9]:


train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# ### Use Train Dataset

# In[10]:


df=train.iloc[:,2:27]
df.shape


# In[11]:


df.head(1)


# ### Removing punctuations

# In[12]:


df.replace('[^a-zA-Z]',' ',regex=True,inplace=True)
df.head(1)


# ### Renaming column names for ease of access

# In[13]:


list1=[i for i in range(0,25)]
print(list1)


# In[14]:


new_index=[str(i) for i in list1]
print(new_index)


# In[15]:


df.columns=new_index
df.head(1)


# ### Convertng headlines to lower case

# In[16]:


for i in new_index:
    df[i]=df[i].str.lower()
df.head(1)    


# ### Join Headlines

# In[17]:


' '.join(str(i) for i in df.iloc[1,0:25])


# In[18]:


headlines=[]
for row in range(0,len(df.index)):
    headlines.append(' '.join(str(i) for i in df.iloc[row,0:25]))


# In[19]:


headlines[3]


# ### Convert into BagOfWords

# In[20]:


import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# In[21]:


cv=CountVectorizer(ngram_range=(2,2))


# In[22]:


X_train=cv.fit_transform(headlines)
X_train.shape


# In[23]:


y_train=train['Label']
y_train.shape


# ### Create Function for Model Building

# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[25]:


def check_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_train)
    print('Accuracy Score:',accuracy_score(y_train,y_pred),'\n')
    print('Confusion Matrix:')
    print(confusion_matrix(y_train,y_pred),'\n')
    print('Classification Report:')
    print(classification_report(y_train,y_pred))


# In[26]:


# LogisticRegression
# check_model(LogisticRegression(),X_train,y_train)


# In[27]:


# RandomForestClassifier
# check_model(RandomForestClassifier(n_estimators=500),X_train,y_train)


# In[28]:


# SVC
# check_model(SVC(),X_train,y_train)


# In[29]:


# DecisionTreeClassifier
# check_model(DecisionTreeClassifier(),X_train,y_train)


# In[30]:


MultinomialNB
check_model(MultinomialNB(),X_train,y_train)


# In[31]:


# # KNeighborsClassifier
# check_model(KNeighborsClassifier(),X_train,y_train)


# ### Select MultinomialNB Model(Accuracy Score: 1.0)

# In[32]:


mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_pred=mnb.predict(X_train)
accuracy_score(y_train,y_pred)


# ### Model Evaluation Using Test Dataset

# In[33]:


test.shape


# In[34]:


test.head(1)


# In[35]:


test_headline=[]
for i in range(0,len(test.index)):
    test_headline.append(' '.join(str(x) for x in test.iloc[i,2:27]))


# In[36]:


test_headline[1]


# In[37]:


for i in range(0,len(test_headline)):
    text=re.sub('[^a-zA-Z]',' ',test_headline[i])
    text=text.lower()
    text=text.split()
    test_headline[i]=' '.join(text)


# In[38]:


test_headline[1]


# In[39]:


X_test=cv.fit_transform(test_headline)
X_test.shape


# In[40]:


y_test=test['Label']
y_test.shape


# In[41]:


mnb.fit(X_test,y_test)
y_pred_test=mnb.predict(X_test)
accuracy_score(y_test,y_pred_test)


# ### Save Model Using Pickle & Joblib

# In[42]:


import pickle,joblib


# In[43]:


pickle.dump(mnb,open('stock_senti_pkl','wb'))


# In[44]:


joblib.dump(mnb,'stock_senti_jbl')


# ### Load Pickle Model

# In[45]:


model_pkl=pickle.load(open('stock_senti_pkl','rb'))


# In[46]:


y_pred_pkl=model_pkl.predict(X_test)


# In[47]:


accuracy_score(y_test,y_pred_test)


# ### Load Joblib Model

# In[48]:


model_jbl=joblib.load('stock_senti_jbl')


# In[49]:


y_pred_jbl=model_jbl.predict(X_test)


# In[50]:


accuracy_score(y_test,y_pred_jbl)


# In[ ]:




