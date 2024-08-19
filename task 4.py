#!/usr/bin/env python
# coding: utf-8

# In[ ]:


spam.csv


# In[43]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')


# In[44]:


df = pd.read_csv('spam.csv')
df.head()


# In[45]:


df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['label', 'message']


# In[46]:


df.head()


# In[47]:


df.info()


# In[48]:


df.describe()


# In[49]:


df.dtypes


# In[50]:


df.isnull()


# In[51]:


df.isnull().sum()


# In[52]:



  >>> import nltk
  >>> nltk.download('stopwords')


# In[56]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)  # Adjust 'max_features' as needed

# Transform the cleaned messages into TF-IDF features
X = tfidf.fit_transform(df['message']).toarray()

# Labels (Spam or Not Spam)
y = df['label'].apply(lambda x: 1 if x == 'spam' else 0).values


# In[57]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


X_train


# In[60]:


y_test


# In[61]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)


# In[62]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)


# In[63]:


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[64]:


print("Classification Report:\n", classification_report(y_test, y_pred))


# In[65]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:




