#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Load The Data

# In[2]:


df = pd.read_csv('heart.csv')


# Exploring the data

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df['target'].value_counts()


# In[7]:


sns.countplot(x='target', data=df)


# Preprocess the Data

# In[8]:


df.isnull().sum


# Correlation Heatmap

# In[9]:


plt.figure(figsize=(30,15))
sns.heatmap(df.corr(), annot=True, cmap = 'coolwarm')


# Train/Test Split

# In[10]:


from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[13]:


X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)


# In[14]:


print("Missing values in X_train:\n", X_train.isnull().sum())
print("Missing values in X_test:\n", X_test.isnull().sum())


# In[15]:


X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())


# In[16]:


print("Final check - Missing values in X_train:\n", X_train.isnull().sum().sum())
print("Final check - Missing values in X_test:\n", X_test.isnull().sum().sum())


# Logistic Regression

# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
print(classification_report(y_test, pred))


# Random Forest

# In[18]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print(classification_report(y_test, pred))


# Decision Tree

# In[19]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
print(classification_report(y_test, pred))


# K-nearest Neighbours

# In[20]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(classification_report(y_test, pred))


# Tune K in KNN

# In[21]:


error_rate = []
for k in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    error_rate.append(np.mean(pred_k != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[22]:


# Example using class_weight in Logistic Regression
logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
print(classification_report(y_test, pred))


# In[ ]:




