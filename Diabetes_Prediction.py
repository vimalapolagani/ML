#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[13]:


dataset=pd.read_csv("diabetes.csv")


# In[14]:


dataset.head(5)


# In[15]:


dataset.shape


# In[16]:


dataset.isnull().sum()


# In[17]:


dataset['Outcome'].value_counts()


# In[18]:


dataset.groupby('Outcome').mean()


# In[19]:


hist = dataset.hist(figsize = (20,20))


# In[21]:


# separating the data and labels
X = dataset.drop(columns = 'Outcome', axis=1)
Y = dataset['Outcome']
print(X,Y)


# In[26]:


scaler=StandardScaler()


# In[27]:


scaler.fit(X)


# In[28]:


standardised_data=scaler.transform(X)


# In[33]:


X = standardised_data
Y = dataset['Outcome']
print(X,Y)


# In[34]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[38]:


classifier=svm.SVC(kernel='linear')


# In[39]:


classifier.fit(X_train,Y_train)


# In[40]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_accuracy)


# In[42]:


# accuracy score on the testing data
X_test_prediction = classifier.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, Y_test)
print(testing_accuracy)


# In[43]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




