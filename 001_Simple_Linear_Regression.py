#!/usr/bin/env python
# coding: utf-8

# DataSet- California_Housing_price.csv
# 
# Using- Simple Linear Regression
# 
# #  Linear regression Algorithm

# In[3]:


import pandas as pd  # library used for data manipulation and analyze
import numpy as np  # it used for numerical operation  array,matrix,linear algebra operation
import matplotlib.pyplot as plt #Plotting graph ,visulalization

# To implement Linear regression need following libraries of sklearn(sci-kit learn)
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv("California_Housing_Prices.csv")
df


# In[4]:


#displaying some initial rows of dataset
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[20]:


X=df[['median_income']]
y=df['median_house_value']


# In[21]:


# Splitting the data into training and test sets (80% train, 20% test)
#training test 80%
#testing data 0.2 means 20%

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[22]:


# Creating a Linear Regression model
model=LinearRegression()


# In[29]:


# Training the model on the training data
model.fit(x_train,y_train)


# In[30]:


# Predicting the house values for the test data
y_pred=model.predict(x_test)


# In[31]:


#Calculating Mean squared erroe

mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)


# In[32]:


mse


# In[33]:


rmse


# In[39]:


#plotting regression line

plt.scatter(x_test, y_test, color="blue", label="Actual values")
plt.plot(x_test,y_pred,color="red",label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("Madian Income")
plt.ylabel("Median House Value")
plt.legend()
plt.show()


# In[40]:


# Display RMSE
print(f"Root Mean Squared Error (RMSE): {rmse}")


# In[42]:


#Above graph show Positive correlation:As median income increases, house prices also rise
#Rmse =84209.01241414454 :have average predication error

#above code for simple Linear Regression (which means 1 feature is used ) to predict price
#To improve the accuracy, we can use to multiple linear regression, which uses multiple features from the dataset to make predictions.

