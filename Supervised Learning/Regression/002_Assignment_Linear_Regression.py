# -*- coding: utf-8 -*-
"""

@author: Vaishnavi

"""
# Simple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  #for plotting the graph

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


df=pd.read_csv("C:/17-Linear Regression/calories_consumed.csv")
df
df.columns

x=df['Weight gained (grams)'].values.reshape(-1, 1)  # Reshape to 2D array  #Represents the Input,features or independent variables
y=df['Calories Consumed']     #Represents the output,target variable or dependent variable that i want to predict.



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#test_size=0.2: This parameter specifies that 20% of the data will be allocated to the testing set, while the remaining 80% will be used for training
#random_state=42

#create linear regression model
model=LinearRegression()
model
#
model.fit(x_train,y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)
y_pred

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(mse)
print(r2)

# Visualize the result
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', label='Predicted')
plt.title('Simple Linear Regression')
plt.xlabel('Weight gained (grams)')
plt.ylabel('Calories Consumed')
plt.legend()
plt.show()