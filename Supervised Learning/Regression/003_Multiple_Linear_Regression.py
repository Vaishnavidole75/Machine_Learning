# -*- coding: utf-8 -*-
"""

@author: Vaishnavi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset (with multiple independent variables)
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])  # Independent variables
y = np.array([2, 3, 4, 5, 6])  # Dependent variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the actual vs predicted values
plt.scatter(y, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit')

# Add labels and title
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.legend()

# Show the plot
plt.show()


######################################

#  Import file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#  Load the CSV data using pandas
data = pd.read_csv('C:/17-Linear Regression/cars.csv')

# Prepare the independent and dependent variables
X = data[['HP', 'VOL', 'SP', 'WT']].values  # Independent variables
y = data['MPG'].values  # Dependent variable

#  Create and train the model
model = LinearRegression()
model.fit(X, y)

#  Make predictions
y_pred = model.predict(X)

# actual vs predicted values
plt.scatter(y, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit')

# Add labels and title
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.legend()

# Show the plot
plt.show()


