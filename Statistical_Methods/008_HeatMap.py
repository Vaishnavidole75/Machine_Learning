# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:29:48 2024

@author: Vaishnavi
"""

#  HeatMap 


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data (correlation matrix)
data = np.random.rand(10, 12)  # 10 rows, 12 columns of random data

# Create the heatmap
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(data, annot=True, cmap='coolwarm')  # annot=True shows values, cmap sets color scheme

# Add labels and a title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Heatmap Example')

# Show the plot
plt.show()
"""
sns.heatmap(data, annot=True, cmap='coolwarm')

data: The matrix to visualize.
annot=True: Adds the numerical values inside the heatmap.
cmap='coolwarm': Color map for visualization, ranges from cool (blue) to warm (red) colors.

"""
############################################################
# Perfect positive correlation

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create sample data 
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 4, 6, 8, 10]  # Feature2 is a multiple of Feature1
})

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Add labels and a title
plt.title('Perfect Positive Correlation Heatmap')

# Show the plot
plt.show()

# Scatter plot
plt.scatter(data['Feature1'], data['Feature2'])
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Perfect Positive Correlation (1)')
plt.show()

##################################################

# Perfect negative correlation

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create sample data 
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [10, 8, 6, 4, 2]  # Feature2 is decreasing as Feature1 increases
})

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Add labels and a title
plt.title('Perfect Negative Correlation Heatmap')

# Show the plot
plt.show()

# Scatter plot
plt.scatter(data['Feature1'], data['Feature2'])
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Perfect Negative Correlation (-1)')
plt.show()

###################################################

'''
How to Read a Heatmap
Colors:

Dark/warm colors indicate higher values; light/cool colors indicate lower values.
Annotations:

Numerical values inside cells provide exact data.
Axes:

Each axis represents different variables.
Legend:

Explains color scale and corresponding values.


1: Perfect positive correlation.
0: No correlation.
-1: Perfect negative correlation.
Heatmap Colors: The color intensity reflects the strength of the correlation:

Dark red = Strong positive correlation.
Dark blue = Strong negative correlation.
Lighter shades indicate weaker correlations.
'''