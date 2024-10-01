# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:01:06 2024

@author: Vaishnavi
"""
#       KDE plot/ Density Plot
#KDE-Kernal Density Estimate plot

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data: Load the Iris dataset
data = sns.load_dataset('iris')

# Create a density plot for the petal length
sns.kdeplot(data['petal_length'], fill=True, color='blue', alpha=0.5)

# Add title and labels
plt.title('Density Plot of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Density')

# Show the plot
plt.show()

'''
How to Read the Density Plot
X-Axis: Represents the values of the variable being analyzed (in this case, petal length).
Y-Axis: Represents the estimated density of the data points. This is not a frequency count but a probability density, meaning the area under the curve sums to 1.
Peaks: The peaks of the curve indicate the values at which the data points are concentrated. Higher peaks represent higher density, suggesting that many observations fall within that range.
Width of the Curve: A wider curve indicates more variability in the data, while a narrower curve suggests that the data points are more clustered around a central value.
Multiple Peaks: If there are multiple peaks (multimodal distribution), it indicates that the dataset may have multiple underlying groups or categories.


'''
