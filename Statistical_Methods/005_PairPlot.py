# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:40:55 2024

@author: lk
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data: Iris dataset
data = sns.load_dataset('iris')

# Create a pair plot
sns.pairplot(data, hue='species', markers=['o', 's', 'D'])

# Show the plot
plt.show()

"""hue='species' colors the points based 
on the species of the iris flower, making 
it easier to differentiate between categories
"""



'''
How to Read the Pair Plot
Diagonal Plots: The diagonal of the pair plot shows the distribution of each variable. In the case of continuous variables, it typically shows histograms or kernel density estimates (KDE).
Off-Diagonal Plots: The plots off the diagonal are scatter plots that show the relationship between pairs of variables. Each point represents an observation in the dataset.
Colors: Different colors represent different categories (here, species). This helps visualize how each species relates to the measured variables.
Correlation: Look for trends in the scatter plots:
A positive slope indicates a positive correlation.
A negative slope indicates a negative correlation.
No discernible pattern may suggest no correlation.
Outliers: Points that are distant from the cluster of points may indicate outliers.


'''