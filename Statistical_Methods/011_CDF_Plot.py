# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:41:52 2024

@author: Vaishnavi
"""
#Cumulative Distribution Function (CDF) Plot


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate some random data (normal distribution)
data = np.random.randn(1000)

# Sort data for CDF plot
sorted_data = np.sort(data)
# Cumulative probabilities
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

# Plot CDF
plt.figure(figsize=(8,6))
plt.plot(sorted_data, cdf, marker='.', linestyle='none')
plt.title('Cumulative Distribution Function (CDF) Plot')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()

'''
How to Read a CDF Plot
X-axis (Value): Represents the values of the variable you're plotting. The values are sorted in ascending order.

Y-axis (Cumulative Probability): Represents the cumulative probability, which is the proportion of data points that are less than or equal to a given value. The Y-axis ranges from 0 to 1 (or 0% to 100%).

Understanding the Shape:

A steep curve means that the data is concentrated in a narrow range of values. For instance, a steep increase in a small X-axis interval indicates that many data points are located within that interval.
A gradual curve indicates that the data is more spread out, with no sharp concentration of values.

'''

