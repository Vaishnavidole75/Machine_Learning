# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:14:31 2024

@author: lk
"""
#   QQ Plot (Quantile-Quantile Plot)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate sample data from a normal distribution
data = np.random.normal(0, 1, 1000)

# Create a QQ plot
stats.probplot(data, dist="norm", plot=plt)
'''scipy.stats provides the probplot function
 for generating the QQ plot.
 
 dist="norm" argument specifies that we are 
 comparing the data to a normal distribution.
 
 '''

# Add title and labels
plt.title('QQ Plot of Normally Distributed Data')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

# Show the plot
plt.show()

'''
How to Read the QQ Plot
X-Axis (Theoretical Quantiles): Represents the quantiles of the theoretical distribution (normal distribution in this case).
Y-Axis (Sample Quantiles): Represents the quantiles of the dataset you are analyzing.
Reference Line: This is the diagonal line on the plot. If the data is normally distributed, most of the points will fall along this line.


'''
