# -*- coding: utf-8 -*-
"""
@author: Vaishnavi
"""
#       Histogram

#Normal Distribution 
import matplotlib.pyplot as plt
import numpy as np

# Sample data 
data = np.random.randn(1000)  # 1000 random numbers from a normal distribution

# Create the histogram
plt.hist(data, bins=30, edgecolor='black')

#  titles and labels
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()

"""
bins: Defines the number of bins or intervals for the histogram (30 in this case).
edgecolor='black': Adds a border to each bar for better visibility.
plt.show(): Displays the histogram.

"""

############################################################

#  Right-Skewed Histogram


import matplotlib.pyplot as plt
import numpy as np

# Right-skewed sample data
# Creating a right-skewed distribution using exponential distribution
right_skewed_data = np.random.exponential(scale=1.0, size=1000)

# Plotting the right-skewed histogram
plt.figure(figsize=(8, 5))  # Set the figure size
plt.hist(right_skewed_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Right-Skewed Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y')  # Add grid for better readability
plt.show()


############################################################

#       left-skewed Histogram

import matplotlib.pyplot as plt
import numpy as np

# Left-skewed sample data
# Creating a left-skewed distribution using a negative exponential distribution
left_skewed_data = -np.random.exponential(scale=1.0, size=1000)

# Plotting the left-skewed histogram
plt.figure(figsize=(8, 5))  # Set the figure size
plt.hist(left_skewed_data, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title('Left-Skewed Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y')  # Add grid for better readability
plt.show()

