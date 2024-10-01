# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 09:34:45 2024

@author: lk
"""

#       Scatter Plot

# positive trend or correleation
import matplotlib.pyplot as plt

# Sample data for Positive Trend
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]  # y increases as x increases

# Create the scatter plot
plt.scatter(x, y)

# Add labels and a title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot: Positive Trend')

# Show the plot
plt.show()

#######################################################3

# Negative trend or correleation
import matplotlib.pyplot as plt

# Sample data for Negative Trend
x = [1, 2, 3, 4, 5]
y = [10, 8, 6, 4, 2]  # y decreases as x increases

# Create the scatter plot
plt.scatter(x, y)

# Add labels and a title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot: Negative Trend')

# Show the plot
plt.show()

#############################################################

#No Trend /Crreleation

import matplotlib.pyplot as plt

# Sample data for No Correlation
x = [1, 2, 3, 4, 5]
y = [5, 2, 9, 1, 6]  # Random y values with no clear trend

# Create the scatter plot
plt.scatter(x, y)

# Add labels and a title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot: No Correlation')

# Show the plot
plt.show()
