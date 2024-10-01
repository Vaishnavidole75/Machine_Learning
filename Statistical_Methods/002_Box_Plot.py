# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 09:11:00 2024

@author: lk
"""
#Box Plot Method
#The Box Plot is a graphical tool used to visualize
# the distribution of a dataset and identify potential outliers 
#using the Interquartile Range (IQR) method.


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Sample data
data = [10, 12, 14, 14, 15, 15, 16, 17, 18, 19, 21, 22, 24, 25, 100]
df = pd.DataFrame(data, columns=['Values'])

# Create the box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Values'], color='lightblue')

# Add plot title
plt.title('Box Plot with Outliers Identified')
plt.show()



