# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 08:58:29 2024

@author: lk
"""
#   IQR

#Using Numpy
import numpy as np

# Sample data
data = [10, 12, 14, 14, 15, 15, 16, 17, 18, 19, 21, 22, 24, 25, 100]

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)

print("Q1:", Q1)
print("Q3:", Q3)

############################################################

#Using Pandas 

import pandas as pd

# Sample data
data = {'Values': [10, 12, 14, 14, 15, 15, 16, 17, 18, 19, 21, 22, 24, 25, 100]}
df = pd.DataFrame(data)

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Values'].quantile(0.25)
Q2=df['Values'].quantile(0.50)
Q3 = df['Values'].quantile(0.75)

print("Q1:", Q1)
print("Q2:",Q2)
print("Q3:", Q3)
