# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 08:54:06 2024

@author: lk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 08:53:32 2024

@author: lk
"""

#Using Numpy
import numpy as np

# Sample data
data = [10, 12, 14, 14, 15, 15, 16, 17, 18, 19, 21, 22, 24, 25, 100]

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Calculate Z-scores
z_scores = [(data - mean) / std_dev ]

print("Z-scores:", z_scores)

########################################################

#Using Pandas
import pandas as pd

# Sample data
data = {'Values': [10, 12, 14, 14, 15, 15, 16, 17, 18, 19, 21, 22, 24, 25, 100]}
df = pd.DataFrame(data)

# Calculate Z-scores using Pandas' mean and std functions
df['Z-Score'] = (df['Values'] - df['Values'].mean()) / df['Values'].std()

print(df)
