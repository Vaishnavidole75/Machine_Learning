# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:24:57 2024

@author: Vaishnavi

"""

# t-SNE Plot 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

# Load sample data: Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# Create a DataFrame for Seaborn plotting
tsne_df = np.concatenate((X_tsne, y[:, np.newaxis]), axis=1)
tsne_df = pd.DataFrame(tsne_df, columns=['Component 1', 'Component 2', 'species'])

# Plot the t-SNE plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='Component 1', y='Component 2', hue='species', palette='viridis', data=tsne_df)
plt.title('t-SNE Plot of Iris Dataset')
plt.show()


'''
How to Read the t-SNE Plot
X and Y Axes: Represent the two reduced dimensions from t-SNE. These are the new, non-linear components that capture the relationships between points.
Data Points: Each point represents a data observation from the high-dimensional space, now mapped into 2D.
Clusters: Data points that are close together in the plot were close together in the high-dimensional space as well, meaning they have similar properties. Look for distinct clusters of points.
Colors: Different colors (from the hue argument) represent different classes or groups in the dataset (e.g., different species of iris flowers).

'''
