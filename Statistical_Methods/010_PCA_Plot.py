# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:30:38 2024

@author: Vaishnavi
"""

# PCA Plot 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Feature matrix
y = iris.target  # Target labels

# Apply PCA to reduce the dataset to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame for Seaborn plotting
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = y

# Plot the PCA plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='species', palette='deep', data=pca_df)
plt.title('PCA Plot of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Show the explained variance ratio for each component
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by PC1: {explained_variance[0]:.2f}')
print(f'Explained variance by PC2: {explained_variance[1]:.2f}')

plt.show()


'''
How to Read the PCA Plot
X and Y Axes: The two axes (PC1 and PC2) represent the first two principal components. These are the directions in the data where the variance is maximized.

PC1 (Principal Component 1): Explains the largest variance in the dataset.
PC2 (Principal Component 2): Explains the second largest variance, orthogonal to PC1.
Data Points: Each point represents a sample (observation) from the dataset, projected onto the new 2D space defined by PC1 and PC2.

Colors: The points are colored by their labels (species in the Iris dataset). Different colors correspond to different species of iris flowers.



'''
