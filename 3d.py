import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Read the Excel file
file_path = ".\\datasets\\concrete\\Concrete_Data.xls"
data = pd.read_excel(file_path)

# Rename columns to be more concise
data.columns = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Concrete Compressive Strength']

target = data.columns[-1]

X = data.drop(columns=[target])
y = data[target]

# Step 1: Standardization (Mean-Centering and Scaling)
# Standardize the features
# Manually standardize the features
means = X.mean()
stds = X.std()
X_scaled = (X - means) / stds

# Step 2: Compute the Covariance Matrix
covariance_matrix = np.cov(X_scaled, rowvar=False)

# Step 3: Compute the Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Step 4: Select Principal Components
# Sort the eigenvalues and corresponding eigenvectors
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]
sorted_parameters = X_scaled.columns[sorted_index]

# Transform the data to 3D
X_reduced_3d = np.dot(X_scaled, sorted_eigenvectors[:, 0:3])

# 3D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(X_reduced_3d[:, 0], X_reduced_3d[:, 1], X_reduced_3d[:, 2], c=y, cmap='viridis_r', marker='o')

# Labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Concrete Data')

plt.show()