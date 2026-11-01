# ============================================================
# K-MEANS AND HIERARCHICAL CLUSTERING
# DATASET: sales_data_sample.csv
# ============================================================

# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Mount Google Drive
drive.mount('/content/drive')

# Read dataset
df = pd.read_csv(
    '/content/drive/MyDrive/sales_data_sample.csv',
    encoding='ISO-8859-1'
)

# Display first five records
print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# Select numerical features
X = df[['SALES', 'QUANTITYORDERED']].copy()

# Remove missing values
X.dropna(inplace=True)

print("\nSelected Features:")
print(X.head())


# ============================================================
# 4. FEATURE SCALING
# ============================================================

# Standardize the features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data:")
print(X_scaled[:5])


# ============================================================
# 5. ELBOW METHOD
# ============================================================

wcss = []

# Calculate WCSS for K = 1 to 10
for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)


# Plot Elbow Graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)

plt.show()


# ============================================================
# 6. K-MEANS CLUSTERING
# ============================================================

# Choose number of clusters from Elbow Graph
k = 3

kmeans = KMeans(
    n_clusters=k,
    init='k-means++',
    random_state=42,
    n_init=10
)

# Fit model and predict clusters
kmeans_labels = kmeans.fit_predict(X_scaled)

# Store cluster labels
X['KMeans_Cluster'] = kmeans_labels

print("\nK-Means Cluster Assignments:")
print(X.head())


# ============================================================
# 7. VISUALIZE K-MEANS CLUSTERS
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=X,
    x='SALES',
    y='QUANTITYORDERED',
    hue='KMeans_Cluster',
    palette='viridis'
)

plt.title('K-Means Clustering')
plt.xlabel('Sales')
plt.ylabel('Quantity Ordered')

plt.show()


# ============================================================
# 8. HIERARCHICAL CLUSTERING
# ============================================================

# Create linkage matrix using Ward method
Z = linkage(
    X_scaled,
    method='ward'
)


# ============================================================
# 9. DENDROGRAM
# ============================================================

plt.figure(figsize=(12, 6))

dendrogram(
    Z,
    truncate_mode='lastp',
    p=30
)

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')

plt.show()


# ============================================================
# 10. AGGLOMERATIVE CLUSTERING
# ============================================================

hierarchical = AgglomerativeClustering(
    n_clusters=k,
    linkage='ward'
)

# Predict hierarchical clusters
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Store cluster labels
X['Hierarchical_Cluster'] = hierarchical_labels

print("\nHierarchical Cluster Assignments:")
print(X.head())


# ============================================================
# 11. VISUALIZE HIERARCHICAL CLUSTERS
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=X,
    x='SALES',
    y='QUANTITYORDERED',
    hue='Hierarchical_Cluster',
    palette='rainbow'
)

plt.title('Hierarchical Clustering')
plt.xlabel('Sales')
plt.ylabel('Quantity Ordered')

plt.show()


# ============================================================
# 12. DISPLAY CLUSTER COUNTS
# ============================================================

print("\nK-Means Cluster Counts:")
print(X['KMeans_Cluster'].value_counts().sort_index())

print("\nHierarchical Cluster Counts:")
print(X['Hierarchical_Cluster'].value_counts().sort_index())


# ============================================================
# 13. SAVE CLUSTERING RESULTS
# ============================================================

# Save results to CSV
X.to_csv(
    '/content/drive/MyDrive/sales_clustering_results.csv',
    index=False
)

print("\nClustering Completed Successfully!")
print("Results saved to Google Drive.")
