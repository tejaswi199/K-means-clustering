import pandas as pd
from sklearn.cluster import KMeans
#Load data from csv
data=pd.read_csv('/content/k-means clutering.csv')
#print first few rows of the data
print(data.head())
#number of clusters(k)
k=2
#initialize kmeans object with the number of clusters
kmeans=KMeans(n_clusters=k)
#fit the kmeans model to the data
kmeans.fit(data)
#get the centroids of the clusters
centroids=kmeans.cluster_centers_
#get the labels for each data point
labels=kmeans.labels_
#print centroids and labels
print("Centeroids:")
print(centroids)
print("\nCluster labels:")
print(labels)


