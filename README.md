# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all necessary packages.
2. Upload the appropiate dataset to perform K-Means Clustering.
3. Perform K-Means Clustering on the requried dataset.
4. Plot graph and display the clusters.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Shaik Shoaib Nawaz 
RegisterNumber: 212222240094  
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
x
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(x)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b','c','m']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],
              color=colors[i],label=f'Cluster {i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1], marker="*" ,s=200,color='k',label='Centroids')
plt.title('K.means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:

i.) DataSet:

![image](https://github.com/shoaib3136/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/117919362/5c35cf35-2b3b-4a5d-92ce-3e366e1e9523)

ii.) DataSet plotted on a graph:

![image](https://github.com/shoaib3136/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/117919362/39f83ae4-64ff-47db-803f-0c70e79e1180)


iii.) Centroid Values:

![image](https://github.com/shoaib3136/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/117919362/0073d15f-f455-4653-b1e5-2dc67f085411)

iv.) K-Means Cluster:

![image](https://github.com/shoaib3136/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/117919362/b06a249b-b241-4991-9a28-eb131ec6b0bd)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
