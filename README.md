# k-means-clustering
In this project, I implement the k-means clustering algorithm in Python. Given is a .csv dataset ("yelp3") to use for clustering.
This algorithm works with the 4 continuous attributes in this dataset: "latitude", "longitude", "reviewCount", and "checkins". It separates 
this data into k clusters and then output the centroids of the clusters and the sum of squared errors of the within-cluster distances as
the scoring function for the algorithm.

Input: python kmeans.py [path/file-name] [k-value] [clustering-option]

Output: 

WC-SSE=15.2179 

Centroid1=[49.00895,8.39655,12,3] 

... 

CentroidK=[33.33548605,-111.7714182,9,97]

For the clustering option:

1 --> The four original attributes for clustering.

2 --> A log transform to reviewCount and checkins.

3 --> Standardize the 4 attributes for clustering.

4 --> Four original attributes and Manhattan distance for clustering.

5 --> A random sample of the data for clustering, specifically the first 6% of the riginal dataset.

6 --> Use your improved score function, which outputs the scoring function as WC-SSE/BC-SSE with BC-SSE being the Between-Cluster Sum of
Squared Error.
