import pandas as pd
import numpy as np
import math
import sys
import os
from array import *
import time
import random

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
runtime = int(current_time[3:5])


def distance(x, y):
    # Euclidean distance between 2 data entries
    la = (x[0] - y[0]) ** 2
    lo = (x[1] - y[1]) ** 2
    rc = (x[2] - y[2]) ** 2
    ch = (x[3] - y[3]) ** 2
    d = math.sqrt(la + lo + rc + ch)
    return d


def manhattan(x, y):
    # Manhattan distance between 2 data entries
    la = (x[0] - y[0])
    lo = (x[1] - y[1])
    rc = (x[2] - y[2])
    ch = (x[3] - y[3])
    d = la + lo + rc + ch
    return d


def create_cluster(data, centroid, clusters, option):
    sse = 0
    # opt = int(sys.argv[3])
    # 1. Put each entry in the closest cluster
    for i in range(len(data)):
        c = [data.loc[i, 'latitude'], data.loc[i, 'longitude'], data.loc[i, 'reviewCount'], data.loc[i, 'checkins']]
        if c in centroid:
            continue

        # Find the shortest distance
        keep, index = -1, -1
        for j in range(len(centroid)):
            # Manhattan distance (4)
            # if opt == 4:
            #   d = manhattan(c, centroid[j])
            # else:
            d = distance(c, centroid[j])
            if keep == -1 or d < keep:
                keep = d
                index = j

        sse += (keep ** 2)

        # Add to closest cluster
        s = 'Cluster' + str(index)
        clusters[s].append(i)

    # Within-cluster sum of squared errors
    if option == 1:
        return sse

    # 2. Calculate new centroids
    cent = []
    for i in range(len(centroid)):
        s = 'Cluster' + str(i)
        la, lo, rc, ch = 0, 0, 0, 0
        for j in clusters[s]:
            la += data.loc[j, 'latitude']
            lo += data.loc[j, 'longitude']
            rc += data.loc[j, 'reviewCount']
            ch += data.loc[j, 'checkins']

        lth = len(clusters[s])
        if lth == 0:
            cent.append([0, 0, 0, 0])
            continue

        la = int((la / lth) * 1000000) / 1000000
        lo = int((lo / lth) * 1000000) / 1000000
        rc = int((rc / lth) * 1000000) / 1000000
        ch = int((ch / lth) * 1000000) / 1000000
        cent.append([la, lo, rc, ch])

    return cent


def standardize(data):
    mean1 = data['latitude'].std(axis=0, skipna=True)
    sd1 = data['latitude'].mean(axis=0)
    data['latitude'] = (data['latitude'] - mean1) / sd1

    mean2 = data['longitude'].std(axis=0, skipna=True)
    sd2 = data['longitude'].mean(axis=0)
    data['longitude'] = (data['longitude'] - mean2) / sd2

    mean3 = data['reviewCount'].std(axis=0, skipna=True)
    sd3 = data['reviewCount'].mean(axis=0)
    data['reviewCount'] = (data['reviewCount'] - mean3) / sd3

    mean4 = data['checkins'].std(axis=0, skipna=True)
    sd4 = data['checkins'].mean(axis=0)
    data['checkins'] = (data['checkins'] - mean4) / sd4

    return data


def cluster(file_name, k, option):
    global runtime
    data = pd.read_csv(file_name, encoding='utf-8')

    # Log transformation (2)
    if option == 2:
        data['reviewCount'] = np.log10(data['reviewCount'])
        data['checkins'] = np.log10(data['checkins'])

    # Standardization (3)
    if option == 3:
        data = standardize(data)

    # 6% of data (5)
    if option == 5:
        data = data.head(int(len(data) * 0.06))

    # Choose k centroids
    centroid = []
    clusters = {}  # Dictionary containing indexes of entries in a centroid
    for i in range(k):
        ii = random.randint(0, len(data) - 1)
        c = [data.loc[ii, 'latitude'], data.loc[ii, 'longitude'], data.loc[ii, 'reviewCount'], data.loc[ii, 'checkins']]
        centroid.append(c)
        s = 'Cluster' + str(i)
        clusters[s] = []

    # iterate until clusters are stable
    c1 = create_cluster(data, centroid, clusters, 0)
    c2 = create_cluster(data, c1, clusters, 0)
    while np.array_equal(c1, c2) is not True:
        c1 = c2
        c2 = create_cluster(data, c1, clusters, 0)
        t2 = time.localtime()
        current_time2 = time.strftime("%H:%M:%S", t2)
        runtime2 = int(current_time2[3:5])
        if (runtime2 - runtime) >= 9:
            break

    # Result
    sse = create_cluster(data, c2, clusters, 1)
    print('WC-SSE=', sse)

    # Improved scoring function (6)
    if option == 6:
        bcsse = 0
        for i in range(len(c2) - 1):
            for j in range(i + 1, len(c2)):
                bcsse += distance(c2[i], c2[j]) ** 2

        print('Improved f(C, D)=', sse / bcsse)

    for i in range(len(c2)):
        print('Centroid', i+1, '=', c2[i])


def main():
    k = int(sys.argv[2])
    opt = int(sys.argv[3])
    cluster(sys.argv[1], k, opt)
    # cluster("../data/given/yelp3.csv", 4, 1)


# Calling main function
if __name__ == "__main__":
    main()
