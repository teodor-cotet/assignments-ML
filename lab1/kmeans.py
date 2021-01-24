# Tudor Berariu, 2016

from sys import argv
from zipfile import ZipFile
from random import randint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers
from mpl_toolkits.mplot3d import Axes3D

import random
import math

def getArchive():
    archive_url = "http://www.uni-marburg.de/fb12/datenbionik/downloads/FCPS"
    local_archive = "FCPS.zip"
    from os import path
    if not path.isfile(local_archive):
        import urllib
        print("downloading...")
        urllib.urlretrieve(archive_url, filename=local_archive)
        assert(path.isfile(local_archive))
        print("got the archive")
    return ZipFile(local_archive)

def getDataSet(archive, dataSetName):
    path = "FCPS/01FCPSdata/" + dataSetName

    lrnFile = path + ".lrn"
    with archive.open(lrnFile, "r") as f:                       # open .lrn file
        N = int(f.readline().decode("UTF-8").split()[1])    # number of examples
        D = int(f.readline().decode("UTF-8").split()[1]) - 1 # number of columns
        f.readline()                                     # skip the useless line
        f.readline()                                       # skip columns' names
        Xs = np.zeros([N, D])
        for i in range(N):
            data = f.readline().decode("UTF-8").strip().split("\t")
            assert(len(data) == (D+1))                              # check line
            assert(int(data[0]) == (i + 1))
            Xs[i] = np.array(list(map(float, data[1:])))

    clsFile = path + ".cls"
    with archive.open(clsFile, "r") as f:                        # open.cls file
        labels = np.zeros(N).astype("uint")

        line = f.readline().decode("UTF-8")
        while line.startswith("%"):                                # skip header
            line = f.readline().decode("UTF-8")

        i = 0
        while line and i < N:
            data = line.strip().split("\t")
            assert(len(data) == 2)
            assert(int(data[0]) == (i + 1))
            labels[i] = int(data[1])
            line = f.readline().decode("UTF-8")
            i = i + 1

        assert(i == N)

    return Xs, labels                          # return data and correct classes

def computeCost(k, Xs, centroids):

    totalCost = 0
    for x in Xs:
        minD = np.inf

        for c in centroids:
            if minD > np.linalg.norm(np.array(x) - np.array(c)):
                minD = np.linalg.norm(np.array(x) - np.array(c))

        totalCost += minD
    return totalCost

def computeClusters(K, Xs, centroids):

    clusters = [0] * len(Xs)

    for i in range(0, len(Xs)):
        minD = np.inf
        cIndex = -1
        for c in range(0, len(centroids)):
            if minD > np.linalg.norm(np.array(Xs[i]) - np.array(centroids[c])):
                cIndex = c
                minD = np.linalg.norm(np.array(Xs[i]) - np.array(centroids[c]))

        clusters[i] = cIndex 
    return clusters

def computeCentroids(K, Xs, clusters):
    
    centroids = np.zeros((K, len(Xs[0])))

    for i in range(0, K):
        cnt = 0
        mean = np.array([0] * len(Xs[0]))

        for j in range(0, len(Xs)):
            if clusters[j] == i:
                mean = mean + np.array(Xs[j])
                cnt += 1
        centroids[i] = mean / cnt
    return centroids

def kMeans(K, Xs):
    (N, D) = Xs.shape
    # N = nr de date
    # D = dimensiuni
    print(N)
    centroidsIndex = []

    i = 0
    while i < K:
        choose = random.randint(0, len(Xs) - 1)
        if choose not in centroidsIndex:
            i += 1
            centroidsIndex.append(choose)

    centroids = [Xs[i] for i in centroidsIndex]
    clusters = np.zeros(N).astype("uint")       # id of cluster for each example

    converge = True
    cost = -1.0
    while converge:
        if math.fabs(cost - computeCost(K, Xs, centroids)) < 1e-3:
            converge = False
            continue
        else:
            cost = computeCost(K, Xs, centroids)

        clusters = computeClusters(K, Xs, centroids)
        centroids = computeCentroids(K, Xs, clusters)

    # TODO: Cerinta 1    
    return clusters, centroids

def getMinDist(centroids, x):

	minD = np.inf

	for c in centroids:
		if np.linalg.norm(np.array(c) -  np.array(x)) < minD:
			minD = np.linalg.norm(np.array(c) -  np.array(x))

	return minD

def kMeanspp(K, Xs):
	(N, D) = Xs.shape
    # N = nr de date
    # D = dimensiuni
	centroidsIndex = []
	centroidsIndex.append(random.randint(0, len(Xs) - 1))
	centroids = [Xs[i] for i in centroidsIndex]
	i = 1
	while i < K:
		choose = random.randint(0, len(Xs) - 1)
		alpha = 0
		distances = []

		for j in range(0, len(Xs)):
			d = getMinDist(centroids, Xs[j]) # 0 when j is in centroids
			alpha += d
			distances.append(d)
		

		distances = list(np.array(distances) / alpha)
		centroidsIndex.append(np.random.choice(len(Xs), p=distances))
		centroids = [Xs[i] for i in centroidsIndex]
		i += 1

	clusters = np.zeros(N).astype("uint")       # id of cluster for each example

	converge = True
	cost = -1.0
	while converge:
		if math.fabs(cost - computeCost(K, Xs, centroids)) < 1e-6:
			converge = False
			continue
		else:
			cost = computeCost(K, Xs, centroids)
		clusters = computeClusters(K, Xs, centroids)
		centroids = computeCentroids(K, Xs, clusters)
	# TODO: Cerinta 1    
	return clusters, centroids

def findCenterest(Xs):

	centroid = np.array([0] * len(Xs[0]))
	
	for x in Xs:
		centroid = centroid + np.array(x)
	
	centroid = centroid / len(Xs)
	closest = 0
	minD = np.inf
	
	for i in range(0, len(Xs)):
		if minD > np.linalg.norm(np.array(Xs[i]) - centroid):
			minD = np.linalg.norm(np.array(Xs[i]) - centroid)
			closest = i
	return Xs[closest]


def kaufman(K, Xs):
	(N, D) = Xs.shape
	cen = findCenterest(Xs)
	centroids = [cen]
	c = [[ 0 for x in range(len(Xs))] for y in range(len(Xs))]

	print(centroids)
	for k in range(1, K):
		g = [0] * len(Xs)
		choose = 0
		for i in range(0, len(Xs)):
			for j in range(0, len(Xs)):

				if j == i:
					continue

				d = getMinDist(centroids, Xs[j]) # closest distance from j to a centroid
				c[i][j] = max(0, d - np.linalg.norm(np.array(Xs[i]) - np.array(Xs[j])))
				# good if a lot of points are far from a the rest of the centroids and close to i
				g[i] += c[i][j]

			if g[choose] < g[i]:
				choose = i

		centroids.append(Xs[choose])
	print(centroids)
	clusters = np.zeros(N).astype("uint")       # id of cluster for each example

	converge = True
	cost = -1.0
	while converge:
		if math.fabs(cost - computeCost(K, Xs, centroids)) < 1e-3:
			converge = False
			continue
		else:
			cost = computeCost(K, Xs, centroids)
		clusters = computeClusters(K, Xs, centroids)
		centroids = computeCentroids(K, Xs, clusters)
	# TODO: Cerinta 1    
	return clusters, centroids

def randIndex(clusters, labels):

    tp = tn = fp = fn = 0

    for i in range(0, len(Xs)):
        for j in range(0, len(Xs)):
            if clusters[i] == clusters[j]:
                if labels[i] == labels[j]:
                    tp += 1
                else:  
                    fp += 1
            else:
                if labels[i] == labels[j]:
                    fn += 1
                else:
                    tn += 1
    return (tp + tn) / (tp + tn + fp + fn)

def plot(Xs, labels, K, clusters):
    labelsNo = np.max(labels)
    markers = []                                     # get the different markers
    while len(markers) < labelsNo:
        markers.extend(list(matplotlib.markers.MarkerStyle.filled_markers))
    colors = plt.cm.rainbow(np.linspace(0, 1, K+1))

    if Xs.shape[1] == 2:
        x = Xs[:,0]
        y = Xs[:,1]
        for (_x, _y, _c, _l) in zip(x, y, clusters, labels):
            plt.scatter(_x, _y, s=500, c=colors[_c], marker=markers[_l])
        plt.scatter(centroids[:,0], centroids[:, 1],
                    s=800, c=colors[K], marker=markers[labelsNo]
        )
        plt.show()
    elif Xs.shape[1] == 3:
        x = Xs[:,0]
        y = Xs[:,1]
        z = Xs[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for (_x, _y, _z, _c, _l) in zip(x, y, z, clusters, labels):
            ax.scatter(_x, _y, _z, s=200, c=colors[_c], marker=markers[_l])
        ax.scatter(centroids[:,0], centroids[:, 1], centroids[:, 2],
                    s=400, c=colors[K], marker=markers[labelsNo]
        )
        plt.show()
    else:
        for i in range(N1):
            print(i, ": ", clusters[i], " ~ ", labels[i])

if __name__ == "__main__":
    if len(argv) < 3:
        print("Usage: " + argv[0] + " dataset_name K")
        exit()
    Xs, labels = getDataSet(getArchive(), argv[1])    # Xs is NxD, labels is Nx1
    K = int(argv[2])                                # K is the numbe of clusters

    clusters, centroids = kaufman(K, Xs)
    print("randIndex: ", randIndex(clusters, labels))

    plot(Xs, labels, K, clusters)