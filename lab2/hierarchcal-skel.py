# Tudor Berariu, 2016

from sys import argv
from zipfile import ZipFile
from random import randint

import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import matplotlib.markers
from mpl_toolkits.mplot3d import Axes3D

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


def dummy(Xs):
    (N, D) = Xs.shape
    Z = np.zeros((N-1, 4))
    lastIndex = 0
    for i in range(N-1):
        Z[i,0] = lastIndex
        Z[i,1] = i+1
        Z[i,2] = 0.1 + i
        Z[i,3] = i+2
        lastIndex = N+i
    return Z

def dist(x1, x2):
    return np.linalg.norm(np.array(x1) - np.array(x2))

def computeDistance(g1, g2, Xs, dd):

    minD = np.inf

    for p1 in g1:
        for p2 in g2:
            if dd[p1, p2] < minD:
                minD = dd[p1, p2]
    return minD

def computeDistanceMax(g1, g2, Xs, dd):

    maxD = -np.inf

    for p1 in g1:
        for p2 in g2:
            if dd[p1, p2] > maxD:
                maxD = dd[p1, p2]
    return maxD

def computeDistanceAvg(g1, g2, Xs, dd):

    d = 0
    cnt = 0
    for p1 in g1:
        for p2 in g2:
            d += dd[p1, p2]
            cnt += 1
    return d / cnt

def findClosestSingleLinkage(gr, Xs, dd):

    minD = np.inf
    g1min = g2min = 0
    for g1 in gr:
        for g2 in gr:
            if g1 != g2:
                d = computeDistance(gr[g1], gr[g2], Xs, dd)
                if minD > d:
                    minD = d
                    g1min = g1
                    g2min = g2
    return (g1min, g2min, minD)

def findClosestCompleteLinkage(gr, Xs, dd):

    minD = np.inf
    g1min = g2min = 0

    for g1 in gr:
        for g2 in gr:
            if g1 != g2:
                d = computeDistanceMax(gr[g1], gr[g2], Xs, dd)
                if minD > d:
                    minD = d
                    g1min = g1
                    g2min = g2
    return (g1min, g2min, minD)

def findClosesetGroupAverage(gr, Xs, dd):
    minD = np.inf
    g1min = g2min = 0
    for g1 in gr:
        for g2 in gr:
            if g1 != g2:
                d = computeDistanceAvg(gr[g1], gr[g2], Xs, dd)
                if minD > d:
                    minD = d
                    g1min = g1
                    g2min = g2
    return (g1min, g2min, minD)

def computeAllD(Xs):
    (N, D) = Xs.shape
    dd = np.zeros( (N, N) )
    for i in range(N):
        for j in range(N):
            dd[i, j] = dist(Xs[i], Xs[j])   
    return dd

def unify(Xs, dd, f):

    (N, D) = Xs.shape
    groups = {i: [i] for i in range(N)}
    Z = np.zeros((N-1, 4))
    
    for i in range(N, 2 * N - 1):
        (g1, g2, d) = f(groups, Xs, dd)
        pp = groups[g1] + groups[g2]
        groups[i] = pp
        del groups[g1]
        del groups[g2]
        Z[i - N, 0] = g1
        Z[i - N, 1] = g2
        Z[i - N, 2] = d
        Z[i - N, 3] = len(pp)
    return Z

def singleLinkage(Xs):
    # TODO 1

    dd = computeAllD(Xs)   
    return unify(Xs, dd, findClosestSingleLinkage)

def completeLinkage(Xs):
    dd = computeAllD(Xs)   
    return unify(Xs, dd, findClosestCompleteLinkage)

def groupAverageLinkage(Xs):
    # TODO 3
    dd = computeAllD(Xs)
    return unify(Xs, dd, findClosesetGroupAverage)


def extractClusters(Xs, Z):
    (N, D) = Xs.shape
    assert(Z.shape == (N-1, 4))
    # cluster when max distance
    # TODO 4
    indMax = 0
    dMax = 0
    for i in range(N - 2):
        if Z[i + 1, 2] - Z[i, 2] > dMax:
            indMax = i + 1
            dMax = Z[i + 1, 2] - Z[i, 2]

    print(Z)

    groups = {i: [i] for i in range(N)}
    for i in range(0, indMax):
        g1 = Z[i, 0]
        g2 = Z[i, 1]
        pp = groups[g1] + groups[g2]
        groups[i + N] = pp
        del groups[g1]
        del groups[g2]

    clusters = np.zeros(N)
    for g in groups:
        for x in groups[g]:
            clusters[x] = g
    cnt = 0
    count = { }
    scaledClusters = np.zeros(N)
    for i in  range(len(clusters)):
        x = clusters[i]
        if x not in count:
            count[x] = cnt
            cnt += 1
        scaledClusters[i] = count[x]

    nrClusters = len(count)
    print(nrClusters)
    return nrClusters, scaledClusters.astype('int')

def randIndex(clusters, labels):
    assert(labels.size == clusters.size)
    N = clusters.size

    a = 0.0
    b = 0.0

    for (i, j) in [(i,j) for i in range(N) for j in range(i+1, N) if i < j]:
        if ((clusters[i] == clusters[j]) and (labels[i] == labels[j]) or
            (clusters[i] != clusters[j]) and (labels[i] != labels[j])):
            a = a + 1
        b = b + 1

    return float(a) / float(b)

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
            plt.scatter(_x, _y, s=200, c=colors[_c], marker=markers[_l])
        plt.show()
    elif Xs.shape[1] == 3:
        x = Xs[:,0]
        y = Xs[:,1]
        z = Xs[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for (_x, _y, _z, _c, _l) in zip(x, y, z, clusters, labels):
            ax.scatter(_x, _y, _z, s=200, c=colors[_c], marker=markers[_l])
        plt.show()
    else:
        for i in range(N1):
            print(i, ": ", clusters[i], " ~ ", labels[i])


if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: " + argv[0] + " dataset_name")
        exit()

    Xs, labels = getDataSet(getArchive(), argv[1])    # Xs is NxD, labels is Nx1
    Z = singleLinkage(Xs)

    plt.figure()
    dn = hierarchy.dendrogram(Z)
    # plt.show()

    K, clusters = extractClusters(Xs, Z)
    print("randIndex: ", randIndex(clusters, labels))

    plot(Xs, labels, K, clusters)
