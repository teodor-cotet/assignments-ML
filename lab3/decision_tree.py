import random
import copy
import numpy
import math
from collections import Counter
from sklearn.model_selection import train_test_split

def getDataSet(path1, path2):
	d = []
	with open(path1, "r") as f:
		for line in f:
			d.append(line.split(","))

	for i in range(len(d)):
		for j in range(len(d[i])):
			d[i][j] = d[i][j].strip()

	attributes = {}

	with open(path2, "r") as f:
		i = 0
		for line in f:
			attributes[i] = line.split(",")
			i += 1

	for key in attributes:
		for j in range(len(attributes[key])):
			attributes[key][j] = attributes[key][j].strip()

	return d, attributes


def findMajority(d, currentExs):
	cnt = { }
	n = len(d[0])

	for i in currentExs:
		if d[i][n-1] not in cnt:
			cnt[d[i][n-1]] = 1
		else:
			cnt[d[i][n-1]] += 1
	maxKey = None
	for key in cnt:
		if maxKey == None or cnt[key] > cnt[maxKey]:
			maxKey = key

	return maxKey

def areTheSame(d, currentExs):
	n = len(d[0])

	if len(currentExs) == 1:
		return d[currentExs[0]][n-1]
	
	for i in range(1, len(currentExs)):
		if d[currentExs[i]][n-1] != d[currentExs[i-1]][n-1]:
			return False

	return d[currentExs[0]][n-1]

def computeEntropy(d, X):

	total = len(X)
	cls = { }
	n = len(d[0])

	for i in range(len(X)):
		f = d[X[i]][n-1]
		if f not in cls:
			cls[f] = 1
		else:
			cls[f] += 1
	ent = 0.0
	for k in cls:
		r = 1.0 * cls[k] / total
		ent += 1.0 * r * math.log(r, 2)

	return -ent

def computeEntropySplit(ind, d, currentExs, chosen):

	cls = {}
	total = len(currentExs)
	#split
	for i in range(len(currentExs)):
		f = d[currentExs[i]][ind]
		if f not in cls:
			cls[f] = [currentExs[i]]
		else:
			cls[f].append(currentExs[i])

	ent = 0.0
	for k in cls:
		ent += 1.0 * len(cls[k]) / total * computeEntropy(d, cls[k])
	return ent

def chooseAttribute(d, currentExs, chosen):
	n = len(d[0])
	ind = 0
	minEntropy = numpy.inf
	entropyX = computeEntropy(d, currentExs)

	for i in range(0, n - 1):
		if i not in chosen:
			e = computeEntropySplit(i, d, currentExs, chosen)
			if e < minEntropy:
				minEntropy = e
				ind = i
	return ind, minEntropy

def random_tree(current, d, depth, currentExs, chosen, attributes, maj, useId3):
	# [attribute feature_to_split  [ attribute feature_to split [] , ], [attribute feature_to_split [] ,[]], ... ]
	# leaf: [attribute, class]

	#if we don't have any examples just put the class of the most examples from the parent's node
	if len(currentExs) == 0:
		current.append(maj)
		return current

	maj = findMajority(d, currentExs)

	if depth == 0:
		current.append(findMajority(d, currentExs))
		return current
	elif areTheSame(d, currentExs) != False:
		current.append(areTheSame(d, currentExs))
		return current
	else:
		n = len(d[0])
		if useId3:
			feature, ent = chooseAttribute(d, currentExs, chosen)
		else:
			feature = random.randint(0, n - 2)
			while feature in chosen:
				feature = random.randint(0, n - 2)

		node = []
		node.append(feature)
		groups = {}
		for i in currentExs:
			if d[i][feature] not in groups:
				groups[d[i][feature]] = [i]
			else:
				groups[d[i][feature]].append(i)

		for v in attributes[feature]:
			if v not in groups:
				groups[v] = []

		newChosen = copy.deepcopy(chosen)
		newChosen.append(feature)
		current.append(feature)

		for g in groups:
			current.append([g])
			random_tree( current[len(current) - 1], d, depth - 1, groups[g], newChosen, attributes, maj, useId3)
		return current

def isLeaf(current):
	return len(current) == 2

def test_example(current, row, li):
	if isLeaf(current): #is leaf
		li.append(current[1])
		if row[len(row) - 1] == current[1]:
			return True
		else:
			return False
	else:
		feature = current[1]
		f = row[feature]
		for i in range(2, len(current)):
			if current[i][0] == f:
				return test_example(current[i], row, li)

def test_tree(tree, tests, pred):

	right = 0
	total = len(tests)

	for i in range(len(tests)):
		t = tests[i]
		r = test_example(tree, t, pred[i])
		if r != False:
			right += 1

	return 1.0 * right / total

def findMaj(li):
	c = Counter(li)
	return c.most_common()[0][0]

def solve(x):
	d, attributes = x
	n = len(d[0])
	resClass = [row[n-1] for row in d]
	spl = int(len(d) * 0.75)
	useId3 = False
	learnB, tests, resLearn, resTests = train_test_split(d, resClass, train_size=spl)
	
	maxi = 0
	nrArbs = 1 if (useId3 == True) else 100

	pred = {i: [] for i in range(len(tests)) }

	for i in range(nrArbs):
		root = [None ]
		learn, _, _, _   = train_test_split(learnB, resLearn, train_size=0.5)

		currentExs = [i for i in range(len(learn))]
		maj = findMajority(d, currentExs)

		random_tree(root, learn, 6, currentExs, [], attributes, maj, useId3)
		m = test_tree(root, tests, pred)
	
	right = 0
	total = len(tests)

	for i in range(len(tests)):
		x = findMaj(pred[i])
		if x == tests[i][len(d[0]) - 1]:
			right += 1

	print(1.0 * right / total)
		

solve(getDataSet("data/car.data", "data/features_cars.data"))
