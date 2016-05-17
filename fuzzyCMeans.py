import numpy
import csv 
import copy
import collections

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def distance(a, b):
	dist = 0
	for x in range(len(a)):
		dist += (a[x] - b[x])*(a[x] - b[x])

	dist = numpy.sqrt(dist)
	return dist

def assignCoefficients(centroids, m, c, dataset, W, dimensions, numberOfPoints):
	
	fuzzyDist = 0.0
	for x in xrange(c):
		for y in xrange(numberOfPoints):
			denom = 0.0
			num = distance(dataset[y], centroids[x])
			for z in xrange(c):
				denom += distance(dataset[y], centroids[z])
				W[y][x] += pow((num/denom),2/(m - 1))
			W[y][x] = 1/W[y][x]
			fuzzyDist += W[y][x] * pow(distance(dataset[y], centroids[x]),2)
	
	return W, fuzzyDist

def calcCentroids(m, c, dataset, W, dimensions, numberOfPoints):

	centroids = numpy.zeros((c,dimensions))
	WTemp = numpy.zeros((numberOfPoints, 2))

	for x in xrange(numberOfPoints):
		WTemp[x] = numpy.power(W[x], m)

	for x in xrange(c):
		for y in xrange(dimensions):
			num = 0.0
			denom = 0.0
			for z in xrange(numberOfPoints):
				num += WTemp[z][x]*dataset[z][y]
				denom += WTemp[z][x]
			centroids[x][y] = num/denom

	return centroids

def fuzzyCMeans(m, c, dataset, W, dimensions, numberOfPoints):

	assign = []
	eps = 0.02

	oldCentroids = numpy.zeros((c,dimensions))
	oldFuzzyDist = 0.0

	while True:
		centroids = calcCentroids(m, c, dataset, W, dimensions, numberOfPoints)
		W, fuzzyDist = assignCoefficients(centroids, m, c, dataset, W, dimensions, numberOfPoints)
		
		if (fuzzyDist - oldFuzzyDist)*(fuzzyDist - oldFuzzyDist) < eps or (oldCentroids == centroids).all():
			break
		oldCentroids = centroids
		oldFuzzyDist = fuzzyDist

	for x in xrange(len(dataset)):
		maxAssign = 0
		for y in xrange(c):
			if W[x][y] > maxAssign:
				maxAssign = W[x][y]
				closest = y
		assign.append(closest)
	
	return assign

def validateClusters(k, assign, dataset, class0, class1):

	count00 = count01 = count10 = count11 = 0
	for x in xrange(len(assign)):
		if assign[x] == 0:
			if dataset[x][0] == 0:
				count00 += 1
			if dataset[x][0] == 1:
				count01 += 1

		if assign[x] == 1:
			if dataset[x][0] == 0:
				count10 += 1
			if dataset[x][0] == 1:
				count11 += 1

	print "Cluster 0 : ", count00, count01, 
	print "\nCluster 1 : ",count10, count11, "\n"


	if count00>count01 and count00>count10:
		count0 = count00
		count1 = count11
	else :
		count0 = count01
		count1 = count10

	accuracy = (float(count0 + count1)/float(count00 + count10 + count11 + count01))*100

	print count00/float(count00 + count01), count01/float(count00 + count01), "\n"
	print count10/float(count11 + count10), count11/float(count10 + count11), "\n"

	purity = count0 + count1

	precision = (count0/(count00 + count01) + count1/(count11 + count10))*100

	recall = (count0/class0 + count1/class1)*100

	return precision, recall, purity, accuracy

if __name__ == "__main__":

	filename = 'Datasets/Heart.csv'
	dataset = loadCsv(filename)

	C = 2
	m = 3.0
	dimensions = 22
	numberOfPoints = len(dataset)
	W = numpy.random.random((numberOfPoints, 2))

	for x in xrange(numberOfPoints):
		sum = 0
		for y in xrange(C):
			sum += W[x][y]

		for y in xrange(C):
			W[x][y] /= sum

	newDataset = []
	class0 = class1 = 0

	for x in xrange(len(dataset)):
		if dataset[x][0] == 0:
			class0 += 1
		else :
			class1 +=1 

		newDataset.append(dataset[x][1:])

	assign = fuzzyCMeans(m, C, newDataset, W, dimensions, numberOfPoints)

	precision, recall, purity, accuracy = validateClusters(C, assign, dataset, class0, class1)

	print "Precision : ", precision, "\nRecall : ", recall, "\nPurity : ", purity, "\nAccuracy : ", accuracy