import numpy
import csv 
import copy
import collections
from scipy.spatial import distance

def compare(x, y):
	return lambda x, y: collections.Counter(x) == collections.Counter(y)

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def assignClusters(k, centroids, dataset):
	assign = []
	for x in range(len(dataset)):
		min_dist = distance.euclidean(centroids[0], dataset[x])
		
		closest = 0
		for y in range(k):
			dist = distance.euclidean(dataset[x], centroids[y])
			if dist < min_dist:
				min_dist = dist
				closest = y
		assign.append(closest)
	return assign

def calcCenter(numberOfClusters, centroids, dataset, eps):
	oldCentroids = []
	oldScore = -1000.0
	score = 0.0

	while ((not compare(oldCentroids, centroids)) or (score - oldScore)*(score - oldScore) > eps):
		#Assign datapoints to clusters
		assign = assignClusters(numberOfClusters, centroids, dataset)

		oldCentroids = centroids
		centroids = []
		oldScore = score
		score = 0.0

		#Compute Score
		for i in xrange(numberOfClusters):
			for j in xrange(len(dataset)):
				if assign[j] == i:
			 		score += distance.euclidean(oldCentroids[i], dataset[j])
		
		#Compute new centroids 
		for i in xrange(numberOfClusters):
			newCentroid = []
			for j in xrange(len(dataset[0])):
				count = 0.0
				newVal = 0.0
				for k in xrange(len(dataset)):
					if assign[k] == i:
						newVal += dataset[k][j]
						count += 1
				if count > 0:
					newVal /= count 	#Mean value
					newCentroid.append(newVal)
			centroids.append(newCentroid)


	return score, assign

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


	if count00>count01:
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

	k = 2
	eps = 0.03
	centroids = []

	x = dataset[0][0]

	centroids.append(dataset[0][1:])

	for i in xrange(len(dataset)):
		if dataset[i][0] != x:
			centroids.append(dataset[i][1:])
			break

	newDataset = []
	class0 = class1 = 0

	for x in xrange(len(dataset)):
		if dataset[x][0] == 0:
			class0 += 1
		else :
			class1 +=1 

		newDataset.append(dataset[x][1:])

	score, assign = calcCenter(k, centroids, newDataset, eps)

	precision, recall, purity, accuracy = validateClusters(k, assign, dataset, class0, class1)

	print "Precision : ", precision, "\nRecall : ", recall, "\nPurity : ", purity, "\nAccuracy : ", accuracy 
				

		


	

