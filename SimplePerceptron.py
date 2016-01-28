import csv
from numpy import random, dot
import math
import copy

def loadCSV(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, foldNumber):
	trainSet = []
	testSet = []
	testClasses = []

	foldSize = int(math.ceil(len(dataset) / 10))

	for i in range(0, 10):
		if i != foldNumber:
			trainSet.extend(dataset[i * foldSize : i*foldSize + foldSize])

		else:
			for j in range(0, foldSize):
				temp = copy.deepcopy(dataset[i * foldSize + j])
				testClasses.append(temp.pop(0))
				testSet.append(temp)

	trainSet.extend(dataset[260:267])

	return [trainSet, testSet, testClasses]

def simplePerceptron(trainSet):

	bias = -1 * random.rand()
	#bias = 0
	weights = random.rand(22)

	learningRate = random.rand()

	counter = 0
	error  = 0 

	while (counter < 1000) :

		counter += 1

		for i in range(len(trainSet)):

			temp = copy.deepcopy(trainSet[i])
			Z = temp.pop(0)
			Y = bias

			Y += dot(temp, weights)
			
			if Y > 0:
				Y = 1
			else:
				Y = 0

			#print Y

			error += abs(Z - Y)

			for x in xrange(len(temp)):
				weights[x] += learningRate*(Z - Y)*temp[x]

		if error == 0:
			break

	return (weights, bias, learningRate)

def testing(weights, bias, testSet, testClasses):

	tp = tn = fp = fn = 0.0
	for i in range(len(testSet)):

		Z = testClasses[i]		
		Y = bias

		Y += dot(testSet[i], weights)

		if Y > 0:
			Y = 0
		else:
			Y = 1

		if Z == Y and Y == 0 :
			tn += 1

		if Z == Y and Y == 1 :
			tp += 1

		if Z != Y and Y == 1 :
			fp += 1

		if Z != Y and Y == 0 :
			fn += 1
	
	print tn, tp, fp, fn
	accuracy  = (tp + tn)/(tp + tn + fn + fp)
	error = 1 - accuracy

	if fp == 0 :
		fpr = 0
	else :
		fpr = fp/(tn + fp)


	if fn!= 0 :
		fnr = fn/(tp + fn)
	else :
		fnr = 0

	if tp!=0 :
		tpr = tp/(tp + fn)
	else :
		tpr = 0

	if tn == 0 :
		tnr = 0
	else :
		tnr = tn/(tn + fp)


	return (accuracy, error, fpr, fnr, tpr, tnr)

if __name__ == "__main__":
	
	filename = "Datasets/Heart.csv"
	dataset = loadCSV(filename)

	totalAccuracy = totalError = totalFPR = totalFNR = totalTPR = totalTNR = 0.0

	for i in range(0,10):
		trainSet, testSet, testClasses = splitDataset(dataset, i)
		weights, bias, learningRate = simplePerceptron(trainSet)
		accuracy, error, fpr, fnr, tpr, tnr = testing(weights, bias, testSet, testClasses)

		print "\nFold - ", i+1, "\nBias : ", "{0:.2f}".format(bias), "Learning Rate : ", "{0:.2f}".format(learningRate), "Weights : ", ["{0:0.2f}".format(j) for j in weights], "\n Accuracy : ", "{0:.2f}".format(accuracy*100)
		
		totalAccuracy += accuracy
		totalError += error
		totalTNR += tnr
		totalTPR += tpr
		totalFNR += fnr
		totalFPR += fpr

	totalAccuracy /= 10

	print "Accuracy : \n", totalAccuracy*100