import csv
from numpy import random, dot, array
import math
import copy
from decimal import *

def loadCSV(filename):
	lines = csv.reader(open(filename, "rb"))
	#random.shuffle(lines)
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
				testSet.append(temp)

	trainSet.extend(dataset[260:267])

	return [trainSet, testSet]

def simplePerceptron(trainSet, weights, bias, learningRate, threshold):

	counter = 0

	while (counter < 10) :

		error = 0
		counter += 1

		for i in range(len(trainSet)):

			Z = trainSet[i][0]
			Y = bias

			Y += dot(trainSet[i][1:], weights)
			
			if Y > threshold:
				Y = 1
			else:
				Y = 0

			error += abs(Z - Y)

			for x in xrange(len(trainSet[i][1:])):
				weights[x] += learningRate*(Z - Y)*trainSet[i][x + 1]

			bias += learningRate*(Z - Y)

		if error < 0.03:
			break
	return (weights)

def testing(weights, bias, testSet, threshold):

	tp = tn = fp = fn = 0.0
	for i in range(len(testSet)):

		Z = testSet[i][0]	
		Y = bias

		Y += dot(testSet[i][1:], weights)

		if Y > threshold:
			Y = 0
		else:
			Y = 1

		if Z == Y:
			tp += 1

		if Z != Y:
			fp += 1
	
	#print tn, tp, fp, fn
	tn = len(testSet) - fp - 1
	fn = len(testSet) - tp - 1

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

	# total = sum(weights)
	
	# for x in xrange(len(weights)):
	# weights[x] = weights[x]/total

	for j in range(1, 11):

		totalAccuracy = totalError = totalFPR = totalFNR = totalTPR = totalTNR = 0.0

		getcontext().prec = 4
		bias = float(1)/float(22)
		weights =  array([float(1)/float(22)]*22)
		learningRate = float(j) * 0.1

		threshold = 3.2

		for i in range(0,10):

			trainSet, testSet = splitDataset(dataset, i)
			weightsNew = simplePerceptron(trainSet, weights, bias, learningRate, threshold)
			accuracy, error, fpr, fnr, tpr, tnr = testing(weightsNew, bias, testSet, threshold)

			#print "\nFold - ", i+1, "\nBias : ", "{0:.2f}".format(bias), "Learning Rate : ", "{0:.2f}".format(learningRate), "Weights : ", ["{0:0.2f}".format(j) for j in weights], "\n Accuracy : ", "{0:.2f}".format(accuracy*100)
			
			totalAccuracy += accuracy
			totalError += error
			totalTNR += tnr
			totalTPR += tpr
			totalFNR += fnr
			totalFPR += fpr

		totalAccuracy /= 10
		totalTNR /= 10
		totalFPR /= 10
		totalTPR /= 10
		totalFNR /= 10

		print "Accuracy : ", totalAccuracy*100, "\tLearning Rate : ", learningRate, "\tThreshold : ", threshold, "\tBias : ", bias
		print "TPR : ", totalTPR*100, "TNR : ", totalTNR*100, "\nFPR :", totalFPR*100, "FNR : ", totalFNR*100