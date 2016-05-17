import csv
import numpy as np
import math
import copy
import operator
from decimal import *

#TO DO :
# Write to csv
# Cross validate with scikit

#load CSV file
def loadCSV(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

#Split the data for 10 fold cross validation
def splitDataset(dataset, foldNumber):
	trainSet = []
	testSet = []
	testClasses = []

	foldSize = int(math.ceil(len(dataset) / 10))

	for i in range(0, 10):
		if i != foldNumber:
			trainSet.extend(dataset[i * foldSize : i * foldSize + foldSize])

		else:
			for j in range(0, foldSize):
				testSet.append(copy.deepcopy(dataset[i * foldSize + j]))
			
	trainSet.extend(dataset[260:267])

	return [trainSet, testSet]

#Error correction and back propagation
def backPropagate(features, hiddenNode, outputNode, actualOutput, learningRate, weightsOutput, weightsInput, biasHidden, biasOutput):
	
	errorOutput = np.dot(outputNode, np.dot((1 - outputNode), (actualOutput - outputNode)))

	errorHidden = []
	for hidden in range(len(hiddenNode)):
		error = hiddenNode[hidden]*(1 - hiddenNode[hidden])
		errW = 0
		for output in range(len(outputNode)):
			errW = errW + errorOutput[output]*weightsOutput[hidden*len(outputNode) + output]
		error = error * errW
		errorHidden.append(error)

	#update Hidden layer Weights
	for y in range(0,5):
		for x in range(len(features)):
			weightsInput[y] += learningRate*errorHidden[y]*features[x]
		biasHidden[y] += learningRate*errorHidden[y]

	#update Output layer weights
	for y in range(len(outputNode)):
		for hidden in hiddenNode:
			weightsOutput[y] += learningRate*errorOutput[y]*hidden
		biasOutput[y] += learningRate*errorOutput[y]

	return [weightsInput, weightsOutput, biasHidden, biasOutput, totalError]

#Computing the output for each node
def feedForward(trainSet, weightsInput, weightsOutput, biasHidden, biasOutput, learningRate):

	counter = 0

	while(counter < 10):
		counter += 1
		error = 0
		for datapoint in trainSet:
			hiddenNode = []
			outputNode = []

			actualClassLabel = datapoint[0]

			for hidden in range(0,5):

				inputH = biasHidden[hidden]
				inputH += np.dot(datapoint[1:], weightsInput[hidden*len(datapoint[1:]) : hidden*len(datapoint[1:]) + len(datapoint[1:])])
				
				outputH = 1/(1 + math.exp(-1*inputH))
				hiddenNode.append(outputH)

			for output in range(0,2):
				
				inputO = biasOutput[output]
				
				for hidden in range(0,5):
					inputO += hiddenNode[hidden]*weightsOutput[output*5 + hidden]
				
				outputO = 1/(1 + np.exp(-1*inputO))
				outputNode.append(outputO)

			for x in xrange(0,2):
				if actualClassLabel == x:	
					actualOutput.append(1)
				else:
					actualOutput.append(0)
			
			weightsInput, weightsOutput, biasHidden, biasOutput, errorEach = backPropagate(datapoint, hiddenNode, outputNode, actualOutput, learningRate, weightsOutput, weightsInput, biasHidden, biasOutput)
		
			error += errorEach
		if error < 0.5:
			break;
	return weightsInput, weightsOutput, biasHidden, biasOutput

def testing(weightsHidden, weightsOutput, biasHidden, biasOutput, testSet):

	getcontext().prec = 4
	tp = tn = fp = fn = Decimal(0.0)

	for i in range(len(testSet)):
		
		Z = testSet[i][0]
		
		hiddenO = []
		for j in xrange(0, 5):
			
			hiddenI = biasHidden[j]
			hiddenI += np.dot(testSet[i][1:], weightsHidden[j*len(testSet[i][1:]) : j*len(testSet[i][1:]) + len(testSet[i][1:])])

			outputH = 1/(1 + np.exp(-1*hiddenI))
			hiddenO.append(outputH)

		outputO = []
		for j in range(0, 2):

			outputI = biasOutput[j]
			outputI += np.dot(hiddenO[:], weightsOutput[j*len(hiddenO) : j*len(hiddenO) + len(hiddenO)])

			Ooutput = 1/(1 + np.exp(-1*outputI))
			outputO.append(Ooutput)

		if(outputO[1] > outputO[0]):
			Y = 1
		else :
			Y = 0
		if Z == Y and Y == 0 :
			tn += 1

		if Z == Y and Y == 1 :
			tp += 1

		if Z != Y and Y == 1 :
			fp += 1

		if Z != Y and Y == 0 :
			fn += 1

	getcontext().prec = 4
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

	print "Accuracy", "\tLearning Rate", "\tTPR", "\tTNR", "\tFPR", "\tFNR"
	
	for j in range(1, 11):

		learningRate = float(j) * 0.1

		getcontext().prec = 4
		biasHidden = np.array([float(1)/float(22)]*5)
		biasOutput = np.array([float(1)/float(5)]*2)
		weightsHidden = np.array([float(1)/float(22)]*110) #22*5 
		weightsOutput = np.array([float(1)/float(5)]*10) #5*2
		totalAccuracy = totalError = totalFPR = totalFNR = totalTPR = totalTNR = Decimal(0.0)

		for i in range(0,10):

			trainSet, testSet = splitDataset(dataset, i)
			weightsHidden, weightsOutput, biasHidden, biasOutput = feedForward(trainSet, weightsHidden, weightsOutput, biasHidden, biasOutput, learningRate)

			accuracy, error, fpr, fnr, tpr, tnr = testing(weightsHidden, weightsOutput, biasHidden, biasOutput, testSet)
	
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

		print totalAccuracy*100, '\t', learningRate, '\t', totalTPR, '\t', totalTNR, '\t', totalFPR, '\t', totalFNR
		
