import csv
import numpy as np
import math
import copy
import operator
from decimal import *

#TO DO :
# Write to csv
# Cross validaate without extra spaces

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
				testSet.append(copy.deepcopy(dataset[i * foldSize + j]))
			
	trainSet.extend(dataset[260:267])

	return [trainSet, testSet]

def backPropagate(features, hiddenNode, outputNode, learningRate, weightsOutput, weightsInput, biasHidden, biasOutput):

	errorOutput = []

	for i in range(len(outputNode)):
		error = outputNode[i]*(1 - outputNode[i])*(i - outputNode[i])
		errorOutput.append(error)

	errorHidden = []
	for hidden in range(len(hiddenNode)):
		
		error = hiddenNode[hidden]*(1 - hiddenNode[hidden])

		for x in xrange(0, 2):
			error += errorOutput[x]*weightsOutput[x][hidden]
		errorHidden.append(error)

	#update Hidden layer Weights
	for y in range(0,5):
		for x in range(len(features[1:])):
			weightsInput[y][x] += learningRate*errorHidden[y]*features[x + 1]
		biasHidden[y] += learningRate*errorHidden[y]

	#update Output layer weights
	for y in range(len(outputNode)):
		for hidden in range(len(hiddenNode)):
			weightsOutput[y][hidden] += learningRate*errorOutput[y]*hiddenNode[hidden]
		biasOutput[y] += learningRate*errorOutput[y]

	return [weightsInput, weightsOutput, biasHidden, biasOutput, totalError]

def feedForward(trainSet, weightsInput, weightsOutput, biasHidden, biasOutput, learningRate):

	counter = 0

	while(counter < 100):
		counter += 1
		
		for datapoint in trainSet:
			datapoint = trainSet[0]
			hiddenNode = []
			outputNode = []

			for hidden in range(0,5):

				inputH = np.dot(datapoint[1:], weightsInput[hidden][:]) + biasHidden[hidden]
				outputH = 1/(1 + np.exp(-inputH))
				hiddenNode.append(outputH)

			for output in range(0,2):

				inputO = np.dot(hiddenNode[:], weightsOutput[output][:]) + biasOutput[output]
				outputO = 1/(1 + np.exp(-inputO))
				outputNode.append(outputO)


			weightsInput, weightsOutput, biasHidden, biasOutput, errorEach = backPropagate(datapoint, hiddenNode, outputNode, learningRate, weightsOutput, weightsInput, biasHidden, biasOutput)
	
	return weightsInput, weightsOutput, biasHidden, biasOutput

def testing(weightsHidden, weightsOutput, biasHidden, biasOutput, testSet):

	tp = tn = fp = fn = 0.0

	for i in range(len(testSet)):
		
		Z = testSet[i][0]
		
		hiddenO = []
		for j in xrange(0, 5):
			
			hiddenI = biasHidden[j]
			hiddenI += np.dot(testSet[i][1:], weightsHidden[j][:])

			outputH = 1/(1 + np.exp(-1*hiddenI))
			hiddenO.append(outputH)

		outputO = []
		for j in range(0, 2):

			outputI = biasOutput[j]
			outputI += np.dot(hiddenO, weightsOutput[j][:])

			Ooutput = 1/(1 + np.exp(-1*outputI))
			outputO.append(Ooutput)

		o1 = outputO[1]*(1 - outputO[1])*(1 - outputO[1])
		o0 = outputO[0]*(1 - outputO[0])*(0 - outputO[0])
		
		if(abs(o1) < abs(o0)):
			Y = 1
		else :
			Y = 0

		if Z == Y :
			tp += 1

		if Z != Y :
			fp += 1


	#print tn, tp, fp, fn
	tn = len(testSet) - fp 
	fn = len(testSet) - tp 

	accuracy  = (tp + tn)/(tp + tn + fn + fp)
	error = 1 - accuracy

	# print accuracy

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

	print "Accuracy", "\tLearning Rate"
	
	for j in range(1, 11):

		learningRate = float(j) * 0.1
		getcontext().prec = 4
		totalAccuracy = totalError = totalFPR = totalFNR = totalTPR = totalTNR = 0.0

		for i in range(0,10):

			getcontext().prec = 4
			biasHidden = np.array([float(1)/float(23)]*5)
			biasOutput = np.array([float(1)/float(6)]*2)
			weightsHidden = np.array([[float(1)/float(23)]*22]*5) #5X22
			weightsOutput = np.array([[float(1)/float(6)]*5]*2) #2X5
			

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
		print totalAccuracy*100, '\t', learningRate
		
