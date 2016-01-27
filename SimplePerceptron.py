import csv
import random
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

	random.seed()

	bias = random.uniform(-1,0)
	weights = []

	for i in range(0,len(trainSet[0])):
		weights.append(random.uniform(0,1)) 

	learningRate = random.uniform(0,1)

	counter = 0
	error  = 0 
	while (counter < 1000) :

		for i in range(len(trainSet)):

			temp = copy.deepcopy(trainSet[i])
			Z = temp.pop(0)

			Y = bias

			for j in range(len(temp)):
				Y = Y + (temp[j] * weights[j])
				
			if Y > 0:
				Y = 1
			else:
				Y = 0

			error += abs(Z - Y)

			for j in range(len(temp)):
				weights[j] += learningRate*(Z - Y)*weights[j]

		counter += 1

		if error == 0:
			
			break
	

	return (weights, bias)

def testing(weights, bias, testSet, testClasses):

	tp = tn = fp = fn = 0.0
	for i in range(len(testSet)):

		Z = testClasses[i]		
		Y = bias
		for j in range(len(testSet[i])):
			Y = Y + (testSet[i][j] * weights[j])

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
	
	#print tn, tp, fp, fn
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
	
	filename = "Heart.csv"
	dataset = loadCSV(filename)

	totalAccuracy = totalError = totalFPR = totalFNR = totalTPR = totalTNR = 0.0

	for i in range(0,10):
		trainSet, testSet, testClasses = splitDataset(dataset, i)
		weights, bias = simplePerceptron(trainSet)
		accuracy, error, fpr, fnr, tpr, tnr = testing(weights, bias, testSet, testClasses)
		totalAccuracy += accuracy

		print accuracy
		totalError += error
		totalTNR += tnr
		totalTPR += tpr
		totalFNR += fnr
		totalFPR += fpr

	totalAccuracy /= 10

	print "Accuracy : \n", totalAccuracy*100