import csv
import random
import math
import copy

def loadCsv(filename):
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

def separateByClass(trainSet):
	separated  = {}
	
	for i in range(len(trainSet)):
		temp = copy.deepcopy(trainSet[i])
		className = temp.pop(0)
		#print len(temp), len(trainSet[i])
		if (className not in separated):
			separated[className] = []
		separated[className].append(temp)
	#print dataset[0]
	return (separated)

def classProbabilities(trainSet):

	# print len(trainSet)
	separated = separateByClass(trainSet)

	probability0 = {}
	probability1 = {}

	for className, features in separated.iteritems():

		probability0[className] = []
		probability1[className] = []

		for j in range(len(features[0])):
			count0 = count1 = 0.0
			for i  in range(len(features)):
				if int(features[i][j]) == 0:
					count0 = count0 + 1
			count1 = len(features) - count0

			probability0[className].append(count0/len(features))
			# print count0/len(features)
			probability1[className].append(count1/len(features))

		#print len(probability0[className]), len(probability1[className])
	return (probability0, probability1)

def naiveBayes(trainData, testData, testClasses):
	
	probability0, probability1 = classProbabilities(trainSet)

	tp = tn = fp = fn = 0.0

	#print len(testData)
	
	for i in range(0, len(testData)):
		prob0 = 0.5
		prob1 = 0.5
		for k in range(0, len(testData[i])):
			#print probability0[0][i*(len(testData[i])) + k]
			if int(testData[i][k]) == 0:
				prob0 *= float(probability0[0][k])
				prob1 *= float(probability0[1][k])
			else :
				prob0 *= float(probability1[0][k])
				prob1 *= float(probability1[1][k]) 

		if prob0 > prob1 :
			result = 0
		else :
			result = 1

		#print prob0, prob1
		#print int(testClasses[i]), result
		if int(testClasses[i]) == result and result == 0 :
			tn += 1

		if int(testClasses[i]) == result and result == 1 :
			tp += 1

		if int(testClasses[i]) != result and result == 1 :
			fp += 1

		if int(testClasses[i]) != result and result == 0 :
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
	filename = 'Datasets/Heart.csv'
	dataset = loadCsv(filename)

	totalAccuracy = totalError = totalFPR = totalFNR = totalTPR = totalTNR = 0.0

	for i in range(0, 10):

		trainSet, testSet, testClasses = splitDataset(dataset, i)

		accuracy, error, fpr, fnr, tpr, tnr = naiveBayes(trainSet, testSet, testClasses)
		#naiveBayes(trainSet, testSet, testClasses)
		#print accuracy

		totalAccuracy += accuracy

		print accuracy
		totalError += error
		totalTNR += tnr
		totalTPR += tpr
		totalFNR += fnr
		totalFPR += fpr

	totalAccuracy /= 10
	print "Accuracy : \n", totalAccuracy*100

