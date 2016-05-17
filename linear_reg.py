import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

lines = csv.reader(open('Datasets/Heart.csv', "rb"))
dataset = list(lines)
datase = []
for i in range(len(dataset)):
  datase.append([float(x) for x in dataset[i][1:]])

data = scale(datase)

n_samples, n_features = data.shape
n_digits = 2#len(np.unique(digits.target))
labels = [0,1]

sample_size = 300
