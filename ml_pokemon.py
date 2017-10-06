import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

#Extracts rows of data from train.csv and puts them into a 2d array
grid = [line.strip().split(',') for line in open("Pokemon.csv")]

#Remove the first row since that's just the name of columns
grid.pop(0)

#Get training data
train = []
for i,row in enumerate(grid):
	train.append(row)
	if(i == 600):
		break;

#Get results of the training data
train_results = []
for row in train:
	train_results.append(row[12])
	row.pop(12)
	row.pop(0)
	row.pop(0)
	row.pop(0)
	row.pop(0)

#Get the test data 
test = []
for i,row in enumerate(grid):
	if i>600:
		test.append(row)

#Remove the unnecessary data 
for row in test:
	row.pop(12)
	row.pop(0)
	row.pop(0)
	row.pop(0)
	row.pop(0)

#sklearn
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(train, train_results)

#92% Accuracy
print(clf.predict(test))

