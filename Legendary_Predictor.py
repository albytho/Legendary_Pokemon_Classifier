# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##############################################################################
# Importing the dataset
dataset = pd.read_csv('Pokemon.csv')
dataset[['Type 2']] = dataset[['Type 2']].fillna(value='None')

#Convert our data into matrices

# Get columns 3 through 12 (indexing starts at 0)
X = dataset.iloc[:, 2:12].values

# Our dependent value is in column 13
y = dataset.iloc[:, 12].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Convert Type1 and Type2 string into actual number
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Add dummy variables to Type 1 column.  This is better than having a column
# that consists of 0,1,2,etc
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

onehotencoder = OneHotEncoder(categorical_features = [17])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
# Normalize all the input data
from sklearn.preprocessing import StandardScaler

#You only need to do transform after doing fit_transform once
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu', input_dim = 43))
classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 16, nb_epoch = 100)

#Part 3 - Making Predictions
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu', input_dim = 43))
    classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size = 16, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer, droprate):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu', input_dim = 43))
    classifier.add(Dropout(p = droprate))
    classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = droprate))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [16],
              'nb_epoch': [100],
              'droprate': [0.0,0.1,0.2],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_