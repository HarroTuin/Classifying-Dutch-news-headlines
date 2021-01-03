import pandas as pd
import numpy as np
from sklearn import svm, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import to_categorical
import timeit

start = timeit.default_timer()

# Import dataset
X = pd.read_csv('input.csv')
y = pd.read_csv('labels.csv')

# Convert df to np.array
X = np.array(X)
y = np.array(y).ravel()
#y = to_categorical(y)

# Stratified Kfold Split
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM classifier
clf = svm.SVC(kernel='rbf', C=10, gamma=0.5)
clf.fit(X_train, y_train)  
svm_pred = clf.predict(X_test)

print("SVM Accuracy Score -> ", accuracy_score(svm_pred, y_test)*100)

# The commented code below can be used to search for optimal hyperparameters
# parameters = {'kernel':['rbf'], 'C':[1,10], 'gamma': 
#               [0.01,0.1,0.5]}
# grid = GridSearchCV(clf, parameters)
# grid.fit(X_train, y_train)
# svm_pred = grid.predict(X_test)
# print("Best estimator ->",grid.best_estimator_)
# print("SVM Accuracy Score -> ", accuracy_score(svm_pred, y_test)*100)

# Naive Bayes classifier
nb = naive_bayes.MultinomialNB()
nb.fit(X_train, y_train) 
nb_pred = nb.predict(X_test)

print("Naive Bayes Accuracy Score -> ", accuracy_score(nb_pred, y_test)*100)

stop = timeit.default_timer()

print('Time: ', stop - start)  