"""
Multiple classifiers using K-Fold Cross validation
"""

import pandas as pd
import numpy as np
from sklearn import naive_bayes
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# Import dataset
X = pd.read_csv('input.csv')
y = pd.read_csv('labels.csv')

# Convert df to np.array
X = np.array(X)
y = np.array(y).ravel()

# Split initialize kfold split
kf = KFold(n_splits=4, shuffle=True, random_state=44)

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
accuracy_nb = []
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    nb.fit(X_train, y_train)
    accuracy_nb.append(nb.score(X_test, y_test))

print("Naive Bayes Accuracy Score -> ", sum(accuracy_nb) / len(accuracy_nb))

# # More classifiers
names = ["kNN", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest",
          "Neural Net", "AdaBoost"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier()]

for name, clf in zip(names, classifiers):
    accuracy_pred = []
    for train_index, test_index in kf.split(X, y):
        X_train = X[train_index,:]
        X_test = X[test_index,:]
        y_train = y[train_index]
        y_test = y[test_index]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracy_pred.append(clf.score(X_test, y_test))
    print("Accuracy:",name,sum(accuracy_pred) / len(accuracy_pred))
