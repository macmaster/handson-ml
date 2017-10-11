# classification.py
# experimenting with hands on classification.
# 
# Author: Ronny Macmaster

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original", data_home="datasets")
print mnist

X, y = mnist["data"], mnist["target"]
print X.shape, y.shape

# display a random digit
digit = X[36000]
digit_img = digit.reshape(28, 28)
plt.imshow(digit_img, cmap=cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# partition the dataset
pivot = 60000 # mnist is divided into train/test here
shuffle = np.random.permutation(pivot)
xtest, ytest = X[pivot:], y[pivot:]
xtrain, ytrain = X[:pivot], y[:pivot], 
xtrain, ytrain = xtrain[shuffle], ytrain[shuffle]

# for binary classification on digit 5
ytrain5 = ytrain == 5
ytest5 = ytest == 5

# stochastic gradient descent 
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(xtrain, ytrain5)
print sgd_clf.predict([digit])

# confusion matrix scoring
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
cval_pred = cross_val_predict(sgd_clf, xtrain, ytrain5, cv=3)
print "confusion_matrix:\n", confusion_matrix(ytrain5, cval_pred)

# # precision recall tradeoff
# from sklearn.metrics import precision_recall_curve
# cval_pred = cross_val_predict(sgd_clf, xtrain, ytrain5, cv=3, method="decision_function")
# print cval_pred.shape, ytrain5.shape
# precisions, recalls, thresholds = precision_recall_curve(ytrain5, cval_pred)
# 
# plt.plot(thresholds, precisions, "b", label="Precision")
# plt.plot(thresholds, recalls, "g", label="Recall")
# plt.xlabel("Threshold")
# plt.legend(loc="lower right")
# plt.ylim([0, 1])
# plt.show()

# train a forest classifier
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()
forest_clf.fit(xtrain, ytrain5)
yprob = cross_val_predict(forest_clf, xtrain, ytrain5, cv=3, method="predict_proba")
yscores = yprob[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(ytrain5, yscores)
plt.plot(fpr_forest, tpr_forest, "g", label="roc")
plt.legend(loc="lower right")
plt.show()
