# classification.py
# experimenting with hands on classification.
# 
# Author: Ronny Macmaster

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
print mnist

X, y = mnist["data"], mnist["target"]
print X.shape, y.shape

# display a random digit
digit = X[36000]
digit_img = .reshape(28, 28)
plt.imshow(digit_img, cmap=cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# partition the dataset
pivot = 60000 # mnist is divided into train/test here
shuffle = np.random.permutation(pivot)
xtrain, ytrain = X[:pivot], y[:pivot], 
xtrain, ytrain = xtrain[shuffle], ytrain[shuffle]
xtest, ytest = X[pivot:], y[pivot:]
xtest, ytest = xtest[shuffle], ytest[shuffle]

# for binary classification on digit 5
ytrain5 = (ytrain == 5)
ytest5 =  (ytest == 5)

# stochastic gradient descent 
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(xtrain, ytrain5)

print sgd_clf.predict(digit)
