# sklearn.neighbors provides functionality for unsupervised and supervised neighbors-based learning methods
# supervised learning has 2 flavors
# 	- classification for data with discrete labels
#   - regression for data with continuous labels
# 
# Basic principal - find a predefined number of training samples closest in distance to the new point
# & predict the label from these

# Classification libs: KNeighborsClassifier, RadiusNeighborsClassifier

# KNeighborsClassifier - optimal k value is dependent on data-size
#  	larger the k more effect of noise, but makes the classification boundaries less distinct
# Can adjust waits using the weights keyword. Default is 'uniform', other option is 'distance', can do a user func as well

# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 50

# import some data to play with 
# for more information about the data set: https://github.com/scikit-learn/scikit-learn/blob/51a765acfa4c5d1ec05fc4b406968ad233c75162/sklearn/datasets/descr/iris.rst
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target
# print(iris.data);

# print(X)
t = iris.target[:]
print(t)
print(list(iris.target_names))

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
# fit the omdel using X as training data and y as target values
clf.fit(X, y)
# Returns the mean accuracy on the given test data and labels
print("Score: " + str(clf.score(X,y)))
print(clf.predict(X))

# to plot, look at: http://scikit-learn.org/stable/modules/neighbors.html