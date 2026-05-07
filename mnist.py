import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

mnist = fetch_openml('mnist_784', as_frame=False)
model = KNeighborsClassifier()


x, y = mnist.data, mnist.target

# test and train data:
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6, 7, 2, 9, 8]}]
model.score(x_test, y_test)
knn = GridSearchCV(model, param_grid,cv=8)
knn.best_estimator_.fit(x_train,y_train)

knn.fit(x_train[:10000], y_train[:10000])
