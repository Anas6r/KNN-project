import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)

x, y = mnist.data, mnist.target

# test and train data:
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
