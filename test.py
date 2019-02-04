#! /usr/bin/env python3
import numpy as np
import math
import mlp
import importlib
import pandas as pd
from sklearn.datasets import load_iris

importlib.reload(mlp)

# data = load_iris()
# X = data.data
# Y = data.target
size = 1000

data = pd.read_csv("../../mnist_train.csv")
X = data.values[:size,1:]
Y = data.values[:size,0]

# X = np.random.normal(size=(1,20))
a = mlp.MLP(784,500,500,10,c=1,init="glorot")
a.train(X, Y, 10, 10, 0.0000001)


# a.fprop(X)


# Y = np.array([0])
# Y = np.array([0,1,2,3,4,1,1])
# a.bprop( Y, 0.1)
