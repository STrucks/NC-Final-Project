# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 17:12:12 2018

@author: Valentin
"""

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.optimizers import SGD
from sklearn import datasets
from numpy import random
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

seed=3
np.random.seed(seed)


NUM_HIDDEN_LAYERS = 2
SIZE_LAYERS = 100


#iris = datasets.load_iris()
#X=iris.data
#Y=iris.target

def baseline_model():

    model = Sequential()
    model.add(Dense(SIZE_LAYERS,input_dim=2,activation='sigmoid'))
    if NUM_HIDDEN_LAYERS==2:
        model.add(Dense(SIZE_LAYERS,activation='sigmoid'))
#        model.add(Dense(SIZE_LAYERS,activation='sigmoid'))
    model.add(Dense(3,activation='softmax'))
    opt = SGD(momentum=0.9,nesterov=False)
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    return model

def load_artificial_ds2():
    import math
    data = random.rand(500, 2) * 8 - 4
    labels = []
    for x, y in data:
        if math.sqrt(x ** 2 + y ** 2) < 2:
            labels.append(0)
        elif math.sqrt(x ** 2 + y ** 2) < 3:
            labels.append(0)
        elif math.sqrt(x ** 2 + y ** 2) < 4:
            labels.append(1)
        else:
            labels.append(2)
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        colors = ['b', 'r', 'g', 'y']
        for row, label in zip(data, labels):
            plt.plot(row[0], row[1], colors[label] + '.')
        plt.show()
    return data, labels

data = load_artificial_ds2()
X = data[0]
Y=data[1]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=baseline_model, epochs=1000, batch_size=50, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
