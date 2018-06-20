# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:34:06 2018

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


iris = datasets.load_iris()
X=iris.data
Y=iris.target

def baseline_model():

    model = Sequential()
    model.add(Dense(SIZE_LAYERS,input_dim=4))
    if NUM_HIDDEN_LAYERS==2:
        model.add(Dense(SIZE_LAYERS))
    model.add(Dense(3,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    return model

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=75, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))