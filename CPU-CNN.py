# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_5pcekWBbXuepq--Us0IsSuEZMcK71It
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from pandas import read_csv


climbingdown_forearm = read_csv("climbingdown_forearm.csv")
climbingup_forearm = read_csv("climbingup_forearm.csv")
# jumping = read_csv("jumping_head.csv")
lying_forearm = read_csv("lying_forearm.csv")
running_forearm = read_csv("running_forearm.csv")
sitting_forearm = read_csv("sitting_forearm.csv")
standing_forearm = read_csv("standing_forearm.csv")
walking_forearm = read_csv("walking_forearm.csv")

trainx_size=20000
image_length = 100
train = []
# for x in (climbingdown,climbingup,jumping,lying,running,sitting,standing,walking):
for x in (climbingdown_forearm,climbingup_forearm,lying_forearm,running_forearm,sitting_forearm,standing_forearm,walking_forearm):

#     x = x[['attr_time','attr_x','attr_y','attr_z']]
    x = x[['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','mag_x','mag_y','mag_z']]
    x = x[:trainx_size]
    x = x.values.tolist()
    for x_ in x:
        train.append(x_)
print(len(train))
trainX = []
# for i in range(0,int(trainx_size*8),image_length):
for i in range(0,int(trainx_size*7),image_length):
    trainX.append(train[i:i+image_length])
trainX = np.asarray(trainX)
print(np.shape(trainX))
print(trainX)

testx_size=trainx_size/5
test = []
# for x in (climbingdown,climbingup,jumping,lying,running,sitting,standing,walking):
for x in (climbingdown_forearm,climbingup_forearm,lying_forearm,running_forearm,sitting_forearm,standing_forearm,walking_forearm):
#     x = x[['attr_time','attr_x','attr_y','attr_z']]
    x = x[['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','mag_x','mag_y','mag_z']]
    x = x[trainx_size:int(trainx_size+testx_size)]
    x = x.values.tolist()
    for x_ in x:
        test.append(x_)
testX = []
# for i in range(0,int(testx_size*8),image_length):
for i in range(0,int(testx_size*7),image_length):
    testX.append(test[i:i+image_length])
testX = np.asarray(testX)
print(np.shape(testX))

import numpy as np
image_length = 100
# trainy_size= int(trainx_size*8/image_length)
trainy_size= int(trainx_size*7/image_length)
trainy_step= int(trainx_size/image_length)
# trainy = np.zeros((trainy_size,8),dtype='i')
trainy = np.zeros((trainy_size,7),dtype='i')
j=0
for i in range(0, trainy_size, trainy_step):
    trainy[i:i+trainy_step,j]= 1
    j=j+1
print(np.shape(trainy))
print(trainy)
    
# testy_size= int(testx_size*8/image_length)
testy_size= int(testx_size*7/image_length)
testy_step= int(testx_size/image_length)
# testy = np.zeros((testy_size,8),dtype='i')
testy = np.zeros((testy_size,7),dtype='i')
j=0
for i in range(0, testy_size, testy_step):
    testy[i:i+testy_step,j]= 1
    j=j+1
print(np.shape(testy))
print(testy)

print(trainX.shape[1], trainX.shape[2], trainy.shape[1])

from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
 
# # load a single file as a numpy array
# def load_file(filepath):
# 	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
# 	return dataframe.values
 
# # load a list of files and return as a 3d numpy array
# def load_group(filenames, prefix=''):
# 	loaded = list()
# 	for name in filenames:
# 		data = load_file(prefix + name)
# 		loaded.append(data)
# 	# stack group so that features are the 3rd dimension
# 	loaded = dstack(loaded)
# 	return loaded
 
# # load a dataset group, such as train or test
# def load_dataset_group(group, prefix=''):
# 	filepath = prefix + group + '/Inertial Signals/'
# 	# load all 9 files as a single array
# 	filenames = list()
# 	# total acceleration
# 	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
# 	# body acceleration
# 	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
# 	# body gyroscope
# 	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
# 	# load input data
# 	X = load_group(filenames, filepath)
# 	# load class output
# 	y = load_file(prefix + group + '/y_'+group+'.txt')
# 	return X, y
 
# # load the dataset, returns train and test X and y elements
# def load_dataset(prefix=''):
# 	# load all train
# 	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
# 	print(trainX.shape, trainy.shape)
# 	# load all test
# 	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
# 	print(testX.shape, testy.shape)
# 	# zero-offset class values
# 	trainy = trainy - 1
# 	testy = testy - 1
# 	# one hot encode y
# 	trainy = to_categorical(trainy)
# 	testy = to_categorical(testy)
# 	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
# 	return trainX, trainy, testX, testy

# fit and evaluate a model
import time
 
start = time.clock()


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 100, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3,activation='relu'))
#     model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
#     model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
#     pred = model.predict(testX) 
#     pred = np.argmax(pred, axis = 1)[:5]  
#     label =testy[0:5,:]  

#     print(pred) 
#     print(label)
    return accuracy
 
# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
#     load data
#     trainX, trainy, testX, testy = load_dataset()
#     repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
    
# run the experiment
run_experiment()
elapsed = (time.clock() - start)
print("Time used:",elapsed)