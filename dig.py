# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 12:51:13 2016

@author: sky
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
 
np.random.seed(36) 
 
batch_size = 128
nb_classes = 10
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
 
# Read the train and test datasets
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
 
# seperate out label
Y = train["label"]  
train.drop(labels = "label", axis = 1, inplace = True)
 
# convert data to np array
train = train.values
test = test.values
 
X_train = train.reshape(train.shape[0], 1, img_rows, img_cols)
X_test = test.reshape(test.shape[0], 1, img_rows, img_cols)
 

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
 
# convert class vectors to binary class matrices (ie one-hot vectors)
Y_train = np_utils.to_categorical(Y, nb_classes)
 
def CNN_model():
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='valid',input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    return model
 
cnn_model = CNN_model()
 
# train the model 
cnn_model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 2)
pred = cnn_model.predict_classes(X_test) 
# create a submission file
np.savetxt(
		'submission.csv', 
		np.c_[range(1,len(pred)+1),pred], 
		delimiter = ',', 
		header = 'ImageId,Label', 
		comments = '', 
		fmt = '%d'
	)