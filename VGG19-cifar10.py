# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 22:03:17 2018

@author: Bowen Deng
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import keras

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import pydot

import keras.backend.tensorflow_backend as KTF
import os



'''
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':1})))
'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data():
    
    train1=unpickle('cifar-10-batches-py/data_batch_1')
    train2=unpickle('cifar-10-batches-py/data_batch_2')
    train3=unpickle('cifar-10-batches-py/data_batch_3')
    train4=unpickle('cifar-10-batches-py/data_batch_4')
    train5=unpickle('cifar-10-batches-py/data_batch_5')
   

    X_train_orig=np.vstack((train1[b'data'],train2[b'data'],train3[b'data'],train4[b'data'],train5[b'data']))
    Y_train_orig=np.hstack((train1[b'labels'],train2[b'labels'],train3[b'labels'],train4[b'labels'],train5[b'labels']))
    
    X_train_change=X_train_orig.reshape(50000,3,32,32).transpose((0,2,3,1))
    Y_train_change=Y_train_orig.reshape(50000,1)
    test=unpickle('cifar-10-batches-py/test_batch')
    X_test_change=test[b'data'].reshape(10000,3,32,32).transpose((0,2,3,1))
    Y_test_change=np.array(test[b'labels']).reshape(10000,1)
    
    return X_train_change,Y_train_change,X_test_change, Y_test_change

# load data and one hot
X_train,Y_train,X_test,Y_test=load_data()
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)
#data preprocessing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train=X_train-np.mean(X_train)
X_test=X_test-np.mean(X_test)

num_classes=10
batch_size=128
epochs=220
iterations=391
dropout=0.5
weight_decay=0.0001

#Build model
#block 1
model=Sequential()
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=(2,2),name='block1_pool'))

#block2

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

#Cifar10
model.add(Flatten(name='flatten'))
model.add(Dense(4096,use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay),kernel_initializer=he_normal(),name='fc_cifa10'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))  
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))      
model.add(Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10'))        
model.add(BatchNormalization())
model.add(Activation('softmax'))

#load weights by name
model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels.h5',by_name=True)

#optimizers
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])



log_filepath ='D:/vgg19_retrain_logs/'

def scheduler(epoch):
    if epoch < 70:
        return 0.1
    if epoch < 140:
        return 0.01
    if epoch< 170:
        return 0.001
    return 0.0001

tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

datagen.fit(X_train)


model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(X_test, Y_test))









model.save('Perfect-VGG19-CIFAR10.h5')

'''



preds = model.evaluate(x=X_test,y=Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

'''

preds = model.evaluate(x=X_test,y=Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))











