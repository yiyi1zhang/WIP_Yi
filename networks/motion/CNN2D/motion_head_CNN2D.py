# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 15:59:36 2017

@author: Thomas Kuestner
"""
import os.path
from random import choice

import scipy.io as sio
import numpy as np                  # for algebraic operations, matrices
import keras
from hyperopt import STATUS_OK
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten#, Layer  Dropout, Flatten
#from keras.layers import containers
from keras.models import model_from_json
#from hyperas.distributions import choice, uniform, conditional
#from hyperopt import Trials, STATUS_OK

from keras.layers.convolutional import Convolution2D
#from keras.layers.convolutional import MaxPooling2D as pool2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from keras.layers.convolutional import ZeroPadding2D as zero2d
from keras.regularizers import l2#, activity_l2
#from theano import function

from keras.optimizers import SGD


def createModel(patchSize):
    cnn = Sequential()
    cnn.add(Convolution2D(32,
                            14,
                            14,
                            init='he_normal',
                           # activation='sigmoid',
                            weights=None,
                            border_mode='valid',
                            subsample=(1, 1),
                            W_regularizer=l2(1e-6),
                            input_shape=(1, patchSize[0,0], patchSize[0,1])))
    cnn.add(Activation('relu'))

#    cnn.add(Convolution2D(32,
#                            7,
#                            7,
#                            init='normal',
#                           # activation='sigmoid',
#                            weights=None,
#                            border_mode='valid',
#                            subsample=(1, 1),
#                            W_regularizer=l2(1e-6)))
#                            #input_shape=(1, patchSize[0,0], patchSize[0,1])))
#    cnn.add(Activation('relu'))
    cnn.add(Convolution2D(64 ,                    #learning rate: 0.1 -> 76%
                            7,
                            7,
                            init='he_normal',
                           # activation='sigmoid',
                            weights=None,
                            border_mode='valid',
                            subsample=(1, 1),
                            W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))

    cnn.add(Convolution2D(128 ,                    #learning rate: 0.1 -> 76%
                            3,
                            3,
                            init='he_normal',
                           # activation='sigmoid',
                            weights=None,
                            border_mode='valid',
                            subsample=(1, 1),
                            W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))

    #cnn.add(pool2(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))

    cnn.add(Flatten())
    #cnn.add(Dense(input_dim= 100,
    #              output_dim= 100,
    #              init = 'normal',
    #              #activation = 'sigmoid',
    #              W_regularizer='l2'))
    #cnn.add(Activation('sigmoid'))
    cnn.add(Dense(output_dim= 2,
                  init = 'normal',
                  #activation = 'sigmoid',
                  W_regularizer='l2'))
    cnn.add(Activation('softmax'))
    return cnn

def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None):
    # grid search on batch_sizes and learning rates
    # parse inputs
    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.01] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs
    for iBatch in batchSizes:
        for iLearn in learningRates:
            fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, iBatch, iLearn, iEpochs)

def fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None, iEpochs=None):
    # parse inputs
    batchSize = 64 if batchSize is None else batchSize
    learningRate = 0.01 if learningRate is None else learningRate
    iEpochs = 300 if iEpochs is None else iEpochs

    print('Training CNN')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath,sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename + str(patchSize[0,0]) + str(patchSize[0,1]) +'_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if(os.path.isfile(model_mat)): # no training if output file exists
      return

    # create model
    cnn = createModel(patchSize)

    #opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
    opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                 ModelCheckpoint(sOutPath + os.sep + 'checkpoints' + os.sep + 'checker.hdf5', monitor='val_acc',
                                 verbose=0,
                                 period=5, save_best_only=True),
                 ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1)]
    cnn.compile(loss='categorical_crossentropy', optimizer=opti)
    json_string = cnn.to_json()
    open(model_json, 'w').write(json_string)
    # wei = best_model.get_weights()
    cnn.save_weights(weight_name)
    cnn.save(model_all)

    result = cnn.fit(X_train,
                     y_train,
                     validation_data=[X_test, y_test],
                     nb_epoch=iEpochs,
                     batch_size=batchSize,
                     show_accuracy=True,
                     callbacks=callbacks,
                     verbose=1)

    score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize)

    prob_test = cnn.predict(X_test, batchSize, 0)

    # save model
    json_string = cnn.to_json()
    open(model_json, 'w').write(json_string)
    #wei = cnn.get_weights()
    cnn.save_weights(weight_name, overwrite=True)
    cnn.save(model_all) # keras > v0.7
    model_png_dir = sOutPath + os.sep + "model.png"
    from keras.utils import plot_model
    plot_model(cnn, to_file=model_png_dir, show_shapes=True, show_layer_names=True)

    #matlab
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']

    print('Saving results: ' + model_name)
    sio.savemat(model_name,{'model_settings':model_json,
                            'model':model_all,
                            'weights':weight_name,
                            'acc':acc,
                            'loss':loss,
                            'val_acc':val_acc,
                            'val_loss':val_loss,
                            'score_test':score_test,
                            'acc_test':acc_test,
                            'prob_test':prob_test})

def fPredict(X_test,y_test,model_name, sOutPath, patchSize, batchSize):

    weight_name = model_name[0] + '_weights.h5'
    model_json = model_name[0] + '_json'
    model_all = model_name[0] + '_model.h5'

#    # load weights and model (OLD WAY)
#    conten = sio.loadmat(model_name)
#    weig = content['wei']
#    nSize = weig.shape
#    weigh = []
#
#    for i in drange(0,nSize[1],2):
#    	w0 = weig[0,i]
#    	w1 = weig[0,i+1]
#    	w1=w1.T
#    	w1 = np.concatenate(w1,axis=0)
#
#    	weigh= weigh.extend([w0, w1])
#
#    model = model_from_json(model_json)
#    model.set_weights(weigh)

    # load weights and model (new way)
    #model = model_from_json(model_json)
    model = createModel(patchSize)
    opti = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss',patience=10,verbose=1)]

    model.compile(loss='categorical_crossentropy', optimizer=opti)
    model.load_weights(weight_name)

    # load complete model (including weights); keras > 0.7
    #model = load_model(model_all)

    # assume artifact affected shall be tested!
    #y_test = np.ones((len(X_test),1))

    score_test, acc_test = model.evaluate(X_test, y_test,batch_size=batchSize,show_accuracy=True)
    prob_pre = model.predict(X_test, batchSize, 0)

    #modelSave = model_name[:-5] + '_pred.mat'
    modelSave = model_name[0] + '_pred.mat'
    sio.savemat(modelSave, {'prob_pre':prob_pre, 'score_test': score_test, 'acc_test':acc_test})


###############################################################################
## OPTIMIZATIONS ##
###############################################################################
def fHyperasTrain(X_train, Y_train, X_test, Y_test, patchSize):
    # explicitly stated here instead of cnn = createModel() to allow optimization
    cnn = Sequential()
#    cnn.add(Convolution2D(32,
#                            14,
#                            14,
#                            init='normal',
#                           # activation='sigmoid',
#                            weights=None,
#                            border_mode='valid',
#                            subsample=(1, 1),
#                            W_regularizer=l2(1e-6),
#                            input_shape=(1, patchSize[0,0], patchSize[0,1])))
#    cnn.add(Activation('relu'))

    cnn.add(Convolution2D(32, #64
                            7,
                            7,
                            init='normal',
                           # activation='sigmoid',
                            weights=None,
                            border_mode='valid',
                            subsample=(1, 1),
                            W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))
    cnn.add(Convolution2D(64 ,                    #learning rate: 0.1 -> 76%
                            3,
                            3,
                            init='normal',
                           # activation='sigmoid',
                            weights=None,
                            border_mode='valid',
                            subsample=(1, 1),
                            W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))

    cnn.add(Convolution2D(128 ,                    #learning rate: 0.1 -> 76%
                            3,
                            3,
                            init='normal',
                           # activation='sigmoid',
                            weights=None,
                            border_mode='valid',
                            subsample=(1, 1),
                            W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))

    #cnn.add(pool2(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))

    cnn.add(Flatten())
    #cnn.add(Dense(input_dim= 100,
    #              output_dim= 100,
    #              init = 'normal',
    #              #activation = 'sigmoid',
    #              W_regularizer='l2'))
    #cnn.add(Activation('sigmoid'))
    cnn.add(Dense(input_dim= 100,
                  output_dim= 2,
                  init = 'normal',
                  #activation = 'sigmoid',
                  W_regularizer='l2'))
    cnn.add(Activation('softmax'))

    opti = SGD(lr={{choice([0.1, 0.01, 0.05, 0.005, 0.001])}}, momentum=1e-8, decay=0.1, nesterov=True)
    cnn.compile(loss='categorical_crossentropy',
                        optimizer=opti)

    epochs = 300

    result = cnn.fit(X_train, Y_train,
              batch_size=128, # {{choice([64, 128])}}
              nb_epoch=epochs,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, Y_test))
    score_test, acc_test = cnn.evaluate(X_test, Y_test, verbose=0)

    return {'loss': -acc_test, 'status': STATUS_OK, 'model': cnn, 'trainresult': result, 'score_test': score_test}

## helper functions
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
    r += step
