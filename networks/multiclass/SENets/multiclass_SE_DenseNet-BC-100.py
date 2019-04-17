'''
SE-DenseNet-BC-100
SE-Blocks in the transitionlayers

'''

import os.path
import scipy.io as sio
import numpy as np
import keras
from keras.layers import Input
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten

from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2  # , activity_l2

from networks.multiclass.SENets.densely_connected_cnn_blocks import *

from keras.optimizers import SGD



def createModel(patchSize, numClasses):
    # ResNet-56 based on CIFAR-10, for 32x32 Images
    print(K.image_data_format())

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    growthRate_k = 12
    compressionFactor = 0.5

    input_tensor = Input(shape=(patchSize[0], patchSize[1], 1))

    # first conv layer
    x = Conv2D(2*growthRate_k, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv1')(input_tensor)

    # 1. Dense Block
    x, numFilters = dense_block(x, numInputFilters=2*growthRate_k, numLayers=16, growthRate_k=growthRate_k, bottleneck_enabled=True)

    # Transition Layer
    x, numFilters = transition_SE_layer(x, numFilters, compressionFactor=compressionFactor, se_ratio=128)

    # 2. Dense Block
    x, numFilters = dense_block(x, numInputFilters=numFilters, numLayers=16, growthRate_k=growthRate_k, bottleneck_enabled=True)

    #Transition Layer
    x, numFilters = transition_SE_layer(x, numFilters, compressionFactor=compressionFactor, se_ratio=128)

    #3. Dense Block
    x, numFilters = dense_block(x, numInputFilters=numFilters, numLayers=16, growthRate_k=growthRate_k, bottleneck_enabled=True)

    # SE Block
    x = squeeze_excitation_block(x, ratio=128)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    # global average pooling
    x = GlobalAveragePooling2D(data_format='channels_last')(x)

    # fully-connected layer
    output = Dense(units=numClasses,
                   activation='softmax',
                   kernel_initializer='he_normal',
                   name='fully-connected')(x)

    # create model
    cnn = Model(input_tensor, output, name='ResNet-56')

    return cnn


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None):
    # grid search on batch_sizes and learning rates
    # parse inputs
    batchSizes = [64] if batchSizes is None else batchSizes
    learningRates = [0.01] if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs

    # change the shape of the dataset -> at color channel -> here one for grey scale
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    #y_train = np.asarray([y_train[:], np.abs(np.asarray(y_train[:], dtype=np.float32) - 1)]).T
    #y_test = np.asarray([y_test[:], np.abs(np.asarray(y_test[:], dtype=np.float32) - 1)]).T

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
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    model_name = sOutPath + os.sep + sFilename + '_lr_' + str(learningRate) + '_bs_' + str(batchSize)
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '.json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        return

    #number of classes
    numClasses = np.shape(y_train)[1]

    # create model
    cnn = createModel(patchSize, numClasses=numClasses)

    # opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
    opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]

    cnn.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

    result = cnn.fit(X_train,
                     y_train,
                     validation_data=[X_test, y_test],
                     epochs=iEpochs,
                     batch_size=batchSize,
                     callbacks=callbacks,
                     verbose=1)

    score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize, verbose=1)

    prob_test = cnn.predict(X_test, batchSize, 0)

    # save model
    json_string = cnn.to_json()
    open(model_json, 'w').write(json_string)

    # wei = cnn.get_weights()
    cnn.save_weights(weight_name, overwrite=True)
    cnn.save(model_all) # keras > v0.7
    model_png_dir = sOutPath + os.sep + "model.png"
    from keras.utils import plot_model
    plot_model(cnn, to_file=model_png_dir, show_shapes=True, show_layer_names=True)

    # matlab
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']

    print('Saving results: ' + model_name)
    sio.savemat(model_name, {'model_settings': model_json,
                             'model': model_all,
                             'weights': weight_name,
                             'acc': acc,
                             'loss': loss,
                             'val_acc': val_acc,
                             'val_loss': val_loss,
                             'score_test': score_test,
                             'acc_test': acc_test,
                             'prob_test': prob_test})


def fPredict(X_test, y_test, model_name, sOutPath, patchSize, batchSize):
    weight_name = model_name[0] + '_weights.h5'
    model_json = model_name[0] + '.json'
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
    # model = model_from_json(model_json)
    model = createModel(patchSize)
    opti = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = []
    callbacks.append(
        ModelCheckpoint(sOutPath + os.sep + 'checkpoints' + os.sep + 'checker.hdf5', monitor='val_acc', verbose=0,
                        period=1, save_best_only=True))  # overrides the last checkpoint, its just for security
    callbacks.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1))

    model.compile(loss='categorical_crossentropy', optimizer=opti)
    model.load_weights(weight_name)

    # load complete model (including weights); keras > 0.7
    # model = load_model(model_all)

    # assume artifact affected shall be tested!
    # y_test = np.ones((len(X_test),1))

    score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize, show_accuracy=True)
    prob_pre = model.predict(X_test, batchSize, 0)

    # modelSave = model_name[:-5] + '_pred.mat'
    modelSave = model_name[0] + '_pred.mat'
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})


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

    cnn.add(Convolution2D(32,  # 64
                          7,
                          7,
                          init='normal',
                          # activation='sigmoid',
                          weights=None,
                          border_mode='valid',
                          subsample=(1, 1),
                          W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))
    cnn.add(Convolution2D(64,  # learning rate: 0.1 -> 76%
                          3,
                          3,
                          init='normal',
                          # activation='sigmoid',
                          weights=None,
                          border_mode='valid',
                          subsample=(1, 1),
                          W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))

    cnn.add(Convolution2D(128,  # learning rate: 0.1 -> 76%
                          3,
                          3,
                          init='normal',
                          # activation='sigmoid',
                          weights=None,
                          border_mode='valid',
                          subsample=(1, 1),
                          W_regularizer=l2(1e-6)))
    cnn.add(Activation('relu'))

    # cnn.add(pool2(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))

    cnn.add(Flatten())
    # cnn.add(Dense(input_dim= 100,
    #              output_dim= 100,
    #              init = 'normal',
    #              #activation = 'sigmoid',
    #              W_regularizer='l2'))
    # cnn.add(Activation('sigmoid'))
    cnn.add(Dense(input_dim=100,
                  output_dim=2,
                  init='normal',
                  # activation = 'sigmoid',
                  W_regularizer='l2'))
    cnn.add(Activation('softmax'))

    #opti = SGD(lr={{choice([0.1, 0.01, 0.05, 0.005, 0.001])}}, momentum=1e-8, decay=0.1, nesterov=True)
    #cnn.compile(loss='categorical_crossentropy', optimizer=opti)

    epochs = 300

    result = cnn.fit(X_train, Y_train,
                     batch_size=128,  # {{choice([64, 128])}}
                     nb_epoch=epochs,
                     show_accuracy=True,
                     verbose=2,
                     validation_data=(X_test, Y_test))
    score_test, acc_test = cnn.evaluate(X_test, Y_test, verbose=0)

    #return {'loss': -acc_test, 'status': STATUS_OK, 'model': cnn, 'trainresult': result, 'score_test': score_test}


## helper functions
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
    r += step
