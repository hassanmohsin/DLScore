from __future__ import print_function
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras import regularizers
from keras import initializers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model
from keras.utils import plot_model
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import random
import os
import itertools
import json
from sklearn.metrics import mean_squared_error
import json
import h5py
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Sensoring outliers
def sensoring(true, pred):
    """ Sensor the predicted data to get rid of outliers"""
    mn = np.min(true)
    mx = np.max(true)
    pred = np.minimum(pred, mx)
    pred = np.maximum(pred, mn)
    
    return pred

def get_hidden_layers():
    x = [128, 256, 512, 768, 1024, 2048]
    hl = []
    
    for i in range(1, len(x)):
        hl.extend([p for p in itertools.product(x, repeat=i+1)])
    
    return hl


# Build the model
def get_model(hidden_layers, dr_rate=0.0, l2_lr=0.01):
    model = Sequential()
    model.add(Dense(hidden_layers[0], kernel_regularizer=regularizers.l2(0.01),
                    activation="relu", 
                    kernel_initializer='glorot_uniform', 
                    input_shape=(280,)))
    model.add(Dropout(dr_rate))
    
    for i in range(1, len(hidden_layers)):
        model.add(Dense(hidden_layers[i], kernel_regularizer=regularizers.l2(0.03),
                        activation="relu",
                        kernel_initializer='glorot_uniform'))
        model.add(Dropout(dr_rate))
   
    model.add(Dense(1, activation="linear"))
    return model


def train(h_layers, plot=True):
    # Load the data
    train_x = np.load('data/train_x.npy')
    train_y = np.load('data/train_y.npy')
    valid_x = np.load('data/valid_x.npy')
    valid_y = np.load('data/valid_y.npy')
    test_x = np.load('data/test_x.npy')
    test_y = np.load('data/test_y.npy')
    print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape, test_y.shape)
#     h_layers = [128, 256, 512, 784]
    model = get_model(h_layers, dr_rate=0.3)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.001)
    rmsprop = keras.optimizers.RMSprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam)
    # model.summary()

    # Network parameters
    epochs = 500
    batch_size = 128
    train_dir = 'train_dir'
    model_name = '-'.join([str(i) for i in h_layers])
    checkpoint_dir = os.path.join(train_dir, model_name)
    if os.path.isfile(os.path.join(checkpoint_dir, 'loss_curve.png')): return
    if not os.path.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)
    # save model
    with open(os.path.join(checkpoint_dir, 'model.json'), 'w') as f:
        f.write(model.to_json())
    # checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:02d}-{val_loss:.2f}.ckpt')
    checkpoint_path = os.path.join(checkpoint_dir, 'weights.h5')

    keras_callbacks = [EarlyStopping(monitor='loss',
                                         min_delta=0,
                                         patience=30,
                                         verbose=0),
                      ModelCheckpoint(checkpoint_path,
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        #save_weights_only=True,
                                        mode='auto', period=1)]

    train_info = model.fit(train_x, train_y, validation_data=[valid_x, valid_y],
                           batch_size=batch_size,
                           epochs=epochs,
                           shuffle=True,
                           verbose=0,
                           #validation_split=0.1,
                           #validation_data=(train_x[valid_index], train_y[valid_index]),
                           callbacks=keras_callbacks)

    # save history
    with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as f:
        json.dump(train_info.history, f)

    # Load the best weights
    model.load_weights(checkpoint_path)
    
    # evaluate train set
    pred = model.predict(train_x).reshape(-1)
    pred = sensoring(train_y, pred)
    pearson_train = pearsonr(pred, train_y)[0]
    mse_train = mean_squared_error(pred, train_y)

    # evaluate test set
    pred = model.predict(test_x).reshape(-1)
    pred = sensoring(test_y, pred)
    pearson_test = pearsonr(pred, test_y)[0]
    mse_test = mean_squared_error(pred, test_y)
    
    # evaluate valid set
    pred = model.predict(valid_x).reshape(-1)
    pred = sensoring(valid_y, pred)
    pearson_valid = pearsonr(pred, valid_y)[0]
    mse_valid = mean_squared_error(pred, valid_y)
    
    results = {'pearson_train': pearson_train, 'pearson_test': pearson_test, 'pearson_valid': pearson_valid,
               'mse_train': mse_train, 'mse_test': mse_test, 'mse_valid': mse_valid}
    with open(os.path.join(checkpoint_dir, 'results.json'), 'w') as f:
        json.dump(results, f)

    print("train pearson: {:.3}, train_mse: {:.3}\ntest pearson: {:.3}, test_mse: {:.3}".format(pearson_train, 
                                                                                                mse_train, 
                                                                                                pearson_test, 
                                                                                                mse_test))
    # plot losses and save
    plt.figure(figsize=(10, 5))
    plt.plot(train_info.history['loss'])
    plt.plot(train_info.history['val_loss'])
    plt.ylim(0, 10)
    plt.legend(['Train MSE', 'Validation MSE'])
    plt.title("Loss curve: " + model_name)
    plt.savefig(os.path.join(checkpoint_dir, 'loss_curve.png'), dpi=300)
    if plot: plt.show()

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('-l', '--layers', nargs='+', default=[])
    args = parser.parse_args()
    hidden_layers = [int(i) for i in args.layers]
       
    train(hidden_layers)
