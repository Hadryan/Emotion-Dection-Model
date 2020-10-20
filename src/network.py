import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import librosa
from librosa import display

from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Softmax, Dropout, Module, Flatten
from torch.optim import Adam, SGD

import tensorflow as tf
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

filePath = '../data/Audio_Speech_Actors_01-24'

def load_files(path=filePath):
    lst = []

    for subdir, dirs, files in os.walk(path):
        for file in files:
            try:
                    # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                X, sample_rate = librosa.load(os.path.join(subdir, file),
                                                  res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                                         n_mfcc=40).T, axis=0)
                    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                file = int(file[7:8]) - 1
                arr = mfccs, file
                lst.append(arr)
                # If the file is not valid, skip it
            except ValueError as err:
                print(err)
                continue

        # Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
    X, y = zip(*lst)

        # Array conversion
    X, y = np.asarray(X), np.asarray(y)
    
    return X, y
    
def main():
    print('Loading Files...')
    X, y = load_files() # load files
    print('Done...')
    
    # split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)
    
    # build model
    model = Sequential()

    model.add(Conv1D(128, 5,padding='same',input_shape=(40,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    print(model.summary())
    
    opt = keras.optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    cnn=model.fit(x_traincnn, y_train, batch_size=16, epochs=1024, validation_data=(x_testcnn, y_test))
    
    plt.plot(cnn.history['loss'])
    plt.plot(cnn.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.close()
    
    plt.plot(cnn.history['accuracy'])
    plt.plot(cnn.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.close()
    

if __name__ == "__main__":
    main()
