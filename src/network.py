import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import librosa
from librosa import display

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

filePath = '../data/Audio_Speech_Actors_01-24'

def load_files(path=filePath):
    lst = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            try:
                #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                #print(os.path.join(subdir,file))
                X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
                file = file[6:8]
                arr = mfccs, file
                lst.append(arr)
            # If the file is not valid, skip it
            except ValueError:
                continue
    return zip(*lst)
    
def main():
    X, y = load_files()
    X = np.asarray(X)
    y = np.asarray(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    

if __name__ == "__main__":
    main()
