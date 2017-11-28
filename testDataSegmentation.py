#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:25:19 2017
This script is written to divide and combine the test data into classes to avoid over size files that can not be uploaded to the github.
@author: shahnawaz
"""

import numpy as np
# converting the hot one coded outputs to the indices of the active classes
def getClassId(classes):
    classIds = np.zeros((classes.shape[0]))
    for instance in range (classes.shape[0]):
        classIds[instance] = np.argmax(classes[instance,:])
    return classIds    
# save the data for different classes
def saveTestData(xtest,ytest,classNames):
    classIds = getClassId(ytest)
    for idx, classname in enumerate(classNames):
        print(idx)
        inds4idx = np.where(classIds==idx)
        print(inds4idx)
        testFileName = 'TestData/'+classname + '.npy'
        np.save(testFileName,np.squeeze(xtest[inds4idx,:,:,:][:,:,:,np.newaxis]))
def shuffle_XY_paths(X,Y):   # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0] )
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:,:]
        newY[i] = Y[idx[i],:]
    return newX, newY
def loadTestData(train_percentage=0.8, preproc=False):
    # pre-allocate memory for speed (old method used np.concatenate, slow)
    X_test = np.zeros((400, 128, 300, 1))
    Y_test = np.zeros((400,50))
    testCount = 0
    allClasses = np.load('Class_names.npy')
    for idx, classname in enumerate(allClasses):
        thisData = np.load('TestData/'+classname+'.npy')
        thisData = thisData[:,:,:,np.newaxis]
        for idx2 in range(thisData.shape[0]):
            X_test[testCount,:,:,:] = thisData[idx2,:,:,:]
            Y_test[testCount,idx] = 1
            testCount += 1
    print("Shuffling test data...")
    X_test, Y_test = shuffle_XY_paths(X_test, Y_test)
    return  X_test, Y_test
