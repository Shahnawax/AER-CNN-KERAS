'''
Evaluationg Audio Event Recognition Model
Author: Muhammad Shahnawaz
This script is to load and evaluate a pretrained model in keras saved as model.h5 for test data saved as X_test.npy with ground truth files present in Y_test.npy
'''
# importing the dependencies
import time, random, numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
# setting a random seed for the reproducibility purposes
random.seed(611)
# creating a time wrapper to calculate the elapsed time in an operation
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print (self.name)
        print( (time.time() - self.tstart),' seconds')
# converting the hot one coded outputs to the indices of the active classes
def getClassId(classes):
    classIds = np.zeros((classes.shape[0]))
    for instance in range (classes.shape[0]):
        classIds[instance] = np.argmax(classes[instance,:])
    return classIds
# given ground truth and prediction variables calculating the confusion matrix.
def obtainConfusionMatrix(GT,PR):
    GT = getClassId(GT)
    PR = getClassId(PR)
    cm = confusion_matrix(GT,PR)
    return cm
def printingAccuraciesForEachClass(cM, labels,title):
    # normalizing the confusionMatrix for showing the probabilities
    print(title)
    cmNormalized = np.around((cM/cM.sum(axis=1)[:,None])*100,2) # rounding to second decimal
    width, height = cM.shape 
    print('Accuracy for each class is given below.')
    for predicted in range(width):
        for real in range(height):
            if(predicted == real):
                print(labels[predicted].ljust(20)+ ':', cmNormalized[predicted,real], '%')
# Entering to the main code
if __name__ == '__main__':
    # loading the model and data (Labels, Groud Truth and X_test for prediction)
    model = load_model('model.h5')
    labels = np.load('Class_names.npy')
    gtTest = np.load('Y_test.npy')
    X_test = np.load('X_test.npy')
    # predicting the outputs for the test data
    testPredictions =  model.predict(X_test,verbose=2)
    # calculating the confusion matrix to print the performance for each class
    testCM = obtainConfusionMatrix(gtTest,testPredictions)
    # printing the performances of the class.
    printingAccuraciesForEachClass(testCM,labels,'TestCM')
