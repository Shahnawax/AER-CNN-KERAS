from __future__ import print_function

''' 
Audio Event Recognition (AER)
Author: Muhammad Shahnawaz

This is a proof of the concept CNN model for the audio even recognition.
Before running this make sure that the preprocessing has been done for all the file and you have the files saved as
./Preproc/className/fileName.npy
This class assumes that the preprocessing is done already and there is a Preproc/ directory available containing all the preprocessed data for ECS-50 data base.
Preprocessing is done using n_mels = 128 and the amplitudes are converted to dB scales using librosa.logamplitude() function. 
See the "preprocess_data.py" file for further details. 
'''

# importing the required dependencies 
import librosa, os, testDataSegmentation, random, numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2 as L2
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
mono=True
# seeding the random function for the reproducibility purposes
random.seed(611)
# function to get all the names of the classes
def get_class_names(path="Preproc/"):  
    class_names = os.listdir(path)
    return class_names
# function to get the count of total, training and testing files
def get_total_files(path="Preproc/",train_percentage=0.8): 
    sum_total = 0
    sum_train = 0
    sum_test = 0
    subdirs = os.listdir(path)
    for subdir in subdirs:
        files = os.listdir(path+subdir)
        n_files = len(files)
        sum_total += n_files
        n_train = int(train_percentage*n_files)
        n_test = n_files - n_train
        sum_train += n_train
        sum_test += n_test
    return sum_total, sum_train, sum_test
# function to obtain the sample diemensions
def get_sample_dimensions(path='Preproc/'):
    classname = os.listdir(path)[0]
    files = os.listdir(path+classname)
    infilename = files[0]
    audio_path = path + classname + '/' + infilename
    melgram = np.load(audio_path)
    return melgram.shape
# encode the class labels using one hot coding
def encode_class(class_name, class_names):  
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None
# randomizing the data.
def shuffle_XY_paths(X,Y,paths):
    assert (X.shape[0] == Y.shape[0] )
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:]
        newY[i] = Y[idx[i],:]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths

'''
So we read data to create trainming and testing data variables in this function. 
Once created these data variables are shuffeled in order to mix the data for different classes.
The splitting between training and test data happens before creation because we want to keep the number for test samples from each class same.
'''
def build_datasets(train_percentage=0.8, path='Preproc'):
    # the default values, we assume the 
    class_names = get_class_names(path=path)
    print("class_names = ",class_names)

    total_files, total_train, total_test = get_total_files(path=path, train_percentage=train_percentage)
    print("total files = ",total_files)

    nb_classes = len(class_names)

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    inputShape = get_sample_dimensions(path=path)  # Find out the 'shape' of each data file
    X_train = np.zeros((total_train, inputShape[1], inputShape[2], inputShape[3]))   
    Y_train = np.zeros((total_train, nb_classes))  
    X_test = np.zeros((total_test, inputShape[1], inputShape[2], inputShape[3]))  
    Y_test = np.zeros((total_test, nb_classes))  
    # creating variables to return
    paths_train = []
    paths_test = []
    train_count = 0
    test_count = 0
    for idx, classname in enumerate(class_names):
        this_Y = np.array(encode_class(classname,class_names) )
        this_Y = this_Y[np.newaxis,:]
        class_files = os.listdir(path+classname)
        n_files = len(class_files)
        n_load =  n_files
        n_train = int(train_percentage * n_load)
        #printevery = 100
        for idx2, infilename in enumerate(class_files[0:n_load]):          
            audio_path = path + classname + '/' + infilename
            melgram = np.load(audio_path)
            sr = 44100
            melgram = melgram[:,:,:,0:inputShape[3]]   # just in case files are differnt sizes: clip to first file size
            if (idx2 < n_train):
                X_train[train_count,:,:] = melgram
                Y_train[train_count,:] = this_Y
                paths_train.append(audio_path)     # list-appending is still fast. (??)
                train_count += 1
            else:
                X_test[test_count,:,:] = melgram
                Y_test[test_count,:] = this_Y
                paths_test.append(audio_path)
                test_count += 1

    print("Shuffling order of data...")
    X_train, Y_train, paths_train = shuffle_XY_paths(X_train, Y_train, paths_train)
    X_test, Y_test, paths_test = shuffle_XY_paths(X_test, Y_test, paths_test)
    return X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr
# model CNN model defenition 
def cnnModel():
    inputShape = (128,300,1)
    numClasses = 50
    model = Sequential() # initializing a Sequential model
    # adding first convolutionial layer with 64 filters and 3 by 3 kernal size, using the rectifier linear unit as the activation
    model.add(Conv2D(64, (3,3),input_shape=inputShape,activation='relu',padding='valid'))
    # adding a batch normalization and maxpooling layer 3 by 5 (Note: We are compressing the 2nd diemension more in order to get a more square shape)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,5)))
    # adding second convolutionial layer with 32 filters and 1 by 3 kernal size, using the rectifier linear unit as the activation
    model.add(Conv2D(32, (1,3),input_shape=(numOfRows, numOfColumns,1),activation='relu',padding = 'valid'))
    # adding the first convolutionial layer with 32 filters and 3 by 3 kernal size, using the rectifier linear unit as the activation
    model.add(Conv2D(32, (3,3),input_shape=(numOfRows, numOfColumns,1),activation='relu',padding = 'valid'))
    # adding batch normalization layer
    model.add(BatchNormalization())
    # adding forth convolutionial layer with 32 filters and 1 by 3 kernal size, using the rectifier linear unit as the activation
    model.add(Conv2D(32, (1,3),input_shape=(numOfRows, numOfColumns,1),activation='relu',padding = 'valid'))
    # adding batch normalization and max pooling layers
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,3)))
    # adding fift convolutionial layer with 64 filters and 3 by 3 kernal size, using the rectifier linear unit as the activation
    model.add(Conv2D(32, (3,3),input_shape=(numOfRows, numOfColumns,1),activation='relu',padding = 'valid'))
    # adding batch normalization and max pooling layers
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    # flattening the output in order to apply the fully connected layer
    model.add(Flatten())
    # adding a drop out for the regularizaing purposes
    model.add(Dropout(0.2))
    #adding a fully connected layer 128 filters
    model.add(Dense(128, activation='relu'))   
    # adding a drop out for the regularizaing purposes
    model.add(Dropout(0.2))
    # adding softmax layer for the classification
    model.add(Dense(numClasses, activation='softmax'))
    # Compiling the model to generate a model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 
    
# entering into the main
if __name__ == '__main__':
    
    # Load the data
    x_train, y_train, paths_train, X_test, Y_test, paths_test, class_names, sr = build_datasets()
    # splitting training and validation data using random indices (as random is supposed to generate a unifrom distribution so we can assume there will be 80% number below 0.8).
    trainSampleIndices = np.random.rand(x_train.shape[0]) < 0.8
    X_train = x_train[trainSampleIndices,:,:,:]
    X_validation = x_train[~trainSampleIndices,:,:,:] 
    Y_train = y_train[trainSampleIndices,:]
    Y_validation = y_train[~trainSampleIndices,:] 
    # Generating the model
    model = cnnModel()
    # visualizing the model
    model.summary()
    # Training the model
    model.fit(X_train,Y_train, validation_data=(X_validation,Y_validation),epochs=20,batch_size=20,verbose=2)
    # Testing the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score KRL:', score[0])
    print('Test accuracyKRL:', score[1])
    # saving the model and the variables for further use and reproduction    
    model.save('model.h5')
    np.save('Class_names.npy',class_names)
    testDataSegmentation.saveTestData(X_test, Y_test, class_names)
