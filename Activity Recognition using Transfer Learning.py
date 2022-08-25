# We will be using InceptionNet V3 for Transfer Learning
#Initially we will manually label images and work on Data augmentation Normalization to avoid overfitting of the model

#Data source- https://www.kaggle.com/datasets/huan9huan/walk-or-run

#import libs
import cv2
import os
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import *
from keras.callbacks import EarlyStopping
from keras import regularizers,optimizers
from keras.callbacks import LearningRateScheduler
from keras import *
from keras.applications.inception_v3 import InceptionV3
import seaborn as sns
import sklearn
from sklearn.metrics import *

## Mounting the drive ##
from google.colab import drive
drive.mount('/content/drive')

# TRAIN_RUN

# Ensures each png file is extracted from the folder
train_run = glob(os.path.join('/content/drive/MyDrive/action_recognition/walk_run_train/train_run', "*.png"))

# TRAIN_WALK
train_walk = glob(os.path.join('/content/drive/MyDrive/action_recognition/walk_run_train/train_walk', "*.png"))

# ADD TRAIN_WALK AND TRAIN_RUN INTO A DATAFRAME
train = pd.DataFrame()
train['file'] = train_run + train_walk
train.head()

# TEST_RUN
# ../input/walk_or_run_train/train/run
test_run = glob(os.path.join('/content/drive/MyDrive/action_recognition/walk_run_test/run_test', "*.png"))

# TRAIN_WALK
test_walk = glob(os.path.join('/content/drive/MyDrive/action_recognition/walk_run_test/walk_test', "*.png"))

# ADD TRAIN_WALK AND TRAIN_RUN INTO A DATAFRAME
test = pd.DataFrame()
test['file'] = test_run + test_walk
test.head()

## Labeling images ##
#TRAIN LABELS
train['label'] = [1 if i in train_run else 0 for i in train['file']]
train.head()

#TEST LABELS
test['label'] = [1 if i in test_run else 0 for i in test['file']]
test.head()

# TRAIN RUN AND WALK IMAGES- having a look at them
plt.figure(figsize=(16,16))
#Checking out an image in train_run with a random index 1- can change this index to another as well as dataframe to train_walk
plt.imshow(cv2.imread(train_run[1]))

## Data Preprocessing ##

#Data augmentation- Introducing a change
#Reshaping the images, Adding a slight shift to pictures, Reshuffling the images etc.

def dataug(files, labels, batch_size=10,randomized=True, random_seed=1):
    randomizer = np.random.RandomState(random_seed)
    img_batch = []
    label_batch = []
    while True:
        ind = np.arange(len(files))
        if randomized:
#ReShuffling the indices
            randomizer.shuffle(ind)
        for index in ind:
#We are iterating through the images on that index but ':' indicates that we are leaving the color channels as they are
#Dividing the value by 255 normalizes the pixel intensity in each of the images
            image = cv2.imread(files[index])[:,:,0:3]/255
            label = labels[index]
#Adding the normalized image to the empty array made previously
            img_batch.append(image)
            label_batch.append(label)
            if len(img_batch) == batch_size:
#yield is equivalent to return used in user defined functions,
#but the difference is that it returns generator functions for ex- here it is returning arrays as generator functions
                yield np.array(img_batch), np.array(label_batch)
#The following statements ensure that the arrays are empty for next batch of 50 pictures
                img_batch = []
                label_batch = []
        
        if len(img_batch) > 0:
                yield np.array(img_batch), np.array(label_batch)
                img_batch = []
                label_batch = []
                
## Load InceptionNet V3 and train ## This code chunk runs fully till the dense layer addition

#We are not taking the top layer from the inceptionNet model so- we specify include_top as False
#We want to use weights from imagenet- We don't want to use input tensors and not going to set any specifications for input shapes except for the channel we want to use- RBG channel
# 3 in input_shape is for color channels
#Pooling for our current use case is fine with average- need to check the use case for min and max pooling
#The original model was trained for 1000 classes, so we have used 1000 classes
transferred=InceptionV3(include_top=False,weights='imagenet',input_tensor=None,input_shape=(None,None,3),pooling='avg',classes=1000)

#Making a sequential model as we want to stack the layers one-by-one in a sequence
model=Sequential()

## Adding the layers for the model ##

#As we removed the top layer from the inceptionNet model we need to add the input layer
model.add((InputLayer(None,None,3)))

#Now we are going to add the rest of the InceptionNet
model.add(transferred)

#Adding a dropout layer as this will help reducing the chances of overfitting the model
#During the fine tuning we can change the value for Dropout
model.add(Dropout(0.5))

#Adding the dense layer where the actual factorization/classification takes place
#This is a fully connected layer
#Using sigmoid as activation function because that is used for classification
model.add(Dense(1,activation='sigmoid'))

#----- After the code-run at this step- InceptionNet is downloaded -----#

# We are going to train only the top layers of the model after freezing the model

#Freezing the model
transferred.trainable=False

#We will compile this model at this step with optimizer adam
#As this is a binary classification problem we will add loss as binary crossentropy
#Metric is chosen as accuracy
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

#Batch file selection is random- can go with any other value
batch_size=500

#To avoid system crash- the recommended value is 50- can reduce it to 5 or 10
epochs=50

#-----Need to run code again here to compile model and set the parameters -----#

# Fitting the model
#Recommended to use GPU runtime at this step by clicking on google colab 'Runtime' tab

# Calling the function created earlier to preprocess images
# Setting steps per epoch and validation data would have same settings as train data with some difference
# steps per epoch will not be there for test and we would use validation steps- Also Random seeds & epochs in case of validation steps need not be specified again
# Modelcheckpoint in callbacks will create a hdf5 file with best weights
model.fit(dataug(train['file'],train['label'],batch_size=batch_size,randomized=True,random_seed=1),steps_per_epoch=int(np.ceil(len(train)/batch_size)), epochs=epochs,
          validation_data=dataug(test['file'],test['label'],batch_size=batch_size,randomized=True),validation_steps=int(np.ceil(len(test)/batch_size)),
          callbacks=[ModelCheckpoint(filepath='./weights.hdf5',monitor='val_loss',verbose=0,save_best_only=True)],
          verbose=2)

transfered.trainable=True
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
batch_size=500
epochs=5

model.fit(dataug(train['file'],train['label'],batch_size=batch_size,randomized=True,random_seed=1),steps_per_epoch=int(np.ceil(len(train)/batch_size)), epochs=epochs,
          validation_data=dataug(test['file'],test['label'],batch_size=batch_size,randomized=True),validation_steps=int(np.ceil(len(test)/batch_size)),
          callbacks=[ModelCheckpoint(filepath='./weights.hdf5',monitor='val_loss',verbose=0,save_best_only=True)],
          verbose=2)
model.load_weights('weights.hdf5')

model.save('model_final.h5')
