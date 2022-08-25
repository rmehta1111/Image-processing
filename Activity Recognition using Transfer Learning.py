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
