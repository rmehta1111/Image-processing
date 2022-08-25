# We will be using InceptionNet V3 for Transfer Learning
#Initially we will manually label images and work on Data augmentation Normalization

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

