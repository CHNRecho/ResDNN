# -*- coding: utf-8 -*-
"""

@author: DELL
"""


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Dropout, Input, BatchNormalization, Dense, Add, ReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
import os
import hdf5storage
import matplotlib.pyplot as plt
from SIDNet import SIDNet
from load_data import load_data
from setting import scheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# loda data
Inputdata_train,outputdata_train,Inputdata_valid,outputdata_valid=load_data()

# Model initialization
model=SIDNet()

# Model Save Path
filepath = r'Z:\SIDNet_best.h5'

# Learning rate decay
reduce_lr = LearningRateScheduler(scheduler)

# Creating the ModelCheckpoint callback function
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True)  

# Model Training
history = model.fit(Inputdata_train, outputdata_train, epochs=100, batch_size=256,
                    validation_data=(Inputdata_valid, outputdata_valid),shuffle=True,
                    callbacks=[checkpoint, reduce_lr, EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0001)])

# Plotting the loss curve
loss = history.history['loss']
val_loss= history.history['val_loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'r',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

# Save the loss curve
hdf5storage.savemat(r"Z:\SIDNet_Loss.mat", {'train_loss':loss, 'valid_loss':val_loss}, format='7.3')
