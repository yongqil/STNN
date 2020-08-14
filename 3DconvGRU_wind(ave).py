""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.layers import Input, Dense, Dropout, Lambda, InputLayer, concatenate, LSTM, TimeDistributed, Embedding, Flatten, Reshape, Deconvolution2D, LeakyReLU
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv3D, Conv2D, Conv3DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D, ConvGRU2D, DeConvGRU2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
import keras as k
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.regularizers import l2
from scipy import stats
from scipy import io
import sys

# from WindForecasting import ConvRNN

activef = LeakyReLU(alpha=0.2)
reg = 0.0000
p = 0.3
inputs = Input(shape=(3, 26, 12, 5))
inter, stateh1 = ConvGRU2D(filters=8, kernel_size=(3, 3), strides=(1, 1), return_state=True, activation=activef, dilation_rate=(1, 1),
                  padding='same', dropout=p, recurrent_dropout=p, kernel_regularizer=l2(reg), recurrent_regularizer=l2(reg), bias_regularizer=l2(reg), return_sequences=True)(inputs, training=True)


inter = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1,2,2), activation=activef,
               padding='same', kernel_regularizer=l2(reg), bias_regularizer=l2(reg), data_format='channels_last')(inter)
inter, stateh2 = ConvGRU2D(filters=16, kernel_size=(3, 3), strides=(1, 1), return_state=True, activation=activef,  dilation_rate=(1, 1),
                  padding='same', dropout=p, recurrent_dropout=p, kernel_regularizer=l2(reg), recurrent_regularizer=l2(reg), bias_regularizer=l2(reg), return_sequences=True)(inter, training=True)


inter = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1,1,2), activation=activef,
               padding='same', kernel_regularizer=l2(reg), bias_regularizer=l2(reg), data_format='channels_last')(inter)
inter, stateh3 = ConvGRU2D(filters=32, kernel_size=(3, 3), strides=(1, 1), return_state=True, activation=activef,  dilation_rate=(1, 1),
                  padding='same', dropout=p, recurrent_dropout=p, kernel_regularizer=l2(reg), recurrent_regularizer=l2(reg), bias_regularizer=l2(reg), return_sequences=True)(inter, training=True)

inter = ConvGRU2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation=activef,  dilation_rate=(1, 1),
                  padding='same', dropout=p, recurrent_dropout=p, kernel_regularizer=l2(reg), recurrent_regularizer=l2(reg), bias_regularizer=l2(reg), return_sequences=True)(inter, initial_state=stateh3, training=True)
inter = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), strides=(1,1,2), activation=activef,
                padding='same', kernel_regularizer=l2(reg), bias_regularizer=l2(reg), data_format='channels_last')(inter)


inter = ConvGRU2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation=activef,  dilation_rate=(1, 1),
                  padding='same', dropout=p, recurrent_dropout=p, kernel_regularizer=l2(reg), recurrent_regularizer=l2(reg), bias_regularizer=l2(reg), return_sequences=True)(inter,initial_state=stateh2,training=True)
inter = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), strides=(1,2,2), activation=activef,
               padding='same', kernel_regularizer=l2(reg), bias_regularizer=l2(reg), data_format='channels_last')(inter)

inter = ConvGRU2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation=activef,dilation_rate=(1, 1),
                  padding='same', dropout=p, recurrent_dropout=p, kernel_regularizer=l2(reg), recurrent_regularizer=l2(reg), bias_regularizer=l2(reg), return_sequences=True)(inter,initial_state=stateh1, training=True)


outputs = Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1,1,1),
               padding='same', kernel_regularizer=l2(reg), bias_regularizer=l2(reg), data_format='channels_last')(inter)



seq = Model(inputs, outputs)

seq.summary()

seq.compile(loss='mean_squared_error', optimizer='Adam')


data1 = np.load("2011_m_wind_data.npy")
data2 = np.load("2012_m_wind_data.npy")
#
dataall = np.vstack((data1, data2))
# dataall =  np.load("2011_m_wind_data.npy")
T = dataall.__len__()
t = range(T)
index = np.where(np.mod(t,12)==0)
dataall2 = np.zeros((8759+8760, 26, 12, 6))
for i in range(8759+8760):
    print(index[0][i])
    print(index[0][i+1])
    dai = np.mean(dataall[index[0][i]:index[0][i + 1]], axis=0, keepdims=True)
    dataall2[i] = dai


data = np.zeros((8759+8760, 26, 12, 6))
data[:,:,:,0] = dataall2[:,:,:,0]
data[:,:,:,1] = dataall2[:,:,:,0]*np.sin(dataall2[:,:,:,1]/180*np.pi)
data[:,:,:,2] = dataall2[:,:,:,0]*np.cos(dataall2[:,:,:,1]/180*np.pi)
data[:,:,:,3] = dataall2[:,:,:,3]
data[:,:,:,4] = dataall2[:,:,:,4]
meanData = np.mean(data, axis=0, keepdims=True)
stdData = np.std(data, axis=0, keepdims=True)
# data_normalized = (data - meanData) / stdData
maxData = np.max(data, axis=0, keepdims=True)
maxData = np.max(maxData, axis=1, keepdims=True)
maxData = np.max(maxData, axis=2, keepdims=True)
minData = np.min(data, axis=0, keepdims=True)
minData = np.min(minData, axis=1, keepdims=True)
minData = np.min(minData, axis=2, keepdims=True)
data_normalized = (data-minData)/ (maxData - minData)

# Train the network
ahead=3

noisy_movies = np.zeros([17513, ahead, 26, 12, 5])
shifted_movies = np.zeros([17513, ahead, 26, 12, 1])
for i in range(ahead):
    noisy_movies[:, i, :, :, :] = data_normalized[i:-6+i, :, :, [0, 1, 2, 3, 4]]
    shifted_movies[:, i, :, :, :] = data_normalized[3+i:-3+i, :, :, [0]]


hist = seq.fit(noisy_movies[0:8753], shifted_movies[0:8753], batch_size=32,
        validation_data=(noisy_movies[8753:8753+8753], shifted_movies[8753:8753+8753]),
        epochs=300)
seq.save_weights("3DCNNGRU_3h_3h_wind_encoder_decoder_0.3(ave300).h5")

print(hist.history)
plt.plot(hist.history.get("val_loss"))
plt.plot(hist.history.get("loss"))
plt.show()