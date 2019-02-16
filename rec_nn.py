# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 00:13:55 2018

@author: Legolas10,krishtna999
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING DATA
train=pd.read_csv('Google_Stock_Price_Train.csv')
train_set=train.iloc[:,1:2].values
#HERE 1:2 CREATES A ARRAY AND JUST 1 GIVES A SINGLE VECTOR..AND NN ACCEPTS ONLY ARRAYS

y_hat=train['Open'].rolling(60).mean().iloc[-1]
#FEATURE SCALING
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
train_set_scaled=sc.fit_transform(train_set)

#CREATING THE STRUCTURE WITH AN OPTIMAL VALUE OF 60 TIMESTEPS AND FOR EACH TIMESTEP 1 OUTPUT..
x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])
    
x_train=np.array(x_train)
y_train=np.array(y_train)

#RESHAPING TO ADD/CREATE A NEW DIMENSION..
#COZ RNN ACEPTS 3-D DATA,I.E BATCHSIZE,NO.OF TIMESTEPS AND NO OF INDICATORS/PREDICTORS
#HERE THE NO OF INDICATORS IS THE SCALED AND TIMESTEPPED GOOGLE STOCK PRICE  MIGHT VARY...CHECK IT OUT
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#BUILDING THE RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#HERE DROP OUT HELPS IN REDUCING OVERFITTING OF DATA BY IGNORING SOME RANDOMLY CATEGORISED NEURAL NETS

regressor=Sequential()

#ADDING THE LSTM LAYERS AND SOME DROPOUT REGULARIZATION....
#HERE THE UNITS PARAMTERTER IS THE NO OF NEURONS OR LSTM CELLS IN ONE LAYER
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))#THIS MEANS 20 % OF THE 50 NEURONS /CELLS WILL BE IGNORED .

#LAYER 2
regressor.add(LSTM(units=50,return_sequences=True))#....the units argument tells there are 50 neurons in previous layer
regressor.add(Dropout(0.2))

#lAYER 3
regressor.add(LSTM(units=50,return_sequences=True))#....the units argument tells there are 50 neurons in previous layer
regressor.add(Dropout(0.2))

#layer 4
regressor.add(LSTM(units=50))#....the units argument tells there are 50 neurons in previous layer
#HERE WE REMOVED THE RETURN_SEQUENCES COZ NO MORE LAYERS ARE GONNA BE THERE..NEXT IS THE OUTPUT LAYER.
regressor.add(Dropout(0.2))

#output layer.
regressor.add(Dense(units=1))

#COMPILING THE RNN WITH A RIGHT OPTIMIZER AND RIGHT LOSS FUNCTION....(MSE).
#optimzer can be anything, but ...use adam or RMSProp (recommended)..try both.
regressor.compile(optimizer='adam',loss='mean_squared_error')

#FITTING THE REGRESSOR..
regressor.fit(x_train,y_train,batch_size=30,epochs=100)

#GETTING X_TEST
test_data=pd.read_csv('Google_Stock_Price_Test.csv')
real_test_values=test_data.iloc[:,1:2].values
 
 #PREDICTING ...
dataset_total=pd.concat([train['Open'],test_data['Open']],axis=0)#axis to concat along: {0/’index’, 1/’columns’}
inputs=dataset_total[len(dataset_total)-len(test_data)-60:].values
#SCALING IT 
inputs=sc.fit_transform(inputs)
inputs=inputs.reshape(-1,1)

#NOW CREATING THE X_TEST SET BY TIMESTEPS....
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
    
x_test=np.array(x_test)
#NOW RESHAPING TO 3-D SO THAT IT IS COMPATIBLE WITH THE RNN INPUT VALUES....
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#PREDICTING...
predict_stock_prices=regressor.predict(x_test)

#NOW VERY IMP...REVERSE THE SCALED VALUE TO ACTUAL VALUE.
predict_stock_prices=sc.inverse_transform(predict_stock_prices)

#VISUALIZING THE RESULTS..
plt.plot(real_test_values,color='red',label='real stock price')
plt.plot(predict_stock_prices,color='blue',label='predicted stock price')
plt.xlabel('time')
plt.ylabel('Stock price values')
plt.plot()

