# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:49:25 2020

@author: loren
"""

# Import Libraries
from pandas_datareader import data
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
import tensorflow as tf

# Get Current Date
today = date.today()
currentDate = today.strftime("%Y/%m/%d")

## Set Info
start_date = '2015-06-10'
end_date = currentDate
stockName = 'AMZN'

# Get the data
stockData = data.get_data_yahoo(stockName, start_date, end_date)

# Overview everything
stockData.head()

# Make pretty plot
stockData["Close"].plot(label=stockName, color='black', marker='.', title="Closing Price")
plt.legend()
plt.grid()
currentFig = plt.gcf()
currentFig.set_facecolor('white')
plt.show()

# Add the dates as columns
stockData['Year'] = stockData.index.year
stockData['Month'] = stockData.index.month
stockData['Day'] = stockData.index.day #print(stockData)

# Shift dataframe by amount to predict 
days = 1
stepsToShift = days * 2

# Create separate dataframe to predict
columns = ['High','Low','Open','Close','Volume']
stockDataToPredict = stockData[columns].shift(-stepsToShift)

# Set up data
dataIn = stockData.values[0:-stepsToShift] #print(x_data.shape)
dataOut = stockDataToPredict.values[:-stepsToShift]

# allocate space for training and test data
numData = len(dataIn)
trainPercentage = 0.7
numTrain = int(trainPercentage * numData)
numTest = numData-numTrain

# split into training and test data
xTrain = stockData[0:numTrain]
yTrain = stockDataToPredict[0:numTrain]

xTest = stockData[numTrain:]
yTest = stockDataToPredict[numTrain:]

print(yTest)

# Check the max/min values
print("Min:", np.min(xTrain))
print("Max:", np.max(yTrain))

# Normalize values between 0-1
scalerX = MinMaxScaler()
xTrainNorm = scalerX.fit_transform(xTrain)
xTestNorm = scalerX.transform(xTest)

scalerY = MinMaxScaler()
yTrainNorm = scalerY.fit_transform(yTrain)
yTestNorm = scalerY.transform(yTest)

# Recurrent Neural Network GRU
model = Sequential()

numIn = stockData.shape[1]
numOut = stockDataToPredict.shape[1]

# add gated recurrent unit
model.add(GRU(units=512, return_sequences=True, input_shape=(None,numIn,)))
model.add(Dense(numOut, activation='sigmoid'))

# set learning rate and compile model
optimizerChoice = RMSprop(lr=1e-3) # try RMS

model.compile(loss='mean_squared_error', optimizer=optimizerChoice)
model.summary()

# Callbacks for early stopping and reducing learning rate and checkpoint
path_checkpoint = '23_checkpoint.keras'

# checkpoints
callbackCheckpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

# early stopping
callbackEarlyStopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

# reducing learning rate
callbackLearningRate = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
# pass when training
callbacks = [callbackCheckpoint, callbackEarlyStopping, callbackLearningRate]


def batch_generator(batchSize, sequenceLength):

    while True:
        # Create array for the batch of input and outputs.
        xShape = (batchSize, sequenceLength, numIn)
        xBatch = np.zeros(shape=xShape, dtype=np.float16)
        yShape = (batchSize, sequenceLength, numOut)
        yBatch = np.zeros(shape=yShape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batchSize):
            # Get a random start-index.
            idx = np.random.randint(numTrain - sequenceLength)
            
            # Copy the sequences of data starting at this index.
            xBatch[i] = xTrainNorm[idx : idx+sequenceLength]
            yBatch[i] = yTrainNorm[idx : idx+sequenceLength]
        
        yield (xBatch, yBatch)
        
# Training the model
xBatches = batch_generator(batchSize=256,sequenceLength=100)

validation_data = (np.expand_dims(xTestNorm, axis=0),
                   np.expand_dims(yTestNorm, axis=0))

model.fit(x=xBatches, epochs=20, steps_per_epoch=100, validation_data=validation_data, callbacks=callbacks)

# Use the model to predict on Test Data.
inputToPredict = np.expand_dims(xTestNorm, axis=0)
predictedNormVal = model.predict(inputToPredict)

# scale back from normalization
predictedValuesY = scalerY.inverse_transform(predictedNormVal[0])

#plt.plot(predictedValues)
#yTest.plot()

# separate result for plotting
highPred = predictedValuesY[:,0]
lowPred = predictedValuesY[:,1]
openPred = predictedValuesY[:,2]
closePred = predictedValuesY[:,3]
volumPred = predictedValuesY[:,4]

#%% Plot Predicted 
plt.plot(closePred, color='black', marker='.')
plt.plot(highPred)
plt.plot(lowPred)

plt.title('Predicted Closing Price in £')
plt.grid()
currentFig = plt.gcf()
currentFig.set_facecolor('white')

#%% Plot Real
xTest['Close'].plot(label=stockName, color='black', marker='.', title="Real Closing Price in £")
xTest['High'].plot()
xTest['Low'].plot()

plt.grid()
currentFig = plt.gcf()
currentFig.set_facecolor('white')
plt.show()