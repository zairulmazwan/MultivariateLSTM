from keras.models import Sequential
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
import  matplotlib.pyplot as pyplot
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
import math as math
import numpy as numpy


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


cols = ["date","lockdown", "new_cases"]
data = pd.read_csv("CleanedDataset.csv", header=0, usecols=cols, index_col=0)



print(data.head(5))
print("No of records : ", len(data))
values = data.values #this is numpy format
#print(values[:5,:])
encoder = LabelEncoder()
#values[:,1] = encoder.fit_transform(values[:,1])#encode the categorical variable
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
scaledDataset = scaler.fit_transform(values)
print("Before converted into supervised : \n",scaledDataset[:5,:])
reframe = series_to_supervised(scaledDataset,1,1)
print(reframe.head(5))
reframe.drop(columns="var1(t)", axis=1, inplace=True)
print("Drop var1(t) : ",reframe.head(5))
print("Reframed cols : ",reframe.columns)
print("Reframed type : ",type(reframe))

#define model
values = reframe.values
#print("Values data : \n",values[:5,:])
totalRec = len(values)
#print("Total records : ",totalRec)
trainSize = int(totalRec*0.2)
testSize = len(values)-trainSize #not used
train = values[:trainSize,:]
test = values[trainSize:len(values),:]
print("Train length : ", len(train), trainSize) #just checking the number side by side
print("Test length : ", len(test), testSize) #just checking the number side by side

#split dataset into input and output for train and test
trainX, trainY = train[:,:-1], train[:,-1]
print("TrainX : \n", trainX[:5,:])
#print("TrainY : \n", trainY[:5])


testX, testY = test[:,:-1], test[:,-1]
#print("TestX : \n", testX[:5,:])
#print("TestY : \n", testY[:5])

#reshape - 3d shape is commonly used in Tensors for deep learning
print("Shape trainX: ",trainX.shape)
trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1])) #(dimension, row, col) - dimension = how many object. In this example, there will be 326 object which each object has 1 row and 2 cols
testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
print("reshape trainX: ",trainX.shape)
print("Shape trainY: ",trainY.shape)
print("reshape testX: ",testX.shape)
print("Shape testY: ",testY.shape)
print("trainX.shape[2]: ",trainX.shape[2])

print("Reshape items : ",trainX[:5,:])



# design network
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(trainX, trainY, epochs=50, batch_size=10, validation_data=(testX, testY), verbose=2, shuffle=False)
#model.fit(trainX, trainY, epochs=100, batch_size=2,verbose=2)



# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
#yhatTrain = model.predict(trainX)
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
print("testX[:, 1:] : ",testX[:, 1:])

# invert scaling for forecast
inv_yhat = concatenate((yhat, testX[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

'''

# invert scaling for actual
testY = testY.reshape((len(testY), 1))
inv_y = concatenate((testY, testX[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# plot baseline and predictions
#pyplot.plot(scaler.inverse_transform(scaledDataset), label='dataset')
#pyplot.plot(yhatTrain)
pyplot.plot(inv_yhat, label='prediction')
pyplot.show()


'''
