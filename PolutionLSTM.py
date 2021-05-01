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
from datetime import datetime


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

def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')


def prepareData ():
	dataset = pd.read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
	#dataset.drop('No', axis=1, inplace=True)
	#print(dataset.head(5))
	return dataset


#print(prepareData ())

dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
print(dataset.iloc[:5,4])
print("Dataset : ", dataset.head(5))
values = dataset.values
print("Values len ", len(values))
print("col-4 ", values[:5,4])

print("Data col : ", len(values[0,:]))
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
# pyplot.figure()
# for group in groups:
# 	pyplot.subplot(len(groups), 1, i)
# 	pyplot.plot(values[:, group])
# 	pyplot.title(dataset.columns[group], y=0.5, loc='right')
# 	i += 1
# pyplot.show()


# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])

print("acv ",values[-5:-1,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print("Reframe cols : ", reframed.columns)
print(reframed.iloc[:5,4])



'''


#define model
values = reframed.values
#print("Values data : \n",values[:5,:])
totalRec = len(values)
#print("Total records : ",totalRec)
# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# trainSize = int(totalRec*0.8)
# testSize = len(values)-trainSize #not used
# train = values[:trainSize,:]
# test = values[trainSize:len(values),:]
print("Train length : ", len(train)) #just checking the number side by side
print("Test length : ", len(test)) #just checking the number side by side

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
history = model.fit(trainX, trainY, epochs=50, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
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
#print("testX[:, 1:] : ",testX[:5, 1:])
#print("testX shape : ", testX.shape)



# invert scaling for forecast
inv_yhat = concatenate((yhat, testX[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
print("inv_yhat : \n", inv_yhat[:5,:])
inv_yhat = inv_yhat[:,0]



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
