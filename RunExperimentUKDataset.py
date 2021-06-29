from cmath import sqrt

import pandas as pd
import numpy as np
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, concatenate
import numpy
import VariableCombinations.VarCombinations as vc



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
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

def experiment(data, iter, i, cols, rawData):

	# filename = 'Datasets/UKCovid-Rawdata_1.csv'
	# cols = vc.getVarCombination(filename, 29)
	# rawData = pd.read_csv(filename, header=0, usecols=cols, index_col=0)


	rawData.dropna(inplace=True)
	#print(rawData.iloc[:5,2])
	#print("Season ",rawData['season'].head())
	# print("Len ",len(rawData))
	print(rawData.columns)
	print("Raw data shape: ",rawData.shape)
	#print("Season col unique values",rawData.iloc[:,2].unique())

	values = rawData.values
	print(values[:5,:])

	encoder = LabelEncoder()
	seasonInd = vc.getSeasonIndex(rawData)
	if (seasonInd != -1):
		#print("season index : ", seasonInd)
		values[:,seasonInd] = encoder.fit_transform(values[:,seasonInd])#encode the categorical variable - season
		#values[:,4] = encoder.fit_transform(values[:,4])#encode the categorical variable - SE for pollution data
		values = values.astype('float32')

		print("Season col",values[:2,seasonInd])
		print("Season col unique values numpy : ",np.unique(values[:,seasonInd])) #to get distinct values of the col


	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)


	# frame as supervised learning
	reframed = series_to_supervised(scaled, 1, 1)
	print("Cols ", reframed.columns)

	print("cols len ", len(cols)-1)
	for k in range(1,len(cols)-1):
		reframed.drop(reframed.columns[[-1]], axis=1, inplace=True)
	print("Cols after drop", reframed.columns)


	#print("varT5 ",reframed.iloc[:5,5])
	#print("Length refarmed ", reframed.shape[1]) #shape[1] - for columns, shape[0] - for row
	#reframed.to_csv("Datasets/tsDataset.csv")


	#define model
	values = reframed.values
	trainSize = int(len(values)*0.75)
	#trainSize = 365*24
	testSize = len(values)-trainSize
	# print("Train : ",trainSize)
	# print("Test : ",testSize)
	train = values[:trainSize,:]
	test = values[trainSize:,:]
	print("Total rec : ",values.shape)
	print("Train : ",len(train))
	print("Test : ",len(test))



	#split dataset into input and output for train and test
	trainX, trainY = train[:,:-1], train[:,-1] #getting the input and the target var for train
	print("Shape TrainX ",trainX.shape)
	print("Shape TrainY ",trainY.shape)
	testX, testY = test[:,:-1], test[:,-1] #getting the input and the target var for test
	print("Shape testX ",testX.shape)
	print("Shape testY ",testY.shape)



	# reshape input to be 3D [samples, timesteps, features]
	trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
	testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
	print(trainX.shape, trainY.shape, testX.shape, testY.shape)

	# design network
	model = Sequential()
	model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

	# fit network
	history = model.fit(trainX, trainY, epochs=100, batch_size=35, validation_data=(testX, testY), verbose=0, shuffle=False)
	# plot history
	# pyplot.plot(history.history['loss'], label='train')
	# pyplot.plot(history.history['val_loss'], label='test')
	# pyplot.legend()
	# pyplot.show()


	# make a prediction
	yhat = model.predict(testX)
	testX = testX.reshape((testX.shape[0], testX.shape[2]))
	#print(testX.shape)
	yhatTrain = model.predict(trainX)
	trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))


	# invert scaling for forecast
	inv_yhat = concatenate((yhat, testX[:, 1:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	print(inv_yhat.shape)
	inv_yhat = inv_yhat[:,0]

	inv_yhatTrain = concatenate((yhatTrain, trainX[:, 1:]), axis=1)
	inv_yhatTrain = scaler.inverse_transform(inv_yhatTrain)
	print(inv_yhatTrain.shape)
	inv_yhatTrain = inv_yhatTrain[:,0]
	print("dataset shape 150 : ", inv_yhatTrain.shape)


	# invert scaling for actual
	testY = testY.reshape((len(testY), 1))
	inv_y = concatenate((testY, testX[:, 1:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]

	trainY = trainY.reshape((len(trainY), 1))
	inv_yTrain = concatenate((trainY, trainX[:, 1:]), axis=1)
	inv_yTrain = scaler.inverse_transform(inv_yTrain)
	inv_yTrain = inv_yTrain[:,0]

	#plot graph
	yplot = concatenate((inv_yTrain, inv_y), axis=0)
	#print("Record for y plot : ", yplot.shape[0])
	pyplot.plot(yplot) #the actual y plot

	#plotting the train y hat
	yHatTrain = numpy.zeros((len(yplot),1))
	yHatTrain[:, :] = numpy.nan
	inv_yhatTrain = inv_yhatTrain.reshape((len(inv_yhatTrain),1))
	print("Shape inv_yhatTrain : ", inv_yhatTrain.shape)
	yHatTrain[1:len(inv_yhatTrain)+1,:] = inv_yhatTrain
	pyplot.plot(yHatTrain)

	#plotting the test y hat
	yHatTest = numpy.zeros((len(yplot),1)) #prepare an array with zero values, same shape of yplot
	yHatTest[:, :] = numpy.nan #nan is required to avoid plotting zero in the graph
	inv_yhat = inv_yhat.reshape((len(inv_yhat),1)) #currently inv_yhat shape is (n,) - a list like
	print("Shape inv_yhat : ", inv_yhat.shape)
	print("Shape yHatTest : ", yHatTest.shape)
	yHatTest [len(inv_yhatTrain):len(yHatTest),:] = inv_yhat #getting the inv_yhat data into the array to a specific data point
	pyplot.plot(yHatTest)

	#pyplot.show()
	pyplot.title("Admission Prediction - Iter "+str(iter), y=1.0, loc='center')

	pyplot.savefig("Results/res_"+str(i)+"_"+str(iter)+".png")
	pyplot.clf()

	# calculate RMSE
	rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
	#print('Test RMSE: %.3f' % rmse)
	print('Test RMSE:',rmse)
	# score = model.evaluate(trainX, trainY, batch_size=25, verbose=0)
	# print(' Train accuracy:', score[1])
	#model.summary()
	data.append(rmse)

