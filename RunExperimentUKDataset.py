import pandas as pd
from pandas import concat
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense


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


cols = ["date","total_cases","new_cases","season","national_lockdown","first_vaccine","second_vaccine","total_admission"]
rawData = pd.read_csv('Datasets/UKCovid-Rawdata.csv', header=0, usecols=cols, index_col=0)

#cols = ["date","lockdown", "new_cases"]
#rawData = pd.read_csv("CleanedDataset.csv", header=0, usecols=cols, index_col=0)

rawData.dropna(inplace=True)
#print(rawData.iloc[:5,2])
print("Season ",rawData['season'].head())
# print("Len ",len(rawData))
print(rawData.columns)

values = rawData.values

encoder = LabelEncoder()
values[:,2] = encoder.fit_transform(values[:,2])#encode the categorical variable
values = values.astype('float32')

print("Season col",values[:5,2])

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#print("Cols ", reframed.columns)
reframed.drop(reframed.columns[[7,8,9,10,11,12]], axis=1, inplace=True)
#print("Cols after drop", reframed.columns)
#print("Length refarmed ", reframed.shape[1]) #shape[1] - for columns, shape[0] - for row

#define model
values = reframed.values
trainSize = int(len(values)*0.67)
testSize = len(values)-trainSize
# print("Train : ",trainSize)
# print("Test : ",testSize)
train = values[:trainSize,:]
test = values[trainSize:,:]
# print("Train : ",len(train))
# print("Test : ",len(test))

#split dataset into input and output for train and test
trainX, trainY = train[:,:-1], train[:,-1]
print("Shape TrainX ",trainX.shape)
print("Shape TrainY ",trainY.shape)
testX, testY = test[:,:-1], test[:,-1]
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
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(trainX, trainY, epochs=100, batch_size=27, validation_data=(testX, testY), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
