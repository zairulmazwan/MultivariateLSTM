import pandas as pd
from pandas import concat
from sklearn.preprocessing import LabelEncoder


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


rawData = pd.read_csv('UKCovid.csv')
#print("Before date format : ",rawData.head())
rawData['date'] = pd.to_datetime(rawData['date'], format='%d/%m/%Y')
rawData=rawData.set_index('date') #setting date as the index
print(rawData.head())
rawData.dropna(inplace=True)
rawData.to_csv("Datasets/RawDataWithoutNA.csv")


'''
#print(rawData.columns)
TSDataset = series_to_supervised(rawData,1,1)
print(TSDataset.columns)
print(TSDataset.head())
TSDataset.to_csv('Datasets/TSDataset.csv')
CleanedTSDataset = TSDataset.drop(TSDataset.columns[[7,8,9,10,11,12]], axis=1)
print(CleanedTSDataset.columns)
print(CleanedTSDataset.head())

#encode categorical var into numerical - season
# values=CleanedTSDataset.values
# encoder = LabelEncoder()
# values[:,2] = encoder.fit_transform(values[:,2])

CleanedTSDataset.dropna(inplace=True)
# ensure all data is float
#values = values.astype('float32')

CleanedTSDataset.to_csv('Datasets/CleanedTSDatasetWithoutNA.csv')


#CleanedTSDataset.to_csv('Datasets/CleanedTSDataset.csv')
#print(len(CleanedTSDataset))
'''
