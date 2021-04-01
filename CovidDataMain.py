from pandas import read_csv
from datetime import datetime

#this program is to prepare Malaysia dataset only

#reading a raw dataset from the file
dataset = read_csv('owid-covid-data.csv', index_col=0)
#dataset = read_csv('raw.csv', index_col=0)
print("Head : ",dataset.head(5))
print("Tail : ",dataset.tail(5))

datasetMsia = dataset.loc[dataset.location == 'Malaysia']#getting specific record based on the value, location is the column name, 'Malaysia' is the value available in the column
print("Malaysia raw dataset : ",datasetMsia.head(10))
#print(datasetMsia['location'].head(10))
#print(len(datasetMsia['location']))
headers = list(datasetMsia.columns)#getting all headers from the dataset
#print("Malaysia raw dataset headers : ",headers)

#getting specific colums from the dataset
col = ['location','date','total_cases','new_cases','total_deaths','new_deaths']
datasetMsia = datasetMsia[col]

print(datasetMsia.head(10))
datasetMsia.to_csv('MalaysiaDataset.csv')
print("Row : ", len(datasetMsia))


#datasetMsiaDropISOCode = datasetMsia.drop(["iso_code"], axis=1, inplace=False)
#datasetMsia.to_csv('DropISOMalaysia.csv')


