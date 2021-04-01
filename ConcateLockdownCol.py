from pandas import read_csv
import pandas as pd

datasetMsia = read_csv("MalaysiaDataset.csv")
#print(datasetMsia.head(5))
print(datasetMsia.columns)
cols = ['date','new_cases'] #define specific cols into an array
datasetNewCases = datasetMsia[cols] #getting specific cols from the the dataset
print(datasetNewCases.head(5))


#datasetNewCases.to_csv('datasetNewCases.csv', index=False) #if do not want the index in the csv file
datasetNewCases.to_csv('datasetNewCases.csv', index_label='No')


#reading malaysia lockdown file
colsList = ['lockdown'] #just to read this col from the csv file
datasetMsiaLockdown = read_csv("MsiaLockdown.csv", usecols=colsList) #use usecols from the read_csv function
#print(datasetMsiaLockdown)

combineDataset=pd.concat([datasetMsiaLockdown, datasetNewCases], axis=1) #joining 2 dataframe append by column : axis = 1
print("Combined dataset : ",combineDataset)

#the current columns order is date, new cases, lockdown
rearrangeCols = ['date','lockdown','new_cases'] #re-arrange the position of the columns in the dataset
combineDataset = combineDataset[rearrangeCols]

print("Re-arrange cols : ",combineDataset)
combineDataset.to_csv("CleanedDataset.csv", index_label='No')
