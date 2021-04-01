from pandas import read_csv
import pandas as pd
from matplotlib import pyplot

dataset = read_csv("CleanedDataset.csv", index_col=0)
dataset['date'] = pd.to_datetime(dataset['date']) #this is to convert the string column into datetime format

#dateformat = pd.to_datetime(dataset['date']) #create a var that a col is a datetime format
#dataset2 = dataset.set_index(dateformat) #set the index of the dataset as the date with the datetime format
#dataset2.drop('date', axis=1, inplace=True) #the date column is still there meed to drop otherwise duplicated with the index
#dataset = dataset.values #when this is used, the var become numpy format
#dataset = dataset.set_index('date')
#print("Dataset is  : ",dataset2)



#colToPlot = [0,1,2]
#pyplot.figure()
#pyplot.plot(dataset[:, colToPlot[2]]) #specify which col to plot
#pyplot.title(dataset.columns[colToPlot[2]], y=0.5, loc='right')

#dataset2.plot(y='new_cases') #use this code if the x is the index
dataset.plot(x='date', y='new_cases')
pyplot.ylabel("No. of Cases")
pyplot.xlabel("Days")
pyplot.title("Daily New Cases", y=1.0, loc='center')

pyplot.show()

