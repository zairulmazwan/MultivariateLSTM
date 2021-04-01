from pandas import read_csv
import pandas as pd

data1 = pd.DataFrame({
    "Name":["Zairul","Mazwan","Jilani"],
    "Group":["Red","Red","Blue"]
})

data2 = pd.DataFrame({
    "Course":["CS","CS","SE"],
    "Level":["4","5","4"]
})


print("Data 1 : ",data1)
print("Data 2 : ",data2)

joinedData = [data1,data2]
print("Joined data : ",joinedData)

'''
fullData = pd.concat(joinedData, axis=1, join='outer')
print(fullData)
print("CS Course only:")
print(fullData.loc[fullData.Course == 'CS'])
#print(fullData.loc[1])
'''
