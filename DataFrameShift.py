import pandas as pd

data = pd.DataFrame({
    "Name":["Zairul","Mazwan","Jilani","Maznah","Lela","Zaid","Rania"],
    "Group":["Red","Red","Blue","Yellow","Blue","Green","Red"],
    "Marks":[10,12,11,14,12,15,16],
    "Status":['A','A','A','A','A','A','A']
})

print(data)
print("No of cols : ", data.shape[1])
print("The cols are : ", data.columns)

print("Shift 1 : ", data.shift(1))
print("Shift -1 : ", data.shift(-1))
colMarkShiftPlusOne = data.iloc[:,2].shift(-1) #this is still pd format to get the specific column and do shifting
print("ColMarks+1",colMarkShiftPlusOne)
print("Data before concat : ", data)
joinedData = [data,colMarkShiftPlusOne]
finalData = pd.concat(joinedData, axis=1, join='outer')
print("Data after concat : \n", finalData)
finalData.dropna(inplace=True)
print("Data after dropped NA records : \n", finalData)
