'''
var1 = 23
var2 = 4

print('testVar1_%d testVar2_%d' %(var1, var2))
'''

import  pandas as pd
from pandas import concat

data = pd.DataFrame({
    "Col1": [10, 20, 15, 30, 45],
     "Col2": [13, 23, 18, 33, 48],
     "Col3": [17, 27, 22, 37, 52]
})

print("Data :\n ",data)
print("Data by index : ",data.loc[0][1])
cols = []

#cols.append(data['Col1'].shift(1))
#cols.append(data['Col2'].shift(1))
#cols.append(data.iloc[:,2]) #getting specific col using index

print("Number of row : ", len(data))
print("Number of colums : ", len(data.loc[0]))
numCol = len(data.loc[0])

for i in range(numCol-1):
    cols.append(data.iloc[:,i].shift(1))

cols.append(data.iloc[:,numCol-1])#get the last colum as the target var


agg = concat(cols, axis=1)

print("Final cols : \n",agg)
print("Type : " ,type(agg))
print("The final data after dropped the first row : \n",agg.drop(agg.index[0], inplace=True)) #inplace is needed as True othewise the var will not be updated with the dropped row
print("Target var : \n", agg.iloc[:,2]) #how to get certain col using index
print("Target var (loc) : \n", agg['Col3']) #how to get certain col using index
