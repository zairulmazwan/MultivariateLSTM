import pandas as pd
from pandas import concat


n_in = 1
n_out = 1
cols, names = list(), list()

data = pd.DataFrame({
    "Col1": [10, 20, 15, 30, 45],
     "Col2": [13, 23, 18, 33, 48],
     "Col3": [17, 27, 22, 37, 52]
})

listEg = [[10, 20, 15, 30, 45],[13, 23, 18, 33, 48],[17, 27, 22, 37, 52]]
print("Row 12 : ",len(listEg) if type(listEg) is list else 99)

#print("Data : ",data)
#print("Data type : ",type(data))
#print("Data shape : ", data.shape[0]) #get the number of colums (1) using shape - for dataframe only. For row (0)

n_vars = 1 if type(data) is list else data.shape[1]
#print("n_vars : ",n_vars)
#print("n_vars type : ",type(n_vars))


for i in range(n_in, 0, -1):
        print("i : ",i)
        print("Data shift : ",data.shift(i))
        cols.append(data.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

print("Cols 1: ",cols)
print("Names 1: ",names)

for i in range(0, n_out):
    cols.append((data.shift(-1)))
    if i==0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        print("This found")
    else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]



print("Cols 2: ",cols)
print("Names 2: ",names)


agg = concat(cols, axis=1)
agg.columns = names

print("Agg : ", agg)


