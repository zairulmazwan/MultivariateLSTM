import pandas as pd
import math
import random

vars = [1,2,3,4,5,6]

#get all variables
def getCols(data='../Datasets/UKCovid-Rawdata_1.csv'):
    rawData = pd.read_csv('../Datasets/UKCovid-Rawdata_1.csv')
    cols = list(rawData.columns)
    cols.pop(0)
    #print(cols)
    return cols


def combination (n, r):
    fn = math.factorial(n)
    fr = math.factorial(r)
    fNminusR = math.factorial((n-r))
    comb = fn/(fr*fNminusR)
    return int(comb)

#get independent variables into dictionary
def varDictionary (cols):
    varDict = {} #create an empty dictionary
    #print("Total var : ", cols)
    depVar = cols.pop(0) #take the first element from the list, and remove it from the list
    inVar = cols #the current list after removing the first element
    #print("dependent var : ", depVar)
    #print("Independent vars : ", inVar)

    for i in range(1,(len(inVar)+1)):
        varDict[i]=inVar[i-1] #key = i = 1, value = from the list = index = i-1, i starts from 1 to 6
    return varDict


def runCombination(n, r):
    cols = getCols()
    varDict = varDictionary(cols)
    comb = combination(n,r)
    res = [] #to store the result of combinations
    if (r<n and r!=0):
        temp = []
        listVar  = [x for x in varDict.keys()]

        for i in range(r):
            ranNum = random.randrange(len(listVar))
            temp += [listVar[ranNum]]        #[list(listDict.keys())[ranNum]] #getting the key retrieve by index, then store into a temp list
            listVar.pop(ranNum)
        res+=[temp]
        counter=1
        while (len(res)<comb):
            temp=[]
            listVar  = [x for x in varDict.keys()]
            for i in range(r):
                ranNum = random.randrange(len(listVar))

                temp += [listVar[ranNum]]        #[list(listDict.keys())[ranNum]] #getting the key retrieve by index, then store into a temp list
                listVar.pop(ranNum)
            avail = False
            for i in res:
                if (all(elem in temp  for elem in i)): #to check the temp combination contains the same elements in the existing res combinations
                    avail = True
                    break
            if (avail==False):
                res+=[temp]
            counter+=1
        # print(res)
        # print(len(res))
        # print("Number of tries : ", counter)
    return res




'''
#res = runCombination(6,5)
#print(res)
with open("combRes.csv", 'a') as wfile:

    for i in range(1,6):
        res = runCombination(6,i)
        for i in res:
            for j in range(len(i)):
                #print(i[j], end =",")
                wfile.write(str(i[j]))
                wfile.write(",")
            wfile.write("\n")
            #print()

'''

def getVarCombination (filename, recPos): #input which record to read from the file combRes for variables

    if (recPos>61):
        print("numVar should be less than 63")
        quit()

    data = pd.read_csv(filename)
    cols = list(data.columns) #read the columns from the file
    #print(cols)
    varDict = varDictionary(cols) #create a dictionary for the variables
    #print(varDict)


    with open("VariableCombinations/combRes.csv", 'r') as readfile:

        lines = readfile.readlines() #read all lines in the file

        line = lines[recPos].split(",") #lines is a text, to convert into a list and split by a comma
        #print(line)
        cols2 = ["date","total_admission_UK"] #the default variables to be used
        #for every element in lines - the key of the variables
        for j in range(0,len(line)-1): #to ignore "\n" in the list - line has this value because of new line in the file

            varInd = int(line[j]) #line consiste of the variable's key in the dictionary, need to convert into int
            cols2 += [varDict[varInd+1]] #adding the variables retrieved from line
        #print("cols2  : ",cols2)
        return cols2



# filename = '../Datasets/UKCovid-Rawdata_1.csv'
# cols = getVarCombination (filename,0)
#
# rawData = pd.read_csv('../Datasets/UKCovid-Rawdata_1.csv', header=0, usecols=cols, index_col=0)
# print(rawData.columns)

def getSeasonIndex (rawData): #passing the rawdata variable
    if ("season" in rawData.columns):
        rowCol = list(rawData.columns)
        #print(rowCol.index("season"))
        return rowCol.index("season")
    else:
        return -1
#
#
# getSeasonIndex(rawData)
#
