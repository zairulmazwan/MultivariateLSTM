import  RunExperimentUKDataset as exp
import pandas as pd
import VariableCombinations.VarCombinations as vc
import os
import statistics



def writeRes (fileName, res, bigRes):
    fileWriter = open(fileName, 'w')
    realRes = [] #realNumberResult
    expRes = [] #experiment result
    for i in res:
        #print(i)
        realNum = i.real
        realRes.append(realNum);
        fileWriter.write(str(realNum))
        fileWriter.write("\n")
    fileWriter.write("Mean")
    fileWriter.write(",")
    fileWriter.write(str(sum(realRes)/len(realRes)))
    expRes.append(sum(realRes)/len(realRes))
    fileWriter.write("\n")
    fileWriter.write("Max")
    fileWriter.write(",")
    fileWriter.write(str(max(realRes)))
    expRes.append(max(realRes))
    fileWriter.write("\n")
    fileWriter.write("Min")
    fileWriter.write(",")
    fileWriter.write(str(min(realRes)))
    expRes.append(min(realRes))
    fileWriter.write("\n")
    fileWriter.write("Stdev")
    fileWriter.write(",")
    fileWriter.write(str(statistics.stdev(realRes)))
    expRes.append(statistics.stdev(realRes))
    bigRes.append(expRes)


bigRes  = [] #to get all results from each experiment (by variables)
for i in range(0,62): #experiment for which var
    #print("Var : ",i)
    filename = 'Datasets/UKCovid-Rawdata_1.csv'
    cols = vc.getVarCombination(filename, i)

    parentDir = "Results/"
    resDir = "Var_"+str(i)
    path = os.path.join(parentDir,resDir)
    os.mkdir(path) #creating the new dir
    #writing variables into a txt file
    fileCols = path+"/Variables_"+str(i)+".txt"
    fileWrite = open(fileCols, 'w')
    for var in cols:
        fileWrite.write(var)
        fileWrite.write(",")


    rawData = pd.read_csv(filename, header=0, usecols=cols, index_col=0)
    data = [] #this variable is created empty for every new experiment
    for j in range(1,26):
        print("Var : ",i)
        print("Iter : ",j)
        exp.experiment(data, j, i, cols, rawData, path)
    #print(data)
    #data = pd.DataFrame(data)
    #data.to_csv("Results/Experiment_Results_"+str(i)+".csv")

    filename=path+"/Experiment_Results_"+str(i)+".csv"
    writeRes(filename, data, bigRes)

fileNameBigRes = "Results/All_Results.csv"
bigRes = pd.DataFrame(bigRes, columns=["Mean","Max","Min","Stdev"])
bigRes.to_csv(fileNameBigRes)






