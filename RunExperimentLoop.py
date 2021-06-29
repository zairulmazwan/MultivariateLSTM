import  RunExperimentUKDataset as exp
import pandas as pd
import VariableCombinations.VarCombinations as vc

data = []



for i in range(1,2): #experiment for which var
    filename = 'Datasets/UKCovid-Rawdata_1.csv'
    cols = vc.getVarCombination(filename, i)
    rawData = pd.read_csv(filename, header=0, usecols=cols, index_col=0)
    for j in range(1,26):
        exp.experiment(data, j, i, cols, rawData)
    print(data)
    data = pd.DataFrame(data)
    data.to_csv("Results/Experiment_Results_"+str(i)+".csv")
