import  RunExperimentUKDataset as exp
import pandas as pd

iter = 5
data = []

for i in range(iter):
    print(iter)
    exp.experiment(data, i)
print(data)
data = pd.DataFrame(data)
data.to_csv("Results/Experiment_Results")
