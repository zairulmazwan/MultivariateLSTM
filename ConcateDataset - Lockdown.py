from pandas import read_csv
import pandas as pd
from datetime import datetime


lockdownData = read_csv("MsiaLockdown.csv", index_col=0)
print(len(lockdownData))


def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')


dataset = read_csv('MalaysiaDataset.csv')
print("Malaysia Dataset info:")
print("Dataset length : ", len(dataset))
print("Dataset colums : ", dataset.columns)
