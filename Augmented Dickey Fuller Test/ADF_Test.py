import matplotlib.pyplot as plt
import random
random.seed(5)

import statsmodels
from statsmodels.tsa.stattools import adfuller
import pandas as pd


# non_stationary_series = []
# for i in range(0,100):
#     non_stationary_series.append(random.random()+(i*.02))
#
# plt.plot(non_stationary_series)
# plt.title('Non-Stationary Time Series')
# plt.show()#
#
# stationary_series = []
# for i in range(0,100):
#     stationary_series.append(random.random())
#
# plt.plot(stationary_series)
# plt.title('Stationary Time Series')
# plt.show()
col = ["total_admission"]
dataframe = pd.read_csv('../Datasets/UKCovid-Rawdata.csv', usecols=col, engine='python')
dataframe.dropna(inplace=True)
print("Shape : ", dataframe.shape)
plt.plot(dataframe)
plt.title('Admission Time Series')
plt.show()

class StationarityTests:

    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def ADF_Stationarity_Test(self, timeseries, printResults = True):

        #Dickey-Fuller test:
        adfTest = adfuller(timeseries, autolag='AIC')

        self.pValue = adfTest[1]

        if (self.pValue<self.SignificanceLevel):
            self.isStationary = True
        else:
            self.isStationary = False

        if printResults:
            dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])

            #Add Critical Values
            for key,value in adfTest[4].items():
                dfResults['Critical Value (%s)'%key] = value

            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)

sTest = StationarityTests()
sTest.ADF_Stationarity_Test(dataframe, printResults = True)
print("Is the time series stationary? {0}".format(sTest.isStationary))
