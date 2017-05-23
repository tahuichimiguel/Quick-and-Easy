import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = 'C:/Users/02laram26/Desktop/occupancy_data'
file  ='datatraining.txt'

#%% Load Data and Inspect
os.chdir(path)
DT = pd.read_csv(file,sep=',')

DT.info()
DT.isnull().values.any()
DT.head(10)

#%% EDA

DT.plot(x='Light',y='Temperature',kind='scatter')
plt.title('Light vs.Temperature')
plt.show()

DT['Humidity'].plot()
plt.title('Humidity')
plt.show()

DT['Light'].plot()
plt.title('Light')
plt.show()

DT['CO2'].plot()
plt.title('CO2')
plt.show()

sns.pairplot(DT,hue = 'Occupancy',diag_kind = 'hist')