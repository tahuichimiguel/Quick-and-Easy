import pandas as pd
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import os

path = 'C:/Users/02laram26/Desktop/occupancy_data'
file  ='datatraining.txt'

#%% Load Data and Inspect
os.chdir(path)
DT = pd.read_csv(file,sep=',')

DT.info()
DT.isnull().values.any()
DT.head(10)

DT['date']=pd.to_datetime(DT['date'])
DT.info()

#%% EDA & Time Series Analysis
 
#Time Series Plots
DT.plot(x='date',subplots=True,sharex=True,figsize = (10,15))

norm_crosscorr_occ_hratio =sig.correlate(DT['Occupancy'].values,DT['HumidityRatio'])/DT['HumidityRatio'].std()
norm_crosscorr_occ_co2 =sig.correlate(DT['Occupancy'].values,DT['CO2'])/DT['CO2'].std()
norm_crosscorr_occ_light =sig.correlate(DT['Occupancy'].values,DT['Light'])/DT['Light'].std()
norm_crosscorr_occ_temp =sig.correlate(DT['Occupancy'].values,DT['Temperature'])/DT['Temperature'].std()
norm_crosscorr_occ_humidity =sig.correlate(DT['Occupancy'].values,DT['Humidity'])/DT['Humidity'].std()

plt.subplot(111)
plt.plot(norm_crosscorr_occ_hratio, label = 'Humidity Ratio')
plt.plot(norm_crosscorr_occ_co2, label = 'CO2')
plt.plot(norm_crosscorr_occ_light, label = 'Light')
plt.plot(norm_crosscorr_occ_temp, label = 'Temperature')
plt.plot(norm_crosscorr_occ_humidity, label = 'Humidity')

plt.title('Normalized Cross Correlations with Occupancy')
plt.legend(ncol=5)
plt.show()

idx_max_temp =np.argmax(norm_crosscorr_occ_temp) 
idx_max_hratio =np.argmax(norm_crosscorr_occ_hratio) 
idx_max_humidity =np.argmax(norm_crosscorr_occ_humidity) 
idx_max_co2 =np.argmax(norm_crosscorr_occ_co2) 
idx_max_light =np.argmax(norm_crosscorr_occ_light) 

ccorr_peak = pd.DataFrame({'Temperature':idx_max_temp,
                           'Humidity Ratio':idx_max_hratio,
                           'Humidity':idx_max_humidity,
                           'CO2':idx_max_co2,
                           'Light':idx_max_light},
                            index = [1])
                          
print(ccorr_peak)

print('Total Number of Observations: ' + str(DT.shape[0]))

print("""Time series analysis shows that the degree of correlation between Occupancy and the predictors
is significantly stratified. The predictor that is most strongly correlated with occupancy is Temperature.
The least strongly correlated predictor is Light.""", end="\n")

print("""It should be noted that Humidity and Humidity Ratio exhibit nearly identical
correlation with Occupancy because Humidity Ratio is uniquely described by the Humidity and Temperature.
It would therefore be redundant to include all 3 of those variables in a predictive model.""",end="\n")

print("""Inclusion of Light as a predictor in a model may introduce noise that reduces the accuracy. This hypothesis
is implied by the numerous, periodic peaks in the cross-correlation spectra. Since peak near 8000 is comparable to
the other peaks, it is possible that Occupancy and Light are weakly dependent.  """,end = '\n')

print("""Most of the predictors have a maximal cross-correlation with Occupancy at 8142 samples, which
corresponds to a sample lag of 1. This is not true for CO2 levels that have a lag of 34 samples, which
suggests that CO2 has a significant delayed response to Occupancy. It is suggested that a model using CO2
levels as a predictor make use of lagged values""",end = '\n')

#%% 