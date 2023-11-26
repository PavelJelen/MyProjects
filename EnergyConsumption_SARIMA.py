# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 18:55:43 2023

@author: pavel
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import mean_squared_error

# Loading data
df = pd.read_csv("C:/Users/pavel/Desktop/python project/PJME_hourly.csv", index_col = [0], parse_dates=[0])
df = df.sort_index()

# EDA
    # Remove outliers
df.plot()    
df.query("PJME_MW < 20000").plot(style =".") # we can query even further
df.query("PJME_MW < 19000").plot(style =".") 
df = df.query("PJME_MW > 19000").copy() # remove everything thats below 19000MW

# Split to train/holdout at this moment to prevent data leakage when selecting p/d/q of ARIMA
train = df.loc[df.index < "01-01-2015"]
holdout = df.loc[df.index >= "01-01-2015"]


# Check for stationarity
    # Plot
color_pal = sns.color_palette()
train.plot(style = ".", color = color_pal[0], title = "PJME_MW")

    # ADF test
from statsmodels.tsa.stattools import adfuller 

def adfuller_test(timeseries):
    results = adfuller(timeseries)
    labels = ["ADF_test_stat", "p-val", "#Lags_used", "Nobs_used"]
    for value,label in zip (results,labels):
        print(label + ":" +str(value))
    if results[1] <= 0.05:
            print("We have a strong evidence against H0 meaning we conclude timeseries to be stationary")
    else:
            print("We failed to reject H0, time series seems to have a unit root and thus indicates non-stationarity ")

adfuller_test(train) # Stationary (thus parameter d = 0)

#  Determine the p and q parameters of ARIMA model
    # Plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(train)
plot_pacf(train)
# We see a huge seasonal effect -> basic ARIMA wont work well but lets try 

    # Auto-select of ARIMA
from pmdarima import auto_arima
stepwise_fit = auto_arima(train, seasonal = False, trace = True) # This suggest ARIMA(5,1,1)

# Train the ARIMA model
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order = (5,1,1))
model_fit = model.fit(method_kwargs={'maxiter':300})
start=len(train)
end = start + len(holdout)-1
preds = model_fit.predict(start = start, end = end)
preds.index = holdout.index

# Visual evaluation
plt.plot(preds)
plt.plot(holdout)
plt.show() # We see a very bad performance of ARIMA model -> expected as we do not consider seasonality

# I accidentally did not save SARIMA which is basically just about modifying line 58 
# by saying seasonal = True and also specicifying the "m" which stands for frequency
# It did not converge due to multiple seasonalities (poor model fit) 
# search on google for SARIMA hourly data for reference (poor model, 24 models etc.)