# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:04:51 2023

@author: pavel
"""

# Importing dependencies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from hypopt import GridSearch

# Start as before
df2 = pd.read_csv("C:/Users/pavel/Desktop/python project/PJME_hourly.csv")
df2.head()
# Setting datetime
df2['Datetime'] = pd.to_datetime(df2['Datetime'])
df2 = df2.set_index("Datetime")
df2 = df2.sort_index()
df2.plot() # We see some outliers


# Outlier removal
df2.query("PJME_MW < 20000").plot(style =".") # we can query even further
df2.query("PJME_MW < 19000").plot(style =".") 
df2 = df2.query("PJME_MW > 19000").copy() # remove everything thats below 19000MW

# Create features
def feature_create(df):
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Quarter"] = df.index.quarter
    df["Month"] = df.index.month
    df["Year"] = df.index.year
    df["DayOfYear"] = df.index.dayofyear
    return df 
df2 = feature_create(df2)

# Add lags
target_map = df2["PJME_MW"].to_dict()
df2["1ylag"] = (df2.index - pd.Timedelta(days=364)).map(target_map)
df2["2ylag"] = (df2.index - pd.Timedelta(days=728)).map(target_map)
df2["3ylag"] = (df2.index - pd.Timedelta(days=1092)).map(target_map)

df2 = df2.dropna()

# train/holdout split
train = df2.loc[df2.index < "01-01-2015"]
holdout = df2.loc[df2.index >= "01-01-2015"]

# Lets do some RF hyper-parameter tuning (only the most important parameters due to computational capacity)
# along with timeseries split
from sklearn.ensemble import RandomForestRegressor

features = ['Hour', 'DayOfWeek', 'Quarter', 'Month', 'Year', 'DayOfYear','1ylag', '2ylag', '3ylag']
target = ["PJME_MW"]

X_train = train[features]
y_train = train[target].values.ravel()
X_holdout = holdout[features]
y_holdout = holdout[target]

tscv = TimeSeriesSplit(n_splits = 5, test_size = 24*365*2, gap = 24) # test_size of 2 years

parameters = {
    'n_estimators': [500, 1000],
    'max_depth': [3, 6, 9],
    'max_features': [2, 3, 4], # Max features recommended for regression tasks is no. of features / 3
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 4]}

rf = RandomForestRegressor()
gridsearch = GridSearchCV(estimator = rf, param_grid = parameters, cv=tscv)
gridsearch.fit(X_train, y_train)

print('\n Best hyperparameters:')
print(gridsearch.best_params_)
print(gridsearch.best_estimator_)

preds = gridsearch.predict(X_holdout)

plt.plot(pd.DataFrame(index = y_holdout.index, data = preds))
plt.plot(y_holdout)
plt.show()

plt.plot(pd.DataFrame(index = y_holdout.index, data = preds)[-100:], label = "predictions")
plt.plot(y_holdout[-100:], label = "ground truth")
plt.legend()
plt.show()

# Results
from sklearn.metrics import mean_absolute_percentage_error

print(f' MAPE score for holdout set: {mean_absolute_percentage_error(y_holdout, preds):0.4f}') # 0.0904 = 9%
