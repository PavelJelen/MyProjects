# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 09:21:22 2023

@author: pavel
"""
# Importing dependencies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

# Lets try to make it better 
# Use:
    # More robust cross-validation
    # Add more features (lags)
    
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

# Simple train/val/holdout split
train = df2.loc[df2.index < "01-01-2015"]
holdout = df2.loc[df2.index >= "01-01-2015"]

# Lets do some XGB hyper-parameter tuning (only the most important parameters due to computational capacity)
features = ['Hour', 'DayOfWeek', 'Quarter', 'Month', 'Year', 'DayOfYear','1ylag', '2ylag', '3ylag']
target = ["PJME_MW"]

X_train = train[features]
y_train = train[target].values.ravel()
X_holdout = holdout[features]
y_holdout = holdout[target]

tscv = TimeSeriesSplit(n_splits = 5, test_size = 24*365*2, gap = 24) # test_size of 2 years

model = xgb.XGBRegressor()
parameters = {
    "max_depth": [4,8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [800, 1000],
    "colsample_bytree": [0.7, 1],
    "objective": ["reg:linear"]
}


grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv = tscv)
grid_search.fit(X_train, y_train)

print('\n Best hyperparameters:')
print(grid_search.best_params_)
print(grid_search.best_estimator_)

Preds_tune = grid_search.predict(X_holdout)
Preds_tune = pd.DataFrame(index = holdout.index, data = Preds_tune)

plt.plot(pd.DataFrame(index = y_holdout.index, data = Preds_tune))
plt.plot(y_holdout)
plt.show()

plt.plot(pd.DataFrame(index = y_holdout.index, data = Preds_tune)[-100:], label = "predictions")
plt.plot(y_holdout[-100:], label = "ground truth")
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_percentage_error

print(f' MAPE score for holdout set: {mean_absolute_percentage_error(y_holdout, Preds_tune):0.4f}') # 0.0927 = 9.3%



