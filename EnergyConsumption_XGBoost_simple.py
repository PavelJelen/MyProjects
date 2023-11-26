# -*- coding: utf-8 -*-

# Importing dependencies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Read and inspect data
df = pd.read_csv("C:/Users/pavel/Desktop/python project/PJME_hourly.csv")
df.head()
# Changing index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index("Datetime")
df = df.sort_index()


# Outlier removal
df.query("PJME_MW < 20000").plot(style =".") # we can query even further
df.query("PJME_MW < 19000").plot(style =".") 
df = df.query("PJME_MW > 19000").copy() # remove everything thats below 19000MW

# Create features
def feature_create(df):
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Quarter"] = df.index.quarter
    df["Month"] = df.index.month
    df["Year"] = df.index.year
    df["DayOfYear"] = df.index.dayofyear
    return df  

# Visualize Feature and Target
df = feature_create(df)
sns.boxplot(data=df, x="Hour", y = "PJME_MW")
sns.boxplot(data=df, x="Month", y = "PJME_MW")

# Simple split into train and test
train = df.loc[df.index < "01-01-2015"]
test = df.loc[df.index >= "01-01-2015"]

fig, ax = plt.subplots()
train.iloc[:,0].plot(ax=ax, label = "Train")
test.iloc[:,0].plot(ax=ax, label= "Test")
plt.show()

# Create XGBoost model
features = ['Hour', 'DayOfWeek', 'Quarter', 'Month', 'Year', 'DayOfYear']
target = ["PJME_MW"]

X_train = train[features]
Y_train = train[target]

X_test = test[features]
Y_test = test[target]

reg = xgb.XGBRegressor(n_estimators = 1000,learning_rate = 0.01)
reg.fit(X_train, Y_train, verbose = 100)

# Feature importance
FI = pd.DataFrame(data = reg.feature_importances_, index=reg.feature_names_in_, columns = ["Importance"])
FI.sort_values("Importance").plot(kind = "barh", title = "Feature Importance")
plt.show()

# Predict on the test set 
test["Preds"] = reg.predict(X_test)
df = df.merge(test[["Preds"]], how = "left", left_index = True, right_index=True)

# Visualize prediction
ax = df[["PJME_MW"]].plot()
df[["Preds"]].plot(ax=ax, style =".")
plt.show()

# Metric
Score_no_tune = np.sqrt(mean_squared_error(test[["PJME_MW"]], test[["Preds"]])) # = 3945

