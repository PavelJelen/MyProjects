# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:21:23 2023

@author: pavel
"""
import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import mean_squared_error 

# Loading data
df = pd.read_csv("C:/Users/pavel/Desktop/python project/PJME_hourly.csv", index_col = [0], parse_dates=[0])
df = df.sort_index()

color_pal = sns.color_palette()
df.plot(style = ".", color = color_pal[0], title = "PJME_MW")

# Remove outliers
df.query("PJME_MW < 20000").plot(style =".") # we can query even further
df.query("PJME_MW < 19000").plot(style =".") 
df = df.query("PJME_MW > 19000").copy() # remove everything thats below 19000MW

# Simple train/val/holdout split
train = df.loc[(df.index >= "28-12-2004") & (df.index < "01-01-2013")]
val = df.loc[(df.index >= "01-01-2013") & (df.index < "01-01-2015")]
holdout = df.loc[df.index >= "01-01-2015"]

# Normalization of data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train)
train = scaler.transform(train)
val = scaler.transform(val)
holdout = scaler.transform(holdout)

# Making series to make it appropriate for feeding LSTM 
train = pd.Series(train.flatten())
val = pd.Series(val.flatten())
holdout = pd.Series(holdout.flatten())

# Function to create input and output for LSTM using a rolling window
def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

# Use the function
window_size = 5
X_train, y_train = df_to_X_y(train)
X_val,y_val = df_to_X_y(val)
X_holdout, y_holdout = df_to_X_y(holdout)

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_holdout.shape, y_holdout.shape


# LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()
 
cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size = 64, callbacks=[cp1]) 

# Evaluate the model
from tensorflow.keras.models import load_model
model1_final = load_model("model1/")

holdout_predictions = model1_final.predict(X_holdout).flatten()
holdout_results = pd.DataFrame(data = {"Hold Pred":holdout_predictions, "Actuals": y_holdout})
plt.plot(holdout_results["Hold Pred"], label = "predictions")
plt.plot(holdout_results["Actuals"], label = "ground truth")
plt.legend()
plt.show()

# Inverse transformation and results
y_holdout_2d = pd.DataFrame(y_holdout)
holdout_predictions_2d = pd.DataFrame(holdout_predictions)

y_holdout_inv = scaler.inverse_transform(y_holdout_2d)
holdout_predictions_inv = scaler.inverse_transform(holdout_predictions_2d)

plt.plot(holdout_predictions_inv, label = "predictions")
plt.plot(y_holdout_inv, label = "ground truth")
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_percentage_error

print(f' MAPE score for holdout set: {mean_absolute_percentage_error(y_holdout_inv, holdout_predictions_inv):0.4f}') # 0.0132 = 1.3%


