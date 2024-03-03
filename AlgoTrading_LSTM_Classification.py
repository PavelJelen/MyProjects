# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:32:43 2024

@author: pavel
"""

import numpy as np
import yfinance as yf
import random
import datetime as dt
import pandas as pd
import pandas_ta


random.seed(42)
# load data
start = dt.datetime(2014,1,1)
end = dt.datetime(2024,2,24)
stockData = yf.download("^GSPC", start, end, auto_adjust=True)
SP500 = stockData[["Close", "Volume"]]
Tesla = yf.download("TSLA", start, end, auto_adjust=True)
Meta = yf.download("META", start, end, auto_adjust=True)
Nvidia = yf.download("NVDA", start, end, auto_adjust=True)
Amazon = yf.download("AMZN", start, end, auto_adjust=True)
Apple = yf.download("AAPL", start, end, auto_adjust=True)
Microsoft = yf.download("MSFT", start, end, auto_adjust=True)
Alphabet = yf.download("GOOG", start, end, auto_adjust=True)

# Create target variable
SP500["TomorrowRise"] = np.where(SP500["Close"].shift(-1) > SP500["Close"],1,0)

# Create MA
def MovingAverage(frame, name):
    #print(n)
    for i in (20,50,100):
            #print(i)
     frame[f"{i}MA_{name}"] = frame["Close"].shift(1).rolling(window=i).mean()
    return frame

Tesla = MovingAverage(Tesla, "Tesla")
Meta = MovingAverage(Meta, "Meta")
Nvidia = MovingAverage(Nvidia, "Nvidia")
Amazon = MovingAverage(Amazon, "Amazon")
Apple = MovingAverage(Apple, "Apple")
Microsoft = MovingAverage(Microsoft, "Microsoft")
Alphabet = MovingAverage(Alphabet, "Alphabet")

# Create RSI
Tesla["rsi_Tesla"] = Tesla[["Close"]].transform(lambda x: pandas_ta.rsi(close=x,length=20))
Meta["rsi_Meta"] = Meta[["Close"]].transform(lambda x: pandas_ta.rsi(close=x,length=20))
Nvidia["rsi_Nvidia"] = Nvidia[["Close"]].transform(lambda x: pandas_ta.rsi(close=x,length=20))
Amazon["rsi_Amazon"] = Amazon[["Close"]].transform(lambda x: pandas_ta.rsi(close=x,length=20))
Apple["rsi_Apple"] = Apple[["Close"]].transform(lambda x: pandas_ta.rsi(close=x,length=20))
Microsoft["rsi_Microsoft"] = Microsoft[["Close"]].transform(lambda x: pandas_ta.rsi(close=x,length=20))
Alphabet["rsi_Alphabet"] = Alphabet[["Close"]].transform(lambda x: pandas_ta.rsi(close=x,length=20))

# Creade Bollinger-Bands
def BB(frame, name):
    frame[f"bb_low_{name}"] = frame["Close"].transform(lambda x: pandas_ta.bbands(close=np.log1p(x).shift(1),length=20).iloc[:,0])
    frame[f"bb_mid_{name}"] = frame["Close"].transform(lambda x: pandas_ta.bbands(close=np.log1p(x).shift(1),length=20).iloc[:,1])
    frame[f"bb_high_{name}"] = frame["Close"].transform(lambda x: pandas_ta.bbands(close=np.log1p(x).shift(1),length=20).iloc[:,2])
    return frame

Tesla = BB(Tesla, "Tesla")
Meta = BB(Meta, "Meta")
Nvidia = BB(Nvidia, "Nvidia")
Amazon = BB(Amazon, "Amazon")
Apple = BB(Apple, "Apple")
Microsoft = BB(Microsoft, "Microsoft")
Alphabet = BB(Alphabet, "Alphabet")

# Shift volume
Tesla["Volume_tesla"] = Tesla["Volume"].shift(1)
Meta["Volume_Meta"] = Meta["Volume"].shift(1)
Nvidia["Volume_Nvidia"] = Nvidia["Volume"].shift(1)
Amazon["Volume_Amazon"] = Amazon["Volume"].shift(1)
Apple["Volume_Apple"] = Apple["Volume"].shift(1)
Microsoft["Volume_Microsoft"] = Microsoft["Volume"].shift(1)
Alphabet["Volume_Alphabet"] = Alphabet["Volume"].shift(1)
SP500["Volume_SP500"] = SP500["Volume"].shift(1)

# Shift close
Tesla["Close_tesla"] = Tesla["Close"].shift(1)
Meta["Close_Meta"] = Meta["Close"].shift(1)
Nvidia["Close_Nvidia"] = Nvidia["Close"].shift(1)
Amazon["Close_Amazon"] = Amazon["Close"].shift(1)
Apple["Close_Apple"] = Apple["Close"].shift(1)
Microsoft["Close_Microsoft"] = Microsoft["Close"].shift(1)
Alphabet["Close_Alphabet"] = Alphabet["Close"].shift(1)
SP500["Close_SP500"] = SP500["Close"].shift(1)

# Dropping columns
Tesla_final = Tesla.iloc[:,5:].copy()
Meta_final = Meta.iloc[:,5:].copy()
Nvidia_final = Nvidia.iloc[:,5:].copy()
Amazon_final = Amazon.iloc[:,5:].copy()
Apple_final = Apple.iloc[:,5:].copy()
Microsoft_final = Microsoft.iloc[:,5:].copy()
Alphabet_final = Alphabet.iloc[:,5:].copy()
SP500_final = SP500.iloc[:,2:].copy()

# Merge all dfs and drop NAs
Final_df = pd.concat([SP500_final, Tesla_final, Meta_final, Nvidia_final, Amazon_final, Apple_final, Microsoft_final, Alphabet_final], axis = 1)
Final_df = Final_df.dropna()

# Train/Val/Test - 60%/20%/20%
train, validate, test = np.split(Final_df, [int(.6*len(Final_df)), int(.8*len(Final_df))])


# Standardize each column
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train.iloc[:,1:])
train.iloc[:,1:] = scaler.transform(train.iloc[:,1:])
validate.iloc[:,1:] = scaler.transform(validate.iloc[:,1:])
test.iloc[:,1:] = scaler.transform(test.iloc[:,1:])

# Multivariare LSTM shapes
def df_to_X_y(df, window_size=5):
    """
    Convert a pandas DataFrame to X, y for multivariate time series forecasting.
    The target variable is at position 0. The rest is the feature variables
    
    Parameters:
    - df: pandas DataFrame containing the time series data with target variable and multiple features
    - window_size: size of the input window (number of time steps)
    
    Returns:
    - X: input data with shape (samples, window_size, num_features)
    - y: target data with shape (samples, num_features)
    """
    df_as_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_as_np) - window_size):
        # Extract the window of data
        window_data = df_as_np[i:i + window_size, 1:]
        X.append(window_data)
        
        # Corresponding label
        label = df_as_np[i + window_size, 0]
        y.append(label)
    
    return np.array(X), np.array(y)

window_size = 5
x_train, y_train = df_to_X_y(train, window_size)
x_val, y_val = df_to_X_y(validate, window_size)
x_test, y_test = df_to_X_y(test, window_size)

# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(InputLayer((x_train.shape[1], x_train.shape[2])))
model.add(LSTM(128))
model.add(Dense(units=64, kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(units=64, kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(units=64, kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(units=1, kernel_initializer = "uniform", activation = "sigmoid"))
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

cp1 = ModelCheckpoint('model/', save_best_only=True)
model.compile(optimizer=Adam(learning_rate=0.0000001), loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size = 32, callbacks=[cp1]) 

from tensorflow.keras.models import load_model
model_final = load_model("model/")

y_pred_prob = model_final.predict(x_test)
y_pred = (y_pred_prob > 0.5)

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

index = test.index[window_size:]

test_close = SP500.loc[index,"Close"]
y_pred = pd.DataFrame(y_pred).set_index(index)

final = pd.concat([test_close, y_pred], axis = 1)
final["TomorrowReturn"] = (np.log(final["Close"]/final["Close"].shift(1))).shift(-1)
final = final.dropna()
final.columns = ["Close", "TomorrowPrediction", "TomorrowReturn"]

final["Strategy"] = 0
final["Strategy"] = np.where(final["TomorrowPrediction"] == True, final["TomorrowReturn"], -final["TomorrowReturn"])
final["Cumulative Market Return"] = np.cumsum(final["TomorrowReturn"])
final["Cumulative Strategy Return"] = np.cumsum(final["Strategy"])

import matplotlib.pyplot as plt
plt.plot(final["Cumulative Market Return"], color = "r", label = "Market Returns")
plt.plot(final["Cumulative Strategy Return"], color = "g", label = "Strategy Returns")
plt.xlabel("Dates", {"color": "orange", "fontsize": 15})
plt.ylabel("Returns", {"color": "orange", "fontsize": 15})
plt.legend()
plt.show()