# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:40:11 2023

@author: pavel
"""

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import os
import matplotlib.pyplot as plt
plt.style.use("ggplot")

data_folder = r"C:\Users\pavel\Desktop\python project"

sentiment_data = pd.read_csv(os.path.join(data_folder,"sentiment_data.csv")) #TwitterSentiment calculted by the data provider. We dont use it

sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])

sentiment_data = sentiment_data.set_index(["date", "symbol"])

# As there might be bots on Twitter hyping up a stock by posting about it and thus skewing the data,
# we calculate engagement ratio using comments and likes

sentiment_data["engagement_ratio"] = sentiment_data["twitterComments"]/sentiment_data["twitterLikes"]

sentiment_data = sentiment_data[(sentiment_data["twitterLikes"]>20) & (sentiment_data["twitterComments"]>10)]

# Aggregate the data monthly and calculate average sentiment

agg_df = (sentiment_data.reset_index("symbol").groupby([pd.Grouper(freq="M"), "symbol"])[["engagement_ratio"]].mean())

agg_df["rank"] = agg_df.groupby(level = 0)["engagement_ratio"].transform(lambda x: x.rank(ascending = False))

# Select top stocks based on their rank of engagement

filtered_df = agg_df[agg_df["rank"]<6].copy()

filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index+pd.DateOffset(days = 1)

filtered_df = filtered_df.reset_index().set_index(["date","symbol"])

# Extract the stocks to form portfolios for each month

dates = filtered_df.index.get_level_values("date").unique().tolist()

fixed_dates = {}

for d in dates:
    fixed_dates[d.strftime("%Y-%m-%d")] = filtered_df.xs(d,level = 0).index.tolist()
    
# Download stock prices for the shortlisted stocks

stocks_list = sentiment_data.index.get_level_values("symbol").unique().tolist()

prices = yf.download(tickers = stocks_list, start = "2021-01-01", end = "2023-03-01")

# Portfolio returns with monthly rebalancing

returns_df = np.log(prices["Adj Close"]).diff()
returns_df = returns_df.drop("ATVI", axis = 1)
returns_df = returns_df.dropna()

portfolios = pd.DataFrame()

for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd()).strftime("%Y-%m-%d")
    cols = fixed_dates[start_date]
    temp_df = returns_df[pd.to_datetime(start_date):pd.to_datetime(end_date)][cols].mean(axis = 1).to_frame("portfolio_return") # Taking the daily return from start to end for each stock and then and taking a mean return of the whole portfolio for each day
    portfolios = pd.concat([portfolios,temp_df], axis = 0)
    
# Compare the strategy to a benchmark (survivorship bias)

benchmark_ret = yf.download(tickers = "QQQ", start = "2021-01-01", end = "2023-03-01") # Nasdaq

benchmark_ret = np.log(benchmark_ret["Adj Close"]).diff().dropna().to_frame("Nasdaq_ret")

port_bench = portfolios.merge(benchmark_ret, left_index = True, right_index = True)
    
# Visualize 
import matplotlib.ticker as mtick

portfolios_cumulative_return = np.exp(np.log1p(port_bench).cumsum()).sub(1)
portfolios_cumulative_return.plot()
plt.title('Twitter Engagement Ratio Strategy Return Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.show()

# Show plots when ranking based on solely likes or comments -> the strategy is pretty bad
# This means that even with alternative data, we need to create some "derivative" features which are potentially valuable to us
