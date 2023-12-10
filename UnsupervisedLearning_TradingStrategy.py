# -*- coding: utf-8 -*-
"""
Based on code by Luke Hardy
"""

from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta

# Load data
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()

end_date = "2023-09-27"
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)

df = yf.download(tickers = symbols_list, start = start_date, end = end_date)

df = df.stack()
df.index.names = ["date", "ticker"]
df.columns = df.columns.str.lower()

# Calculate features and technical indicators for each stock

    # Garman-Klass Volatility
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

    # rsi
df["rsi"] = df.groupby(level = 1)["adj close"].transform(lambda x: pandas_ta.rsi(close=x,length=20))

# BB - using log1p to normalize the distribution
df["bb_low"] = df.groupby(level = 1)["adj close"].transform(lambda x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,0])
df["bb_mid"] = df.groupby(level = 1)["adj close"].transform(lambda x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,1])
df["bb_high"] = df.groupby(level = 1)["adj close"].transform(lambda x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,2])

    # average true range
def calculate_atr (stock_date):
    atr = pandas_ta.atr(high=stock_date["high"],
                        low = stock_date["low"],
                        close = stock_date["close"],
                        length = 14)
    return atr.sub(atr.mean()).div(atr.std()) # normalizing output

df["atr"] = df.groupby(level = 1, group_keys = False).apply(calculate_atr)

    # MACD
def calculate_macd(close):
    macd = pandas_ta.macd(close = close, length = 20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

df["macd"] = df.groupby(level = 1, group_keys = False)["adj close"].apply(calculate_macd)

    # dollar volume
df["dollar_volume"] = (df["adj close"]*df["volume"])/1000000


# Aggregate to monthly level and filter top 150 most liquid stock for each month


    # Modifying the data in a way that the we take the mean of dollar volume for each month as well as the last observation of technical indicators for each month
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open','high', 'low', 'close']]

data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                  axis=1)).dropna()

    #Calculate 5y rolling average of dollar volume
data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

# Calculate monthly returns for different horizons to add additional features (aka lags)

    # using geometric mean returns
def calculate_returns(df):
    cutoff = 0.005
    lags = [1,2,3,6,9,12]
    for lag in lags:
        df[f"returns_{lag}m"] = (df["adj close"].pct_change(lag).pipe(lambda x: x.clip(lower  = x.quantile(cutoff),upper = x.quantile(1-cutoff))).add(1).pow(1/lag).sub(1))
    return df    

data = data.groupby(level = 1, group_keys = False).apply(calculate_returns).dropna()

# Fama-French Factors and rolling factor betas

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100) # Factors are as of the beginning of month, we just artifically stamp them here as end-of-month and divide by 100
factor_data.index.name = 'date'
factor_data = factor_data.join(data['returns_1m']).sort_index() # We are joining the end-of-month returns to the start-of-month factors and we will regress it -> figure out whether it is predictive or not
factor_data

    # filter out stock with less than 10 months of data
observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

    # Calculate rolling factor betas
betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['returns_1m'], 
                                     exog=sm.add_constant(x.drop('returns_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

# Now we need to join the betas to the original dataset. 
# We also have to take into account look ahead bias and therefore shift the betas one month ahead 
data = (data.join(betas.groupby('ticker').shift()))

# Fill NAs with mean
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
data = data.drop('adj close', axis=1)
data = data.dropna()
data.info()

# We have 18 features and we are ready to apply some ML models. We will focus only on long portfolios

# K-means clustering algorithm
from sklearn.cluster import KMeans

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init="random").fit(df).labels_
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

# Visualize clusters for each month
def plot_clusters(data):

    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,0] , cluster_0.iloc[:,6] , color = 'red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,0] , cluster_1.iloc[:,6] , color = 'green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,0] , cluster_2.iloc[:,6] , color = 'blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,0] , cluster_3.iloc[:,6] , color = 'black', label='cluster 3')
    
    plt.legend()
    plt.show()
    return

plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist():
    g = data.xs(i, level=0)
    plt.title(f'Date {i}')
    plot_clusters(g)
    
# As we used random initialization, the clusters are dispersed randomly across RSI
# If we want to invest in momentum stocks (RSI around 70) then we need to initialize the clusters
target_rsi_values = [30,45,55,70]
initial_centroids = np.zeros((len(target_rsi_values),18))
initial_centroids[:,6] = target_rsi_values

# Lets do k means clustering again
data = data.drop('cluster', axis=1)

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df

data = data.groupby('date', group_keys=False).apply(get_clusters)

# visualize again
plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist():
    g = data.xs(i, level=0)
    plt.title(f'Date {i}')
    plot_clusters(g)
    
# For each month select assets to invest in based on cluster and form a portfolio based on Efficient Frontier and Sharpe ratio maximization
filtered_df = data[data["cluster"]==3].copy()

filtered_df = filtered_df.reset_index(level = 1)

filtered_df.index = filtered_df.index + pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(["date", "ticker"])

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    
# Portfolio optimization
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices = prices, frequency = 252)
    cov = risk_models.sample_cov(prices = prices, frequency = 252)
    ef = EfficientFrontier(expected_returns = returns, cov_matrix = cov, weight_bounds = (lower_bound , 0.1), solver = "SCS")
    ef.max_sharpe()
    return ef.clean_weights()

# download fresh daily prices for only the filtered stocks
stocks = data.index.get_level_values("ticker").unique().tolist()
new_df = yf.download(tickers = stocks, start = data.index.get_level_values("date").unique()[0]-pd.DateOffset(months = 12),end = data.index.get_level_values("date").unique()[-1])

# calculate daily returns for each stock in our portfolio
# then loop over each months start and calculate weights for next month
# use max sharpe ratio optimization or equally-weighted weights
# calculate each day portfolio return
returns_dataframe = np.log(new_df["Adj Close"]).diff()

portfolio_df = pd.DataFrame() 

for start_date in fixed_dates.keys():
    
    try:

        end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

        cols = fixed_dates[start_date]

        optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')

        optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        
        optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]
        
        success = False
        try:
            weights = optimize_weights(prices=optimization_df,
                                   lower_bound=round(1/(len(optimization_df.columns)*2),3))

            weights = pd.DataFrame(weights, index=pd.Series(0))
            
            success = True
        except:
            print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
        
        if success==False:
            weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                     index=optimization_df.columns.tolist(),
                                     columns=pd.Series(0)).T
        
        temp_df = returns_dataframe[start_date:end_date]

        temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                   .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                          left_index=True,
                          right_index=True)\
                   .reset_index().set_index(['Date', 'index']).unstack().stack()

        temp_df.index.names = ['date', 'ticker']

        temp_df['weighted_return'] = temp_df['return']*temp_df['weight']

        temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')

        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
    
    except Exception as e:
        print(e)

# Finally, visualize portfolio returns and benchmark w/ SP500
spy = yf.download(tickers = "SPY", start = "2015-01-01", end = dt.date.today())
spy_ret = np.log(spy[["Adj Close"]]).diff().dropna().rename({"Adj Close":"SPY Buy_Hold"}, axis = 1)

portfolio_spy_df = portfolio_df.merge(spy_ret, left_index = True, right_index = True)

portfolio_cum_returns = np.exp(np.log1p(portfolio_spy_df).cumsum())-1

import matplotlib.ticker as mtick

plt.style.use('ggplot')

portfolio_cum_returns.plot(figsize=(16,6))

plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()