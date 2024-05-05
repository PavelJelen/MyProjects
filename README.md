# MyProjects

## Energy Consumption Forecasting
This dataset consists of ca 16 years (2002 - 2018) of hourly data on energy consumption. This dataset is available by PJM Interconnection LLC (PJM)   
With this data at hand, I conducted several machine learning models. I divided the data so that the holdout set consists of ca 2.5 years (2015 - 2018). The rest is used for training and validation. 
It is important to note that this data is already cleaned and only limited time is spent on feature engineering. The focus is mainly on the implementation of different machine learning models and their comparison. 
For comparison of the models, I used an intuition-friendly Mean Average Percentage Error (MAPE) which states the average percentage error between the forecasted value and the ground truth value. Of course, more involved loss function or information criteria might have been used. 

### Simple XGBoost
Feature engineering:  
1) Outlier removal  
2) Feature creation: Day of week, Quarter, Month, Year, Day of Year

Train/test split:  
Datapoints < 01-01-2015 used for training, the rest for test

XGBoost implementation:  
n_estimators = 1000  
learning_rate = 0.01  

**Mean Average Percentage Error (MAPE):** 9.32% 

### XGBoost w/ hyperparameter tuning and time series cv
Feature engineering:  
1) Outlier removal  
2) Feature creation: Day of week, Quarter, Month, Year, Day of Year
3) Feature creation: 1-year lag, 2-year lag, 3-year lag
4) Drop n/a values due to the lag creation

Train/test split:  
Datapoints < 01-01-2015 used for training, the rest for test

XGBoost implementation:
1) Time series 5-k cross-validation with validation size of cca 2 years and gap of 24 datapoints
2) Tuned parameters: max_depth, learning_rate, n_estimators, colsample_bytree

**Mean Average Percentage Error (MAPE):** 9.27% 

### Random Forest w/ hyperparameter tuning and time series cv
Feature engineering, train/test split same as before

Random Forest implementation:
1) Time series 5-k cross-validation with validation size of cca 2 years and gap of 24 datapoints
2) Tuned parameters: n_estimators, max_depth, max_feature, min_sample_leaf, min_sample_split
   
**Mean Average Percentage Error (MAPE):** 9.04% 

### LSTM Neural Network 
Feature engineering:  
Outlier removal  
Normalization of data  
Sequential input/output for feeding LSTM (window size of 5)  

Train/Val/Test split:  
2004 > Datapoints < 2013 used for training (this follows the dataset size available for XGB and RF after removing n/a due to lag creation)  
2013 > Datapoints < 2015 used for validation, the rest for test  

LSTM implementation:  
1) Initial LSTM of 64 units  
2) Added 1 more dense layer with ReLu activation of 8 units  
3) Optimizer = Adam with learning_rate of 0.0001  
4) batch_size of 64 epochs of 10
5) loss = MSE

**Mean Average Percentage Error (MAPE):** 1.32%

### Conclusion:
Additionally, I have implemented **Prophet** model as well as **SARIMA** model. The Prophet model has not performed as good as the other models. Moreover, the SARIMA model has failed due to convergence error. This stems from mutliple seasonality in the data. One possible remedy would be to model each hour separetely resulting in 24 different SARIMA models. Lastly, SARIMA is more suited for short-term forecasting than a long-term such as 2 years

Finally, we see that LSTM NN yields the best MAPE. This might be due to several reasons. Firstly, there could definitely be done more feature engineering when it comes to XGB and RF which would in turn improve MAPE. The same applies to more involved hyperparameter tuning. Lastly, this type of data (long-term time-series) is where deep learning such as LSTM generally performs really well.  

## Trading Strategies
In this project, I explore various trading strategies ranging from simple strategies based on technical indicators to more involved quantitative strategies 
### Unsupervised Learning
This strategy consisted of selecting 150 most liquid stocks from SP500 index and **computing 18 different features such as Bollinger Bands, MACD, RSI or Fama-French rolling betas**. These F-F risk factors has been shown to empirically explain asset returns. Therefore, it is reasonable to include exposure to these factors as financial features   
Afterwards, **K-Means clustering** is applied to the whole range of stocks for each month. The main focus is a long-only momentum strategy, i.e., invest in stocks which have high momentum (high RSI) on monthly basis.
Firstly, I create a portfolio based on the highest RSI cluster for each month.
I **optimize the weights** for each stock on monthly basis using EfficientFrontier optimizer to maximize Sharpe Ratio and ensure diversification by using a lower bound of half of equally-weighted weights and upper bound of 10% for a single stock.   
Finally, I compare this portfolio to the returns of SP500 index   

![Unsupervised learning](https://github.com/PavelJelen/MyProjects/assets/151863506/3327b318-835c-46f5-a7e4-8893e6c00d51)

### Multivariate LSTM Classification of S&P500   
This strategy revolves around classification of next days' UP or DOWN moves. This strategy is based on the extracting multiple features from Magnificent 7 stocks - Apple, Amazon, Tesla, Nvidia, Alphabet, Meta, Microsoft to predict the S&P500 moves.    
The daily data ranges from 2014 to 2024 and I extract **multiple features of magnificant 7 companies** - 20MA, 50MA, 100MA, RSI, Bollinger Bands and Volume as well as previous day's Close and Volume of S&P500   
This provides around 65 features to be used for classification.
The window to be used for LSTM input shapes is 5   
After modifying the parameters for LSTM several times in order to produce best results, the LSTM is constructed using 3 hidden layers of 64 neurons and batch size of 32.   
The resulting accuracy for the test set is ca 54%.   
The strategy buys (sells) each day based on the prediction being up (down). This is compared to the buy&hold strategy. 

![AlgoTrading_LSTM_Classification](https://github.com/PavelJelen/MyProjects/assets/151863506/1a2967db-d42b-4b4c-8d84-490576b8218f)   


### Twitter Sentiment
This strategy is based on analyzing the sentiment considering various tickers on Twitter. The data consist of variables such as Ticker, Number of Posts, Likes and Comments. As it is often the case with alternative data, there is a need to find some kind of derivative of this data in order for them to be valuable. Therefore, I did not base my strategy on solely using the number of posts, comment or likes and I created an engagement ratio of comments/likes. This prevents from being influenced by various bots hyping a stock and making artifical posts about it.   
I rebalance the stock portfolio monthly and each time select only 5 stocks with the highest engagement ratio based on equal-weights.   
Finally, I compare the strategy return to the benchmark of Nasdaq returns.   

![Twitter Engagement Strategy](https://github.com/PavelJelen/MyProjects/assets/151863506/7fd33c37-1d48-460a-a356-4f092b06dfcb)

### Intraday Volatility using GARCH
This trategy is based on simulated daily and 5-min one-asset data. First, I fit the GARCH model on the daily data to predict one-day ahead volatility in a rolling window. This prediction is used to calculate a features called prediction premium and its associated standard deviation which are used to generate a daily signal. Next, the daily data is merged with the 5-min data. Using the 5-min data, I calculate RSI and Bollinger Bands technical indicators which are used to generate the intraday signal. That being said, I have two signals for each 5-min tick - one from daily timeframe and the other from 5-min timeframe.   
The final trading strategy is based on mean reversion logic. I only care about the first trading signal of the day and hold it until end of the day. 

![Intraday Strategy Return](https://github.com/PavelJelen/MyProjects/assets/151863506/6d177c70-5d7d-40e5-bae6-2f82ff404211)


## Model Selection using Monte Carlo 
In this project, I take a closer look on the topic of "Model Selection". The topic of model selection stems from the bias-variance trade-off. Usually, in order to select a model, one uses an "empirical risk minimizer" such as some loss function which yields the minimal training error.However, such a training error is overly optimistic and there is a need to account for this optimism. Therefore, we are more interested in an average expected loss aka test error. In the machine learning community, cross-validation is typically used for this purpose. Nevertheless, in this project, I use information criteria such as AIC and BIC which appropriately account for the optimism by favouring less complex models and penalizing number of coefficients. There are also different purposes of the information criteria. For example, AIC is equivalent to picking best-predicting model in large samples whereas BIC is equivalent to consistent model selection (picking a true model). 
I contrast these two information criteria with a traditional adjusted R_squared.

![ModelSelection_MonteCarlo](https://github.com/PavelJelen/MyProjects/assets/151863506/448cbeab-874b-40f4-bdbb-5f5882c6974d)


We see that the plot is aligned with the discussion on the topic of model selection => R_squared (even though it is adjusted) favors 
more complex models. In other words, adding more parameters results in higher R2 although the true model is different than the more complex models. 
Finally, we see that the AIC and BIC, especially BIC, yields much better results when it comes to model selection. BIC always selects the true model over the more complex models and thus we verify the purpose of BIC -> Consistently select the true model among a set of models
