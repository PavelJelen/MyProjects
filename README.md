# MyProjects

## Energy Consumption Forecasting
This dataset consists of ca 16 years (2002 - 2018) of hourly data on energy consumption. This dataset is available by PJM Interconnection LLC (PJM) which is a regional transmission organization (RTO) in the United States.
With this data at hand, I conducted several machine learning models. I divided the data so that my holdout set consists of ca 2.5 years (2015 - 2018). The rest is used for training and validation. 
It is important to note that this data is already cleaned and only limited time is spent on feature engineering. The focus is mainly on the implementation of different machine learning models and their comparison. 
For comparison of the models, I used an intuitive Mean Average Percentage Error (MAPE) which states the average percentage error between the forecasted value and the ground truth value. Of course, more involved loss function or information criteria might have been used. 

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
Moving on, I create a portfolio based on the highest RSI cluster for each month.
I **optimize the weights** for each stock on monthly basis using EfficientFrontier optimizer to maximize Sharpe Ratio and ensure diversification by using a lower bound of half of equally-weighted weights and upper bound of 10% for a single stock.   
Finally, I compare this portfolio to the returns of SP500 index   

![Unsupervised learning](https://github.com/PavelJelen/MyProjects/assets/151863506/3327b318-835c-46f5-a7e4-8893e6c00d51)

## Model Selection using Monte Carlo 
In this project, I take a closer look on the topic of "Model Selection". The topic of model selection stems from the bias-variance trade-off. Usually, in order to select a model, one uses an "empirical risk minimizer" such as some loss function which yields the minimal training error.However, such a training error is overly optimistic and there is a need to account for this optimism. Therefore, we are more interested in an average expected loss aka test error. In the machine learning community, cross-validation is typically used for this purpose. Nevertheless, in this project, I use information criteria such as AIC and BIC which appropriately account for the optimism by favouring less complex models and penalizing number of coefficients. There are also different purposes of the information criteria. For example, AIC is equivalent to picking best-predicting model in large samples whereas BIC is equivalent to consistent model selection (picking a true model). 
I contrast these two information criteria with a traditional adjusted R_squared.

![ModelSelection_MonteCarlo](https://github.com/PavelJelen/MyProjects/assets/151863506/448cbeab-874b-40f4-bdbb-5f5882c6974d)


We see that the plot is aligned with the discussion on the topic of model selection => R_squared (even though it is adjusted) favors 
more complex models. In other words, adding more parameters results in higher R2 although the true model is different than the more complex models. 
Finally, we see that the AIC and BIC, especially BIC, yields much better results when it comes to model selection. BIC always selects the true model over the more complex models and thus we verify the purpose of BIC -> Consistently select the true model among a set of models
