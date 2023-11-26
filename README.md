# MyProjects

## Energy Consumption Forecasting
This dataset consists of cca 16 (2002 - 2018) years of hourly data on energy consumption. This dataset is available by PJM Interconnection LLC (PJM) which is a regional transmission organization (RTO) in the United States.
With this data at hand, I conducted several machine learning models. I divided the data so that my holdout set consists of 3 years (2015 - 2018). The rest is used for training and validation. 
It is important to note that this data is already cleaned and only limited time is spent on feature engineering. The focus is mainly on the implementation of different machine learning models and their comparison. 
For comparison of the models, I used an intuitive Mean Average Percentage Error (MAPE) which states the average percentage error between the forecasted value and the ground truth value. Of course, more involved loss function or information criteria might have been used. 

### Simple XGBoost
Feature engineering steps:
  Outlier removal
  Feature creation: Hours, Day of week, Quarter, Month, Year, Day of year
  
Simple train/test split
  Datapoints earlier than 01-01-2015 used for training

XGBoost Implementation:
  n_estimators = 1000
  learning rate = 0.01

**Mean Average Percentage Error (MAPE):** 9.3% 




## Model Selection using Monte Carlo 
In this project, I take a closer look on the topic of "Model Selection". The topic of model selection stems from the bias-variance trade-off. Usually, in order to select a model, one uses an "empirical risk minimizer" such as some loss function which yields the minimal training error.However, such a training error is overly optimistic and there is a need to account for this optimism. Therefore, we are more interested in an average expected loss aka test error. In the machine learning community, cross-validation is typically used for this purpose. Nevertheless, in this project, I use information criteria such as AIC and BIC which appropriately account for the optimism by favouring less complex models and penalizing number of coefficients. There are also different purposes of the information criteria. For example, AIC is equivalent to picking best-predicting model in large samples whereas BIC is equivalent to consistent model selection (picking a true model). 
I contrast these two information criteria with a traditional adjusted R_squared.

![ModelSelection_MonteCarlo](https://github.com/PavelJelen/MyProjects/assets/151863506/448cbeab-874b-40f4-bdbb-5f5882c6974d)


We see that the plot is aligned with the discussion on the topic of model selection => R_squared (even though it is adjusted) favors 
more complex models. In other words, adding more parameters results in higher R2 although the true model is different than the more complex models. 
Finally, we see that the AIC and BIC, especially BIC, yields much better results when it comes to model selection. BIC always selects the true model over the more complex models and thus we verify the purpose of BIC -> Consistently select the true model among a set of models
