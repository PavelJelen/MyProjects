

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Arbitrary parameters for Monte Carlo
n = 100
beta = 0.05
M = 1000
p = 50

# 3 placeholder matrices
select_adj_r2 = np.zeros((M, 45))
select_BIC = np.zeros((M, 45))
select_AIC = np.zeros((M, 45))

for x in range(M):
    Predictors = np.random.randn(n, p)
    errors = np.random.randn(n)
    Target = np.sum(beta * Predictors[:, :5], axis=1) + errors
    TrueModel = sm.OLS(Target,Predictors[:,:5]).fit()
    true_r2 = TrueModel.rsquared_adj
    true_AIC = TrueModel.aic
    true_BIC = TrueModel.bic
    for m in range(6, p + 1):
        FalseModel = sm.OLS(Target, Predictors[:,:m]).fit()
        if(true_r2 > FalseModel.rsquared_adj): select_adj_r2[x, m-6] = 1
        if(true_AIC < FalseModel.aic): select_AIC[x, m-6] = 1
        if(true_BIC < FalseModel.bic): select_BIC[x, m-6] = 1

# Plotting the results
Grid_R2 = np.sum(select_adj_r2, axis=0) / M
Grid_AIC = np.sum(select_AIC, axis=0) / M
Grid_BIC = np.sum(select_BIC, axis=0) / M
sequence = np.arange(1, 46, 1)
df = pd.DataFrame({'sequence': sequence, 'Grid_R2': Grid_R2, 'Grid_AIC': Grid_AIC, 'Grid_BIC': Grid_BIC})

plt.plot(df['sequence'], df['Grid_R2'], color="red", label='Grid_R2')
plt.plot(df['sequence'], df['Grid_AIC'], color="blue", label='Grid_AIC')
plt.plot(df['sequence'], df['Grid_BIC'], color="green", label='Grid_BIC')
plt.legend(loc='best')
plt.xlabel('sequence')
plt.ylabel('Score')
plt.show()

# Monte Carlo simulation is a simulation experiment that repeatedly generates (random) variables
# to conduct some (statistical) analysis. It is a simple way to obtain a representative distribution of a result for a specic, known process.

# We see that the plot is aligned with a discussion on topic of model selection => R_2 (even though it is adjusted) favors 
# more complex models. In other words, adding more parameters will result in higher R2 although the true model
# is different than the more complex models. 
# Moreover, we see that the AIC and BIC, especially BIC which is used for consistent model selection, will yield
# much better results when it comes to model selection. BIC always selects the true model over the more complex models and thus
# we verify the purpose of BIC -> Consistently select the true model among a set of models
