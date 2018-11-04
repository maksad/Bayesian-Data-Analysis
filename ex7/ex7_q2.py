#%%
import matplotlib
matplotlib.use('TkAgg')

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan

#%% The data
machines = pd.read_fwf('./ex7/factory.txt').values

#%% Pooled model
'''
Pooled model
'''
stan_code_pooled = '''
data {
    int<lower=0> N;       // number of data points
    vector[N] y;          //
}
parameters {
    real mu;              // group means
    real<lower=0> sigma;  // common std
}
model {
    y ~ normal(mu, sigma);
}
generated quantities {
    real ypred;
    ypred = normal_rng(mu, sigma);
}
'''

#%% fitting data to stan model
machines_pooled = machines.flatten()
model_pooled = pystan.StanModel(model_code=stan_code_pooled)
data = dict(
    N=machines_pooled.size,
    y=machines_pooled
)

#%% sampling
fit_pooled = model_pooled.sampling(data=data)
print(fit_pooled)

#%% hist
y_pred_pooled = fit_pooled.extract()['ypred']
plt.hist(y_pred_pooled, bins=20, ec='white')
plt.savefig('./ex7/report/pooled_hist.png')
plt.figure(0)
