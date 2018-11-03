#%%
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pystan

drowning_data = pd.read_fwf('./Ex7/drowning.txt').values
years = drowning_data[:, 0]
drowning = drowning_data[:, 1]

#%%
plt.plot(years, drowning)

z = np.polyfit(years, drowning, 1)
trend = np.poly1d(z)
plt.plot(years, trend(years), 'r--')

plt.savefig('./ex7/report/drowining.png')
plt.figure(0)

#%%
stan_code = '''
data {
  int<lower=0> N; // number of data points
  vector[N] x;    // observation year
  vector[N] y;    // observation number of drowned
  real xpred;     // prediction year
  real tau;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] mu;
  mu = alpha + beta * x;
}
model {
  beta ~ normal(0, tau*tau);
  y ~ normal(mu, sigma);
}
generated quantities {
  real ypred;
  ypred = normal_rng(alpha + beta * xpred, sigma);
}
'''

#%% fitting data to stan model
stan_model = pystan.StanModel(model_code=stan_code)

data = dict(
    N=len(years),
    x=years,
    y=drowning,
    xpred=2019,
    tau=5,
)

#%% sampling
fit = stan_model.sampling(data=data)
print(fit)
