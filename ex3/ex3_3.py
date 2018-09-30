import matplotlib
matplotlib.use('TkAgg')
from math import sqrt
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def get_attributes(data):
    n = len(data)
    estimated_mean = np.mean(data)
    estimated_variance = stats.tvar(data)
    x_range = np.arange(
        estimated_mean - 3 * sqrt(estimated_variance),
        estimated_mean + 3 * sqrt(estimated_variance),
        0.01
    )
    mu = stats.t.pdf(
        x=x_range,
        df=n-1,
        loc=estimated_mean,
        scale=estimated_variance/n
    )
    return [n, estimated_mean, estimated_variance, x_range, mu]

y1 = [13.357,14.928,14.896,15.297,14.82,12.067,14.824,13.865,17.447]
y2 = [15.98,14.206,16.011,17.25,15.993,15.722,17.143,15.23,15.125,16.609,14.735,15.881,15.789]
n_1, estimated_mean_1, estimated_variance_1, x_range_1, mu_1 = get_attributes(y1)
n_2, estimated_mean_2, estimated_variance_2, x_range_2, mu_2 = get_attributes(y2)

mu_1_samples = stats.t.rvs(df=n_1-1, loc=estimated_mean_1, scale=estimated_variance_1/n_1, size=100000)
mu_2_samples = stats.t.rvs(df=n_2-1, loc=estimated_mean_2, scale=estimated_variance_2/n_2, size=100000)
mu_diff = mu_1_samples - mu_2_samples

plt.hist(mu_diff, bins=50, ec='white',  alpha=0.5)
plt.savefig('./ex3/report/3_3_a_histogram.png')
print('Intervals', round(np.percentile(mu_diff, 2.5), 4), round(np.percentile(mu_diff, 97.5), 4))

'''
b)
'''
print(stats.percentileofscore(mu_diff, 0), '%')
