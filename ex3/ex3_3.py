import matplotlib
matplotlib.use('TkAgg')
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

data_1 = [13.357,14.928,14.896,15.297,14.82,12.067,14.824,13.865,17.447]
n_1 = len(data_1)
estimated_mean_1 = np.mean(data_1)
estimated_variance_1 = stats.tvar(data_1)
x_range = np.arange(
    estimated_mean_1 - 1 * sqrt(estimated_variance_1),
    estimated_mean_1 + 1 * sqrt(estimated_variance_1),
    0.01
)
y_range = stats.t.pdf(
    x=x_range,
    df=n_1-1,
    loc=estimated_mean_1,
    scale=estimated_variance_1/n_1
)
figure = plt.plot(x_range, y_range)
plt.savefig('./ex3/report/ex3_3_data_1.png')
plt.figure(0)
