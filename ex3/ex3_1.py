import matplotlib
matplotlib.use('TkAgg')
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

data = [
    13.357,
    14.928,
    14.896,
    15.297,
    14.82,
    12.067,
    14.824,
    13.865,
    17.447,
]
n = len(data)
estimated_mean = np.mean(data)
estimated_variance = stats.tvar(data)
x_range = np.arange(
    estimated_mean - 3 * sqrt(estimated_variance),
    estimated_mean + 3 * sqrt(estimated_variance),
    0.01
)

'''
a) What can you say about the unknown μ?
'''
y_range = stats.t.pdf(
    x=x_range,
    df=n-1,
    loc=estimated_mean,
    scale=estimated_variance/n
)
intervals = stats.t.interval(
    0.95,
    df=n-1,
    loc=estimated_mean,
    scale=estimated_variance/n
)
low, up = intervals
print('a)')
print('-- intervals:', [round(low, 3), round(up, 3)])
print('-- estimated mean:', round(estimated_mean, 3))
print('-- estimated variance:', round(estimated_variance, 3))
print('-- estimated standard deviation:', round(sqrt(estimated_variance), 3))
figure = plt.plot(x_range, y_range)
plt.savefig('./ex3/report/3_1_a_mean_student_distribution.png')

'''
b)
'''
std_y = np.std(data, ddof=1)
scale = sqrt(1 + 1/n) * std_y
y_posterior_mu = stats.t.pdf(
    x=x_range,
    df=n-1,
    loc=estimated_mean,
    scale=scale
)
figure = plt.plot(x_range, y_posterior_mu)
plt.savefig('./ex3/report/3_1_b_posterior_mean.png')
