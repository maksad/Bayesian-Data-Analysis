from scipy import stats
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

MEAN = 0.2
VARIANCE = 0.01

fig, axis = plt.subplots(1, 1)

alfa = MEAN * ( (MEAN * (1 - MEAN) / VARIANCE) - 1 )
beta = alfa * (1 - MEAN) / MEAN

x_range = numpy.linspace(0, 1, 100)
y_range = stats.beta.pdf(x_range, alfa, beta)
random_samples = stats.beta.rvs(alfa, beta, size=1000)

# a) Plot the density function of Beta-distribution
axis.plot(x_range, y_range)

# b) Take a sample of 1000 random numbers and plot a histogram
axis.hist(random_samples, density=True, alpha=0.5)
fig.savefig('prob_distribution.png')

# c) Compute the sample mean and variance from the drawn sample
sample_mean = numpy.mean(random_samples)
sample_variance = numpy.var(random_samples)
print('---------------------')
print('sample mean: ', sample_mean)
print('sample variance: ', sample_variance)

# d) Estimate the central 95%-interval from the drawn samples
sample_percentile = numpy.percentile(random_samples, q=97.5)
print('sample central percentile 95%: ', sample_percentile)
