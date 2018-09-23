from scipy import stats
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# a) summarize
x_range = numpy.arange(0, 0.65, 0.001)
prior = stats.beta.pdf(x_range, a=2, b=10)/1000
posterior = stats.beta.pdf(x_range, a=46, b=240)/1000

plt.plot(x_range, prior, label='Prior Beta(2,10)')
plt.plot(x_range, posterior, label='Posterior Beta(46,240)', color='orange')

plt.xlabel('P(algae exist) = π')
plt.legend()
plt.savefig('./ex2/report.tex/prob_distribution.png')
plt.show()

# a) P(π0= 0.2)
cumulative = stats.beta.cdf(0.2, a=46, b=240)
print('cumulative at 0.2: ', cumulative)

x_range2_line = numpy.arange(0.075, 0.3, 0.001)
posterior2_line = stats.beta.pdf(x_range2_line, a=46, b=240)/1000

x_range2 = numpy.arange(0.096, 0.2, 0.001)
posterior2 = stats.beta.pdf(x_range2, a=46, b=240)/1000

plt.fill_between(x_range2, posterior2, alpha=0.7)
plt.plot(x_range2_line, posterior2_line, color='orange')

plt.xlabel('P(algae exist) = π')
plt.legend()
plt.savefig('./ex2/report.tex/cumulative.png')
plt.show()
