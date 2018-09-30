import matplotlib
matplotlib.use('TkAgg')
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

x_range = np.arange(0, 0.2, 0.001)

'''
a)
'''
control = 674
control_died = 39
control_alpha = control_died + 1
control_beta = control - control_alpha + 1
control_posterior = control_alpha/control
control_pdf = stats.beta.pdf(x_range, control_alpha, control_beta)
control_pdf_line = plt.plot(x_range, control_pdf)

treatment = 680
treatment_died = 22
treatment_alpha = treatment_died + 1
treatment_beta = treatment - treatment_alpha + 1
treatment_posterior = treatment_alpha/treatment
treatment_pdf = stats.beta.pdf(x_range, treatment_alpha, treatment_beta)
treatment_pdf_line = plt.plot(x_range, treatment_pdf)

plt.legend(
    [*control_pdf_line, *treatment_pdf_line],
    ['Control group', 'Treatment group']
)
plt.savefig('./ex3/report/3_2_a_control_beta.png')
plt.figure(0)

'''
b)
'''
p_control = stats.beta.rvs(control_alpha, control_beta, size=10000)
p_treatment = stats.beta.rvs(treatment_alpha, treatment_beta, size=10000)
odd_ratio = (p_treatment/(1-p_treatment))/(p_control/(1-p_control))

plt.hist(odd_ratio, alpha=0.5, bins=40, ec='white')
plt.savefig('./ex3/report/3_2_b_histog.png')
plt.figure(0)

'''
intervals
'''
control_percentile_25 = np.percentile(p_control, q=2.5)
control_percentile_95 = np.percentile(p_control, q=97.5)

treatment_percentile_25 = np.percentile(p_treatment, q=2.5)
treatment_percentile_95 = np.percentile(p_treatment, q=97.5)
print('-----------')
print(
    'Control central percentile 95%: ',
    [round(control_percentile_25, 4), round(control_percentile_95, 4)]
)
print(
    'Treatment central percentile 95%: ',
    [round(treatment_percentile_25, 4), round(treatment_percentile_95, 4)]
)
print('Control posterior mean: ', round(np.mean(control_posterior), 4))
print('Treatment posterior mean: ', round(np.mean(treatment_posterior), 4))
