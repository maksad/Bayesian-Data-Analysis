import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
from psrf import psrf
from bioarraylp import bioassaylp

# Init all the params based on the description
sigma_a = 2
sigma_b = 10
mu_a = 0
mu_b = 10
cor = 0.5
cov_matrix = np.array([
    [sigma_a**2,                cor * sigma_a * sigma_b],
    [cor * sigma_a * sigma_b,   sigma_b**2]
])
mean = np.array([mu_a, mu_b])

doses = np.array([-0.86, -0.3, -0.05, 0.72])
deaths = np.array([0, 1, 3, 5])
number_of_animals = np.array([5, 5, 5, 5])

# reusable functions for Metropolis algorithm
def jump(theta_prev, cov):
    j = stats.multivariate_normal(theta_prev, cov)
    theta_sample = j.rvs(1)
    return np.array(theta_sample)

def ratio_can_be_accepted(ratio):
    if ratio >= 1:
        return True
    else:
        uniform_random_sample = stats.uniform(0,1).rvs(1)[0]
        if uniform_random_sample < ratio:
            return True
        else:
            return False

def get_next_theta(theta_prev, cov):
    theta_new = jump(theta_prev, cov)
    likelihood_theta_new = bioassaylp(
        theta_new[0],
        theta_new[1],
        doses,
        deaths,
        number_of_animals
    )
    likelihood_theta_prev = bioassaylp(
        theta_prev[0],
        theta_prev[1],
        doses,
        deaths,
        number_of_animals
    )

    prior_multivar_nor = stats.multivariate_normal(mean, cov_matrix)
    prior_new = prior_multivar_nor.pdf(theta_new)
    prior_prev = prior_multivar_nor.pdf(theta_prev)

    post_new = np.exp(likelihood_theta_new) * prior_new
    post_prev = np.exp(likelihood_theta_prev) * prior_prev

    ratio = post_new / post_prev

    if ratio_can_be_accepted(ratio):
        return theta_new

    return theta_prev

def trim_burnin(chains, burnin_size):
    trimmed_chains = []
    for chain in chains:
        trimmed_chains.append(chain[burnin_size:])
    return trimmed_chains

def generate_chains(sample_size, number_of_chains, burnin_size):
    chains = []
    for i in range(number_of_chains):
        starting_points = [random.randint(-2, 4), random.randint(-5, 30)]
        print('starting points', starting_points)
        chain = [starting_points]

        for j in range(sample_size):
            next_theta = get_next_theta(chain[-1], cov_matrix/10)
            chain.append(next_theta)

        chains.append(chain)
    return trim_burnin(chains, burnin_size=500)

chains = generate_chains(sample_size=3000, number_of_chains=10, burnin_size=500)

for chain in chains:
    plt.plot(
        np.array(chain)[:, 0],
        np.array(chain)[:, 1],
        alpha=0.5,
        marker='.',
        linewidth=0,
        markersize=1,
    )
plt.savefig('./ex5/report/1_scatter_plot.png', dpi=150)
plt.figure()

print('\nSingle chain')
chain = generate_chains(sample_size=10000, number_of_chains=1, burnin_size=500)[0]
print('\nPotential Scale Reduction Factor (PSRF) is: ', psrf(chain))
plt.plot(
    np.array(chain)[:, 0],
    np.array(chain)[:, 1],
    alpha=0.5,
    marker='.',
    linewidth=0,
    markersize=1,
)
plt.savefig('./ex5/report/2_scatter_plot_with_one_chain.png', dpi=150)
plt.figure()


'''Outputs:
starting points [-2, 7]
starting points [3, 23]
starting points [2, 18]
starting points [4, 16]
starting points [1, 26]
starting points [2, 24]
starting points [-1, 1]
starting points [2, -2]
starting points [-1, 14]
starting points [1, 20]

Single chain
starting points [-1, -5]

Potential Scale Reduction Factor (PSRF) is:  [1.00178023 1.00435052]
'''
