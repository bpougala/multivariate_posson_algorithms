import math
import sys
from functools import reduce

import numpy as np
from scipy.stats import random_correlation

import MultivariatePoisson as mvp
from MultivariatePoisson import MultivariatePoisson as mvp

results_x = list()


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


def generate_experiment_gaussian(num_dim, num_samples):
    cov = None
    mean = None
    cov_2 = np.array([[1, 0.2], [0.2, 1]])
    cov_3 = random_correlation.rvs((1.1, 0.7, 1.2))
    cov_4 = random_correlation.rvs((0.3, 1.6, 0.8, 1.3))
    cov_5 = random_correlation.rvs((0.3, 1.8, 1.2, 1.1, 0.6))
    cov_6 = random_correlation.rvs((.8, 2.0, .6, 1.5, .1, 1.0))
    if num_dim == 2:
        cov = cov_2
    elif num_dim == 3:
        cov = cov_3
    elif num_dim == 4:
        cov = cov_4
    elif num_dim == 5:
        cov = cov_5
    elif num_dim == 6:
        cov = cov_6
    else:
        raise Exception("Invalid number of dimensions chosen")
    mean = np.random.randint(0, 20, size=num_dim)
    multipoiss = mvp(cov=cov, family="gaussian")
    data = multipoiss.rvs(mu=mean, size=(cov.shape[0], num_samples))
    pmf = multipoiss.pmf(data, mean)
    cov_hat, mean_hat = multipoiss.optimise_params(data=data)
    multipoiss_hat = mvp(cov=cov_hat, family="gaussian")
    pmf_hat = multipoiss_hat.pmf(data, mean_hat)
    return kl_divergence(pmf, pmf_hat)


def kullback_leibler(pmf_x, pmf_y):
    return pmf_x * np.log10(np.divide(pmf_x, pmf_y))


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def generate_experiment(data_size, data_dimensions, family, alpha=None, iter=50, cov=None):
    if family == "clayton" or "gumbel":
        avg = 0
        j = 0
        while j < iter:
            mp = mvp(family, alpha)
            data, mean = mp.rvs(size=(data_dimensions, data_size))
            pmf = mp.pmf(data, mean)
            alpha_hat, mu = mp.optimise_params(data, mean, 11.0)
            mp_hat = mvp(family, alpha_hat[0])
            pmf_hat = mp_hat.pmf(data, mu)
            kl = kl_divergence(pmf, pmf_hat)
            if not math.isinf(kl):
                avg += kl
                j += 1
        return avg / j


def main():
    mode = sys.argv[1]
    # num_dimensions = int(sys.argv[2])
    # num_samples = int(sys.argv[3])
    # alpha = float(sys.argv[4])
    iterations = int(sys.argv[2])
    file = open("results-kl-div-gumbel.txt", "a+", buffering=1)
    if mode == "clayton" or mode == "gumbel":
        samps = [20, 40, 100, 1000]
        dims = [2, 3, 4, 5, 6]
        alphas = [1.6, 4.6, 11.6]
        for d in dims:
            for s in samps:
                for a in alphas:
                    kld = generate_experiment(s, d, mode, alpha=a, iter=iterations)
                    file.write(str(d) + " " + str(s) + " " + str(a) + " " + str(kld) + "\n")
        file.close()
    elif mode == "gaussian":
        kl = generate_experiment_gaussian(num_dimensions, num_samples)
    else:
        raise Exception("No valid copula family selected.")


if __name__ == '__main__':
    main()
