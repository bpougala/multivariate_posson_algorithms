from functools import reduce

import numpy as np

import MultivariatePoisson as mvp
from MultivariatePoisson import MultivariatePoisson as mvp

results_x = list()


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


def generate_experiment_gaussian(data_size, data_dimensions, cov=None):
    if cov is None:
        cov = np.array([[1, 0.3], [0.3, 1]])
    mean = np.array[2, 7]
    print("Running the experiment")
    multipoiss = mvp.MultivariatePoisson(cov=cov, family="gaussian")
    data = multipoiss.rvs(mu=mean, size=(data_dimensions, data_size))
    pmf = multipoiss.pmf(data, mean)
    cov_hat, mean_hat = multipoiss.optimise_params(data=data)
    multipoiss_hat = mvp.MultivariatePoisson(cov=cov_hat, family="gaussian")
    pmf_hat = multipoiss_hat.pmf(data, mean_hat)
    return kullback_leibler(pmf, pmf_hat)


def kullback_leibler(pmf_x, pmf_y):
    return pmf_x * np.log10(np.divide(pmf_x, pmf_y))


def generate_experiment(data_size, data_dimensions, family, alpha=None, cov=None):
    if family == "clayton" or "gumbel":
        mp = mvp(family, alpha)
        data, mean = mp.rvs(size=(data_dimensions, data_size))
        x, mu = mp.optimise_params(data, mean, 11.0)
        results_x.append(x)


def main():
    kl = generate_experiment_gaussian(20, 2)
    print("Kullback-Leibler value is now: " + str(kl))


if __name__ == '__main__':
    main()
