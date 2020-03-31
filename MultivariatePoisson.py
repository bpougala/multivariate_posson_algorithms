import itertools
import math

import numpy as np
import sklearn.datasets as skd
from scipy.optimize import minimize
from scipy.stats import poisson

from CopulaGenerator import CopulaGenerator


class MultivariatePoisson:
    cop = None
    cov = None
    seed = None
    alpha = None
    family = None

    def __init__(self, family, alpha=None, cov=None, seed=1234):
        self.alpha = alpha
        self.family = family
        self.cov = cov
        self.cop = CopulaGenerator(family=family, alpha=alpha, cov=cov)
        self.seed = seed

    def choose_family(self, family):
        switcher = {
            0: "clayton",
            1: "gumbel",
            2: "gaussian"
        }
        return switcher.get(family, None)

    def rvs(self, mu=None, size=1, random_state=None):
        # Generates random samples from the given family of copulas
        arr = []
        copulas = None
        num_dim = 1
        try:
            num_dim = int(size)
            shape = (num_dim, 100)
        except:
            shape = size
            num_dim = size[0]
        if mu is None:  # if no vars is passed, randomly generate dependence
            mu = np.random.uniform(0.1, 10, size=num_dim)
        if self.family.lower() == "clayton":
            copulas = self.cop.MultiDimensionalClayton(d=shape)
        elif self.family.lower() == "gumbel":
            copulas = self.cop.MultiDimensionalGumbel(d=shape)
        elif self.family.lower() == "gaussian":
            if self.cov is None:
                cov = skd.make_spd_matrix(num_dim)
                self.cop.cov = cov
            copulas = self.cop.Gaussian(shape)  # TODO: fix the parameters to match the ones from CopulaGenerator
        else:
            print("No valid family set. Defaulted to the Clayton family.")
        for i in range(num_dim):
            poiss = np.array(poisson.ppf(copulas[i], mu[i]), dtype=float)
            arr.append(poiss)
        return np.array(arr)

    def cdf(self, x, mu):  # for marginal distributions
        e = np.array([(math.exp(-mu) * mu ** i) / math.factorial(i) for i in x])
        return np.cumsum(e)

    def subtract_correct_m(self, x, m):
        arr = []
        for i in range(x.shape[0]):
            if m[i] == 1:
                arr.append(x[i] - 1)
            else:
                arr.append(x[i])
        return np.array(arr)

    def pmf(self, x, mu):  # for multivariate distributions
        dim = x.shape[0]
        num_data = x.shape[1]
        m = list(itertools.combinations_with_replacement([0, 1], num_data))
        sum_m = np.array([sum(i) for i in m])
        arr = []
        sum_k = np.zeros(x[0].shape)
        for k in range(num_data):
            indices = np.array(np.where(sum_m == k))
            correct_ms = np.take(m, indices.flatten(), axis=0)
            sum_fx = np.zeros(num_data)
            for correct_m in correct_ms:
                sub = self.subtract_correct_m(x, correct_m)
                cdf = self.cop.joint_cdf(sub, mu)
                sum_fx = np.add(sum_fx, cdf)
            # substraction between the original x input array and the values mi such that m sums up to k
            sum_k = np.add(sum_k, (((-1) ** k) * sum_fx))
        arr.append(sum_k)
        return np.array(arr).flatten()

    def log_likelihood_archimedean(self, alpha, data, mean, family):
        copula = CopulaGenerator(family=family, alpha=alpha)
        poiss = MultivariatePoisson(family=family, alpha=alpha)
        pm = poiss.pmf(data, mean)
        pm[pm == 0] = 1e-3
        return -sum(np.log10(pm))

    def log_likelihood_gaussian(self, cov, data, mean, family):
        copula = CopulaGenerator(family="gaussian", cov=cov)
        poiss = MultivariatePoisson(family=family, cov=cov)
        pm = poiss.pmf(data, mean)
        pm[pm == 0] = 1e-3
        return -sum(np.log10(pm))

    def optimise_params(self, data, d_mean=None, start_alpha=None):
        mean = np.array([np.mean(x) for x in data])
        if self.family == "gaussian":
            cov_comb = np.corrcoef(data)
            return cov_comb, mean
        else:
            res = minimize(self.log_likelihood_archimedean, np.array([start_alpha]),
                           (data, d_mean, self.family), options={'disp': False})
            return res.x, mean
