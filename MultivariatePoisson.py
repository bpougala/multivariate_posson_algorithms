import itertools
import math

import numpy as np
import sklearn.datasets as skd
from scipy.optimize import minimize, minimize_scalar
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
        return np.array(arr), mu

    def cdf(self, x, mu):  # for marginal distributions
        arr = []
        for i in range(x.shape[0]):
            arr.append((mu ** i) / math.factorial(i))

        return np.cumsum(math.exp(-mu) * np.array(arr))

    def subtract_correct_m(self, x, m):
        arr = []
        for i in range(x.shape[0]):
            if m[i] == 1:
                arr.append(x[i] - 1)
            else:
                arr.append(x[i])
        return np.array(arr)

    def mean(self, x):
        arr = []
        for dim in x:
            mu = sum(dim) / len(dim)
            arr.append(mu)
        return np.array(arr)

    def subtract_m(self, x, m):
        arr = []
        for i, elem in enumerate(x):
            temp = elem - m[i]
            arr.append(temp)

        arr = np.array(arr)
        arr[arr < 0] = 0
        return arr

    def parallel_pmf(self, x, mu):
        dim = x.shape[0]
        num_data = x.shape[1]
        m = list(itertools.product([0, 1], repeat=dim))
        sum_m = np.array([sum(i) for i in m])
        temp = self.cop.biv_cdf(x, mu)
        sum_k = np.zeros(temp.shape)
        for k in range(dim + 1):
            indices = np.array(np.where(sum_m == k))
            correct_ms = np.take(m, indices.flatten(), axis=0)
            sum_fx = np.zeros(temp.shape)
            for correct_m in correct_ms:
                sub = self.subtract_m(x, correct_m)
                mean = self.mean(sub)
                cdf = self.cop.biv_cdf(sub, mean)
                sum_fx = np.add(sum_fx, cdf)
            sum_fx = ((-1) ** k) * sum_fx
            sum_k = np.add(sum_k, sum_fx)
        return sum_k

    def biv_pmf(self, x, mu):
        dim = x.shape[0]
        num_data = x.shape[1]
        m = list(itertools.product([0, 1], repeat=dim))
        sum_m = np.array([sum(i) for i in m])
        arr = []
        sum_k = np.zeros((num_data, num_data))
        for k in range(dim + 1):
            indices = np.array(np.where(sum_m == k))
            correct_ms = np.take(m, indices.flatten(), axis=0)
            for correct_m in correct_ms:
                # print("correct_m: " + str(correct_m))
                sub = self.subtract_m(x, correct_m)
                # print("sub: " + str(sub))
                sum_fx = np.zeros((num_data, num_data))
                mean = self.mean(sub)
                cdf = self.cop.biv_cdf(sub, mean)
                sum_fx = np.add(sum_fx, cdf)
            sum_k = np.add(sum_k, (((-1) ** k) * sum_fx))
            arr.append(sum_k)
        return np.array(arr)
        # return np.array()
        # cdfs = self.cop.biv_cdf(s

    def pmf(self, x, mu):  # for multivariate distributions
        print("hello")
        dim = x.shape[0]
        num_data = x.shape[1]
        m = list(itertools.product([0, 1], repeat=num_data))
        sum_m = np.array([sum(i) for i in m])
        arr = []
        sum_k = np.zeros((num_data, num_data))
        for i in range(dim):
            for k in range(num_data + 1):
                indices = np.array(np.where(sum_m == k))
                correct_ms = np.take(m, indices.flatten(), axis=0)
                print("correct_ms: " + str(sum(correct_ms)))
                sum_fx = np.zeros((num_data, num_data))
                for correct_m in correct_ms:
                    sub = self.subtract_m(x, correct_m)
                    print("sub: " + str(sub))
                    mean = self.mean(sub)
                    cdf = self.cop.biv_cdf(sub, mean)
                    # print("cdf: " + str(cdf))
                    sum_fx = np.add(sum_fx, cdf)
                    # print("sum_fx: " + str(sum_fx))
                # substraction between the original x input array and the values mi such that m sums up to k
                sum_k = np.add(sum_k, (((-1) ** k) * sum_fx))
            arr.append(sum_k)
        return np.array(arr).flatten()

    def log_likelihood_archimedean(self, alpha, data, mean, family):
        copula = CopulaGenerator(family=family, alpha=alpha)
        poiss = MultivariatePoisson(family=family, alpha=alpha)
        pm = poiss.parallel_pmf(data, mean)
        # print("sum pmf: " + sum(pm))
        pm[pm <= 0] = 1e-13
        return -np.sum(np.log(pm))

    def log_likelihood_gaussian(self, cov, data, mean, family):
        copula = CopulaGenerator(family="gaussian", cov=cov)
        poiss = MultivariatePoisson(family=family, cov=cov)
        pm = poiss.pmf(data, mean)
        pm[pm <= 0] = 1e-3
        return -sum(np.log10(pm))

    def optimise_params(self, data, d_mean=None, start_alpha=None):
        if d_mean is None:
            d_mean = np.array([np.mean(x) for x in data])
        if self.family == "gaussian":
            cov_comb = np.corrcoef(data)
            return cov_comb, mean
        else:
            res = minimize(self.log_likelihood_archimedean, np.array([start_alpha]),
                           (data, d_mean, self.family), options={'disp': False})
            return res.x, d_mean


class Optimisation:
    x = None
    cop = None
    mu = None
    family = None

    def __init__(self, x, cop):
        self.x = x
        self.cop = cop
        self.family = cop.family
        self.mu = np.array([np.mean(i) for i in x])

    def log_likelihood(self, alpha):
        poiss = MultivariatePoisson(family=self.family, alpha=alpha)
        pm = poiss.parallel_pmf(self.x, self.mu)
        pm[pm <= 0] = 1e-13
        return np.sum(np.log(pm))

    def optimise_alpha(self, alpha):
        return minimize_scalar(self.log_likelihood)
