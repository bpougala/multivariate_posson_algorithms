import numpy as np
from scipy.stats import poisson
from CopulaGenerator import CopulaGenerator
import sklearn.datasets as skd
import itertools


class MultivariatePoisson:
    cop = None
    seed = None

    def __init__(self, seed=1234):
        self.cop = CopulaGenerator()
        self.seed = seed

    def choose_family(self, family):
        switcher = {
            0: "clayton",
            1: "gumbel",
            2: "gaussian"
        }
        return switcher.get(family, None)

    def rvs(self, family, mu=None, size=1, random_state=None, cov=None):
        # Generates random samples from the given family of copulas
        arr = []
        copulas = None
        num_dim = 1
        try:
            num_dim = int(size)
            shape = (num_dim, 1000)
        except:
            shape = size
        if mu is None:  # if no vars is passed, randomly generate dependence
            mu = np.random.uniform(0.1, 10, size=size)
        if family.lower() == "clayton":
            copulas = self.cop.MultiDimensionalClayton(1.5, d=shape)
        elif family.lower() == "gumbel":
            copulas = self.cop.MultiDimensionalGumbel(1.5, d=shape)
        elif family.lower() == "gaussian":
            if cov is None:
                cov = skd.make_spd_matrix(size)
            copulas = self.cop.Gaussian(cov)
        else:
            print("No valid family set. Defaulted to the Clayton family.")
        for i in range(num_dim):
            poiss = np.array(poisson.ppf(copulas[i], mu[i]))
            arr.append(poiss)
        return np.array(arr), mu

    def cdf(self, *args):
        for cdf in args:
            print("hey")

    def pmf(self, x, mu):
        dim = len(x)
        m = list(itertools.combinations_with_replacement([0, 1], dim))
        sum_m = np.array([sum(i) for i in m])
        sum_k = np.zeros(dim)
        for k in range(dim):
            indices = np.array(np.where(sum_m == k))
            correct_ms = np.take(m, indices.flatten(), axis=0)
            sum_fx = np.zeros(dim)
            for correct_m in correct_ms:
                sum_fx = np.add(sum_fx,
                                poisson.cdf(np.subtract(x, correct_m), mu[0]))  # sum elements from the element-wise
            # substraction between the original x input array and the values mi such that m sums up to k
            sum_k = np.add(sum_k, ((-1) ** k * sum_fx))
        return sum_k
