import math

import numpy as np
import sklearn.datasets as skd
from scipy.stats import multivariate_normal, norm, poisson, gamma, levy_stable


class CopulaGenerator:
    family = None
    seed = None
    alpha = None
    cov = None

    def __init__(self, family, alpha=None, cov=None):
        self.seed = 1234
        self.alpha = alpha
        self.family = family
        self.cov = cov

    def cdf(self, u_i_s):
        if self.family.lower() == "clayton" and self.alpha is not None:
            dim = len(u_i_s)
            u_i_s[u_i_s == 0] = 1e-4
            u = 1 - dim + sum([u ** -self.alpha for u in u_i_s])
            return max(0, u) ** (-1 / self.alpha)
        elif self.family.lower() == "gaussian":
            print("gaussian")
        elif self.family.lower() == "gumbel" and self.alpha is not None:
            u_i_s[u_i_s == 0] = 1e-4
            u = np.array(sum([(-np.log(u_i)) ** self.alpha for u_i in u_i_s]))
            return np.exp(-u ** (1 / self.alpha))
        elif self.alpha is None:
            raise Exception("If Clayton or Gumbel copula selected, alpha parameter of CopulaGenerator cannot be null")
        else:
            raise Exception("No valid family was selected. Please select one of Clayton, Gumbel or Gaussian.")

    def pmf(self, x):
        print(x)

    def rvs(self, x):
        print(x)

    def norm_ppf(self, x):
        arr = []
        for i in range(x.shape[0]):
            arr.append(norm.ppf(i))

        return np.array(arr)

    def sin_cdf(self, x, mu):  # for marginal distributions
        arr = []
        for i in range(x.shape[0]):
            arr.append(poisson.cdf(i, mu))

        # return np.cumsum(math.exp(-mu) * np.array(arr))
        return np.array(arr)

    def removeNans(self, arr):
        return arr[~np.isnan(arr)]

    def Gaussian(self, d=(2, 1000), lambdas=None):
        if lambdas is None:  # if no vars is passed, randomly generate dependence
            lambdas = np.random.uniform(1e-5, 10, size=self.cov.shape[1])
        mean = np.zeros(self.cov.shape[1])
        val = np.random.multivariate_normal(mean, self.cov, d[1]).T
        arr_poisson = np.zeros((val.shape[0], d[1]))
        distribution = norm()
        iter = 0
        for matrix in val:
            stats_cdf = distribution.cdf(matrix)
            arr_poisson[iter] = stats_cdf
            iter += 1

        return arr_poisson

    def clayton_generator(self, t):
        return (1 + t) ** (-1 / self.alpha)

    def MultiDimensionalClayton(self, d=(2, 1000)):
        arr = []
        v = gamma.rvs(a=1 / self.alpha, scale=1, size=(d[1],))
        for i in range(d[0]):
            x = np.random.uniform(size=(d[1],))
            u_x = self.clayton_generator(-np.log10(x) / v)
            arr.append(u_x)

        return np.asarray(arr)

    def Clayton(self, x, alpha, d=(1000,)):
        s = np.random.RandomState(1234)
        self.alpha = alpha
        y = s.uniform(size=d)
        v = gamma.rvs(a=1 / alpha, scale=1, size=d, random_state=s)
        u_x = self.clayton_generator(-np.log10(x) / v)
        u_y = self.clayton_generator(-np.log10(y) / v)
        return v

    def gumbel_generator(self, t):
        return np.exp(-t ** (1 / self.alpha))

    def MultiDimensionalGumbel(self, d=(2, 1000)):
        arr = []
        beta = (math.cos(math.pi / (2 * self.alpha))) ** self.alpha
        v = levy_stable.rvs(1 / self.alpha, 1, loc=0, scale=beta, size=(d[1],))
        for i in range(d[0]):
            x = np.random.uniform(size=(d[1],))
            u_x = self.gumbel_generator(-np.log10(x) / v)
            arr.append(u_x)

        return np.asarray(arr)

    def Gumbel(self, alpha, d=(1000,)):
        self.alpha = alpha
        x = np.random.uniform(size=d)
        y = np.random.uniform(size=d)
        beta = (math.cos(math.pi / (2 * alpha))) ** alpha
        v = levy_stable.rvs(1 / alpha, 1, loc=0, scale=beta, size=d)
        u_x = self.gumbel_generator(-np.log10(x) / v)
        u_y = self.gumbel_generator(-np.log10(y) / v)
        return u_x, u_y

    def generate_data(self, copulas, lambdas=None):
        arr = []
        if lambdas is None:  # if no vars is passed, randomly generate dependence
            lambdas = np.random.uniform(0.5, 6, size=len(copulas))
        firstArr = self.removeNans(copulas[0])
        arr_poisson = np.array(poisson.ppf(firstArr, lambdas[0]))
        for i in range(len(copulas)):
            poiss = np.array(poisson.ppf(self.removeNans(copulas[i]), lambdas[i]))
            arr.append(poiss)
        return np.asarray(arr)

    def multi_cdf(self, data, mu):
        arr = []
        for i, dim in enumerate(data):
            j = self.sin_cdf(dim, mu[i])
            arr.append(j)

        if self.family.lower() == "gumbel":
            x = np.array([(-np.log(k)) ** self.alpha for k in np.array(arr)])
            y = arr[0]
            for i in range(1, len(arr)):
                y = np.add.outer(y, arr[i])
            z = y ** 1 / self.alpha
            print("shape of z: " + str(z.shape))
            return np.exp(-z)
        elif self.family.lower() == "clayton":
            x = np.array([j ** (-self.alpha) for j in np.array(arr)])
            y = arr[0]
            for i in range(1, len(arr)):
                y = np.add.outer(y, arr[i])
            z = -1 + z
            z[z < 0] = 0
            z = np.power(y, -1 / self.alpha)
            print("shape of z: " + str(z.shape))
            return z
        else:
            return 0

    def biv_cdf(self, data, mu):
        j0 = self.sin_cdf(data[0], mu[0])
        j1 = self.sin_cdf(data[1], mu[1])
        # print("j0: " + str(j0))
        if self.family.lower() == "gumbel":
            x0 = (-np.log(j0)) ** self.alpha
            x1 = (-np.log(j1)) ** self.alpha
            # print("x0: " + str(x0))
            # print("x1: " + str(x1))
            y = np.add.outer(x0, x1)
            z = y ** 1 / self.alpha
            # print("z:" + str(z))
            w = np.exp(-z)
            return w
        elif self.family.lower() == "clayton":
            x0 = j0 ** (-self.alpha)
            x1 = j1 ** (-self.alpha)
            y = np.add.outer(x0, x1)
            y = -1 + y
            y[y < 0] = 0
            z = np.power(y, -1 / self.alpha)
            return z
        else:
            x0 = norm.ppf(j0)
            x1 = norm.ppf(j1)
            arr = []
            arr.append(x0)
            arr.append(x1)
            return self.multinorm(np.array(arr), self.cov)

    def multinorm(self, x, cov, mu=None):
        if mu is None:
            mu = np.array([0, 0])

        a = (-0.5 * (x.T - mu)) @ np.linalg.inv(cov)
        b = np.exp(a @ (x.T - mu).T)
        c = math.sqrt(2 * math.pi * np.linalg.det(cov))
        return b / c

    def joint_cdf(self, data, mu=None):
        # The entire CDF is zero if at least one coordinate is 0
        if self.family.lower() == "gaussian":
            if mu is None or mu.size == 0:
                raise ValueError("You must provide a non-empty NumPy array for mu")
            dim = data.shape[1]
            num_dim = data.shape[0]
            cov = self.cov
            if cov is None:
                cov = skd.make_spd_matrix(dim)
            second_arr = []
            for i in range(num_dim):
                second_arr.append(norm.ppf(self.cdf(data[i], mu[i])))
            cdfs = np.array(second_arr)
            arr = multivariate_normal.cdf(cdfs.T, mean=mu, cov=cov)
            return arr
        elif self.family.lower() == "clayton":
            if mu is None or mu.size == 0:
                raise ValueError("You must provide a non-empty NumPy array for mu")
            dim = data.shape[1]
            num_dim = data.shape[0]
            arr = np.zeros(dim)
            gau = 1e-3

            for j in range(data.shape[0]):
                cdf = self.cdf(data[j], mu[j])
                raised_to_power = [np.maximum(x, gau) ** -self.alpha for x in
                                   cdf]  # TODO: fix the zero-raised-to-negative-power issue
                arr = np.add(arr, raised_to_power)
            arr = 1 - num_dim + arr
            arr = np.maximum(arr, gau)  # get rid of all the zero values
            pow = np.power(arr, (-1 / self.alpha))
            neg_alpha = -1 / self.alpha
            return np.array([y ** neg_alpha for y in arr])
        elif self.family.lower() == "gumbel":
            dim = data.shape[1]
            num_dim = data.shape[0]
            arr = np.zeros(dim)
            for i in range(num_dim):
                cdf = self.cdf(data[i], mu[i])
                u = [-np.log(u_i) ** self.alpha for u_i in cdf]
                # log_u = np.array([(- np.log(x)) ** self.alpha for x in cdf])
                arr = np.add(arr, u)

            arr = -(arr ** (1 / self.alpha))
            arr = np.exp(arr)

            return arr
