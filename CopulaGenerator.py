from scipy.stats import multivariate_normal, norm, poisson, gamma, levy_stable
import numpy as np
import math
import sklearn.datasets as skd


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
        # x = np.random.uniform(size=d)
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
                second_arr.append(norm.ppf(poisson.cdf(data[i], mu[i])))
            cdfs = np.array(second_arr)
            arr = multivariate_normal.cdf(cdfs, mean=mu, cov=cov)
            print(arr)
            return arr
        elif self.family.lower() == "clayton":
            if mu is None or mu.size == 0:
                raise ValueError("You must provide a non-empty NumPy array for mu")
            dim = data.shape[1]
            num_dim = data.shape[0]
            arr = np.zeros(dim)
            gau = 1e-3

            for j in range(data.shape[0]):
                cdf = poisson.cdf(data[j], mu[j])
                raised_to_power = [np.maximum(x, gau) ** -self.alpha for x in cdf]  # TODO: fix the zero-raised-to-negative-power issue
                arr = np.add(arr, raised_to_power)
            arr = 1 - num_dim + arr
            arr = np.maximum(arr, gau) # get rid of all the zero values
            pow = np.power(arr, (-1 / self.alpha))
            neg_alpha = -1 / self.alpha
            return np.array([y ** neg_alpha for y in arr])
        elif self.family.lower() == "gumbel":
            dim = data.shape[1]
            num_dim = data.shape[0]
            arr = np.zeros(dim)
            for i in range(num_dim):
                cdf = poisson.cdf(data[i], mu[i])
                log_u = np.array([(- np.log(x)) ** self.alpha for x in cdf])
                arr = np.add(arr, log_u)

            arr = -(arr ** (1 / self.alpha))
            arr = np.exp(arr)

            return arr
