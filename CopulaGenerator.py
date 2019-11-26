from scipy.stats import multivariate_normal, norm, poisson, gamma, levy_stable
import numpy as np
import math


class CopulaGenerator:
    theta = None
    copula = None
    seed = None
    alpha = None

    def __init__(self):
        self.seed = 1234

    def removeNans(self, arr):
        return arr[~np.isnan(arr)]

    def Gaussian(self, cov, lambdas=None):
        if lambdas is None:  # if no vars is passed, randomly generate dependence
            lambdas = np.random.uniform(1e-5, 10, size=cov.shape[1])
        mean = np.zeros(cov.shape[1])
        val = np.random.multivariate_normal(mean, cov, 5000).T
        arr_poisson = np.zeros((val.shape[0], 5000))
        distribution = norm()
        iter = 0
        for matrix in val:
            stats_cdf = distribution.cdf(matrix)
            poiss = np.array(poisson.ppf(stats_cdf, lambdas[iter]))
            arr_poisson[iter] = poiss
            iter += 1

        return arr_poisson

    def clayton_generator(self, t):
        return (1 + t) ** (-1 / self.alpha)

    def  MultiDimensionalClayton(self, alpha, d=(2,1000)):
      self.alpha = alpha 
      arr = []
      for i in range(d[0]):
        x = np.random.uniform(size=(d[1],))
        v = gamma.rvs(a=1/alpha, scale=1, size=(d[1],))
        u_x = self.clayton_generator(-np.log10(x) / v)
        print("Size of u_x: " + str(u_x.shape))
        arr.append(u_x)
        print("Size of arr: " + str(len(arr))) 
      
      return arr  
    def Clayton(self, alpha, d=(1000,)):
        self.alpha = alpha
        x = np.random.uniform(size=d)
        y = np.random.uniform(size=d)
        v = gamma.rvs(a=1 / alpha, scale=1, size=d)
        u_x = self.clayton_generator(-np.log10(x) / v)
        u_y = self.clayton_generator(-np.log10(y) / v)
        return u_x, u_y

    def gumbel_generator(self, t):
        return np.exp(-t ** (1 / self.alpha))

    def Gumbel(self, alpha, d=(1000,)):
        self.alpha = alpha
        x = np.random.uniform(size=d)
        y = np.random.uniform(size=d)
        beta = (math.cos(math.pi / (2 * alpha))) ** alpha
        v = levy_stable.rvs(1 / alpha, 1, scale=1, size=d)
        u_x = self.gumbel_generator(-np.log10(x) / v)
        u_y = self.gumbel_generator(-np.log10(y) / v)
        return u_x, u_y

    def frank_psi(self, u1, v2):
        numerator = np.expm1(self.alpha) * v2
        denominator = v2 * np.expm1(-self.alpha * u1) - np.exp(self.alpha * u1)
        u2 = -1 / self.alpha * np.log1p(numerator / denominator)
        return u2

    def frank2d(self, alpha, d=(1000,)):
        self.alpha = alpha
        v1 = np.random.uniform(size=d)
        v2 = np.random.uniform(size=d)
        U_1 = v1
        U_2 = self.frank_psi(U_1, v2)

        return U_1, U_2

    def hello(self):
        return "hello"

    def generate_data(self, copulas, lambdas=None):
        print("calling removeNans")
        if lambdas is None:  # if no vars is passed, randomly generate dependence
            lambdas = np.random.uniform(0.5, 6, size=len(copulas))
            firstArr = self.removeNans(copulas[0])
        arr_poisson = np.array(poisson.ppf(firstArr, lambdas[0]))
        for i in range(1, len(copulas)):
            poiss = np.array(poisson.ppf(self.removeNans(copulas[i]), lambdas[i]))
            np.concatenate((arr_poisson, poiss), axis=0)
        return arr_poisson

    def inverse_transform_frank(self, k):
        return ((1 - math.exp(-k)) ** k) / k * self.alpha

    def RLAPTRANS_implementation(self, n, b, k_max):
        u = np.sort(np.random.uniform(size=(n,)))
        # Find a value x_max such that F(x_max)
        x_start = 1
        j_max = 1
        x_max = x_start
        j = 0
        while self.inverse_transform_frank(x_max) < u[-1] and j < j_max:
            x_max = b * x_max
            j += 1
        if j == j_max:
            return 1
        x = list()
        x.append(0)
        for i in range(1, n + 1):
            x_L = x[i - 1]
            x_U = x_max
            k = 0
            t = x_L
            while (np.linalg.norm(self.inverse_transform_frank(t)) > t and k < k_max):
                k += 1
                t = t - (self.inverse_transform_frank(t) - u[i]) / self.frank_transform(t)
                if t not in range(x_L, x_U):
                    x_L = t
                else:
                    x_U = t
            if k == k_max:
                return 1
            else:
                x[i] = t

        return np.random.permutation(x)
