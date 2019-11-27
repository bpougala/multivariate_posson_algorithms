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
      v = gamma.rvs(a=1 / alpha, scale=1, size=(d[1],))
      for i in range(d[0]):
        x = np.random.uniform(size=(d[1],))
        u_x = self.clayton_generator(-np.log10(x) / v)
        arr.append(u_x)
      
      return np.asarray(arr)  
    def Clayton(self, x, alpha, d=(1000,)):
        s = np.random.RandomState(1234) 
        self.alpha = alpha
        #x = np.random.uniform(size=d)
        y = s.uniform(size=d)
        v = gamma.rvs(a=1 / alpha, scale=1, size=d, random_state=s)
        u_x = self.clayton_generator(-np.log10(x) / v)
        u_y = self.clayton_generator(-np.log10(y) / v)
        return v

    def gumbel_generator(self, t):
        return np.exp(-t ** (1 / self.alpha))

    def MultiDimensionalGumbel(self, alpha, d=(2,1000)):
        self.alpha = alpha 
        arr = []
        beta = (math.cos(math.pi / (2 * alpha))) ** alpha
        v = levy_stable.rvs(1 / alpha, 1, loc=0, scale=beta, size=(d[1],))
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
