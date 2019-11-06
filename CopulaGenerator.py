from scipy.stats import multivariate_normal, norm, poisson
import numpy as np

class CopulaGenerator:
    theta = None
    copula = None
    seed = None
    def __init__(self):
        self.seed = 1234

    def Gaussian(self, cov, corr=None):
        if corr is None: # if no vars is passed, randomly generate dependence
            corr = np.random.uniform(1e-5, 10, size=cov.shape[1])
        mean = np.zeros(cov.shape[1])
        val = np.random.multivariate_normal(mean, cov, 5000).T
        arr_poisson = np.zeros((val.shape[0], 5000))
        distribution = norm()
        iter = 0
        for matrix in val:
            stats_cdf = distribution.cdf(matrix)
            poiss = np.array(poisson.ppf(stats_cdf, corr[iter]))
            arr_poisson[iter] = poiss
            iter += 1

        return arr_poisson
