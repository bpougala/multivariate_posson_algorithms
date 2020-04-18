import numpy as np
from scipy.optimize import minimize

from MultivariatePoisson import MultivariatePoisson as multi_poisson

for i in range(100):
    mvp = multi_poisson("clayton", 7.1)
    data, mean = mvp.rvs(size=(2, 100))
    pmf = mvp.parallel_pmf(data, mean)
    d_mean = np.array([np.mean(x) for x in data])
    res = minimize(mvp.log_likelihood_archimedean, np.array([1.5]),
                   (data, d_mean, "clayton"), method='Powell', options={'disp': False})
