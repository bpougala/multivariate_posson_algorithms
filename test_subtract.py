import math

import numpy as np


def subtract(x, m):
    arr = []
    for i, elem in enumerate(x):
        temp = elem - m[i]
        arr.append(temp)

    return np.array(arr)


def multinorm(x, cov, mu=None):
    if mu is None:
        mu = np.array([0, 0])

    a = (-0.5 * (x.T - mu)) @ np.linalg.inv(cov)
    b = np.exp(a @ (x.T - mu).T)
    c = math.sqrt(2 * math.pi * np.linalg.det(cov))
    return b / c


x = np.random.uniform(0, 1, size=(2, 10))
m = np.array([2.2, 7.6])
cov = np.array([[1, 0.6], [0.6, 1]])
a = multinorm(x, cov=cov)
print(a.shape)
