import numpy as np


def subtract(x, m):
    arr = []
    for i, elem in enumerate(x):
        temp = elem - m[i]
        arr.append(temp)

    return np.array(arr)


x = np.random.randint(1, 10, size=(4, 4))
print("x: " + str(x))
m = np.array([1, 0, 0, 1])
t = subtract(x, m)
print("t: " + str(t))
