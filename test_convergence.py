from functools import reduce

from MultivariatePoisson import MultivariatePoisson as mvp

results_x = list()


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


def generate_experiment(data_size, data_dimensions, family, alpha=None, cov=None):
    if family == "clayton" or "gumbel":
        mp = mvp(family, alpha)
        data, mean = mp.rvs(size=(data_dimensions, data_size))
        x, mu = mp.optimise_params(data, mean, 11.0)
        results_x.append(x)


def main():
    for i in range(50):
        generate_experiment(40, 2, "clayton", 4.6)

    print(Average(results_x))


if __name__ == '__main__':
    main()
