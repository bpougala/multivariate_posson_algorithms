# HonoursProject

UG4 Project: Algorithms for Handling Multivariate Poisson Distributions

This Python module is designed to model and study multivariate Poisson distribu$

To create a new Poisson distribution, run:

```python
dist = MultivariatePoisson(family=<family>, alpha=<alpha>)
```   
Where `family` is one of `clayton`, `gaussian` or `gumbel`. `alpha` is a real number that measures the strength of the dependency between data dimensions.
