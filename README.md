# Algorithms for Handling Multivariate Poisson Distributions

The University of Edinburgh, School of Informatics, 2020. 
Biko Pougala. All rights reserved. 

## Project focus

Multivariate Poisson distributions that allow dependencies can be used to model various types of data such as traffic accidents, RNA sequencing or insurance claims. However, no well-defined and practical formula has been studied to model these, so we introduce a different approach that relies on a copula function. A valid multivariate distribution can be constructed by mapping a copula function to a set of marginals. This method allows to account for a large variety of dependencies between Poisson random variables. In this project we review some copula functions and develop algorithms to generate data, fit a multivariate Poisson distribution and select the optimal parameters of the copula function. We review their performance across various choices of ground truth parameters and conclude that fitting a multivariate distribution improves when the number of samples of each variable increases. We propose the Powell method as the optimal method for the convergence of the inference for margins protocol to optimise the parameters of the copula.

## Example statistical distribution
This project requires Python 3. 

To create a new Poisson distribution, run:

```python
dist = MultivariatePoisson(family=<family>, alpha=<alpha>)
```   
Where `family` is one of `clayton`, `gaussian` or `gumbel`. `alpha` is a real number that measures the strength of the dependency between data dimensions.

To generate artificial data with a known set of dimensions and data samples, then run: 

```python
data = dist.rvs(size=(d,s))
```
Where `size` is of shape `d` (number of dimensions) and `s` (number of samples along each dimension). 

To fit a multivariate probability mass function, run:
```python
multivariate_pmf = dist.pmf(data)
```
This function does a lot of things under the hood: It calculates the cumulative distribution functions of each marginals, computes the copula mapping according to the copula family and alpha value set above, then computes the final joint probability mass function using the Sylvester-Poincaré's inclusion-exclusion principle and an algorithm I came up with. 

## How to run it

The dissertation file `s1651792-dissertation.pdf` goes in detail into what this project is about. It's many pages but is not a long read at all! Also feel free to play around with the Jupyter notebooks where I carried out all my experiments (hosted on the Microsoft Azure cloud for better performance). 
