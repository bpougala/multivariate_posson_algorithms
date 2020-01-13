# verify that all required packages are available on the client's machine. Otherwise, return an error message
try:
    import scipy
    import sklearn
    import matplotlib
    import numpy
except ImportError as e:
    raise SystemExit("[ERROR] Package(s) not found: Make sure the following packages are installed: scipy, scikit-learn, matplotlib, numpy.")


from CopulaGenerator import CopulaGenerator
from MultivariatePoisson import MultivariatePoisson
cop = CopulaGenerator()
poiss = MultivariatePoisson()
