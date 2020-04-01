eval "$(conda shell.bash hook)"
conda activate MultivariatePoisson
python test_convergence.py clayton 100
