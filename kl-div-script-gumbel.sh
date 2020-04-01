eval "$(conda shell.bash hook)"
conda activate MultivariatePoisson
python test_convergence_5.py gumbel 100
