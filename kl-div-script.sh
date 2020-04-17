eval "$(conda shell.bash hook)"
conda activate MultivariatePoisson
python test_convergence_1.py clayton 100
