# Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning

Source code for the paper "Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning" published at NeuRIPS 2024 (Spotlight).


## Creating the environment

A virtual Python environment can be created from the requirement file holding all the needed packages to reproduce the results of the paper.

    virtualenv -p python3.9 pess_ls
    source pess_ls/bin/activate
    pip install -r requirements.txt

## Run experiments

### Run OPE/OPS experiments

These experiments are heavily inspired by the experiments conducted in "Confident Off-Policy Evaluation and Selection through
Self-Normalized Importance Weighting" and its associated Github https://github.com/google-deepmind/offpolicy_selection_eslb.

OPE and OPS experiments are defined in the ope_ops folder, move to it if you want to run them:

To run OPE experiments, run:

    python policy_evaluation.py

To run OPS experiments, run:

    python policy_selection.py

### Run OPL experiments

OPL experiments are defined in the opl folder, move to it if you want to run them:

To run OPL experiments, run:

    python policy_learning.py
